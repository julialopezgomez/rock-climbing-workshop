"""Operational entrypoints for challenge images and calibrated camera packets."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

from ._paths import CACHE, WEIGHTS

PACKAGE = "rcr.perception.challenge"
REPO = Path(__file__).resolve().parents[4]


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_packet(path):
    import numpy as np
    from PIL import Image

    packet = json.loads(Path(path).read_text())
    images = [
        np.asarray(Image.open(Path(path).parent / camera["rgb"]).convert("RGB"))
        for camera in packet["cameras"]
    ]
    return packet, images


def doctor(args):
    checks = []

    def check(name, operation):
        try:
            value = operation()
            checks.append({"name": name, "passed": True, "value": value})
        except Exception as error:
            checks.append({"name": name, "passed": False, "error": str(error)})

    for package in ["numpy", "scipy", "mujoco", "trimesh", "Pillow"]:
        check(package, lambda name=package: importlib.metadata.version(name))
    for name in ["rcr-seedscan", "rcr-seedverify"]:
        check(name, lambda name=name: str((CACHE / "bin" / name).stat().st_size))
    check("RCR_CAD", lambda: (CACHE / "assets/rcr/microban_hold.stl").stat().st_size)
    if args.robot:

        def model_inventory():
            from .robot_pose.model_adapter import JOINTS, RobotModel

            model = RobotModel()
            return {"path": str(model.path), "joints": JOINTS, "visual_meshes": model.model.nmesh}

        check("model", model_inventory)

        def native_libraries():
            value = subprocess.check_output(["ldd", str(CACHE / "bin/microban_m3t")], text=True)
            if "not found" in value:
                raise RuntimeError(value)
            return "All native libraries resolved"

        check("native_libraries", native_libraries)
        check("M3T", lambda: (CACHE / "bin/microban_m3t").stat().st_size)
        for name in ["torch", "ultralytics", "nvdiffrast", "pyrealsense2"]:
            check(name, lambda name=name: importlib.metadata.version(name))
        for folder in ["body", "trunk", "trunk_large"]:
            check(folder, lambda folder=folder: (WEIGHTS / folder / "best.pt").stat().st_size)
    result = {
        "passed": all(v["passed"] for v in checks),
        "cache": str(CACHE),
        "checks": checks,
        "hardware": "Device enumeration and physical accuracy require separate camera checks",
    }
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


def gpu_guard(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    arguments = []
    path_flags = {
        "--" + key.replace("_", "-"): str(value.resolve())
        for key, value in vars(args).items()
        if isinstance(value, Path)
    }
    tokens = []
    for token in sys.argv[1:]:
        tokens.extend(token.split("=", 1) if token.startswith("--") and "=" in token else [token])
    for index, token in enumerate(tokens):
        arguments.append(
            path_flags[tokens[index - 1]] if index and tokens[index - 1] in path_flags else token
        )
    command = [
        sys.executable,
        "-m",
        PACKAGE + ".gpu_guard",
        "--state",
        str(out / "gpu.json"),
        "--log",
        str(out / "run.log"),
        "--peak-mib",
        "6000",
        "--reserve-mib",
        "2000",
        "--poll-seconds",
        "1800",
        "--",
        sys.executable,
        "-m",
        PACKAGE,
        *arguments,
        "--gpu-confirmed",
    ]
    result = subprocess.run(command, cwd=REPO)
    if result.returncode:
        print((out / "run.log").read_text()[-6000:], file=sys.stderr)
    return result.returncode


def make_estimator(args):
    if args.estimator == "markers":
        from .markers import MarkerEstimator

        if not args.marker_config:
            raise ValueError(
                "Marker mode requires a measured --marker-config; see co"
                "nfig/perception/marker.example.json"
            )
        return MarkerEstimator(json.loads(args.marker_config.read_text()))
    if args.task == "live":
        from .enhanced_live import EnhancedLivePerception

        return EnhancedLivePerception(
            profile=args.profile,
            calibration=args.calibration,
            cup_method="scene_grid_dense",
            socket_refinement=not args.no_socket_prior,
            resilient=args.profile in {"accuracy", "max"},
            gpu_confirmed=True,
            sequence=args.sequence,
            fixed_cameras=args.fixed_cameras,
        )
    if args.profile == "trunk":
        from .trunk_estimator import TrunkEstimator

        return TrunkEstimator(calibration=args.calibration, sequence=args.sequence)
    if args.profile == "max":
        from .robot_pose.robust_edges import AdaptiveRobustEstimator

        cls = AdaptiveRobustEstimator
    elif args.profile == "accuracy":
        from .robot_pose.resilient_pipeline import ResilientEstimator

        cls = ResilientEstimator
    else:
        from .robot_pose.pipeline import RobotPoseEstimator

        cls = RobotPoseEstimator
    return cls(
        profile=args.profile,
        calibration=args.calibration,
        sequence=args.sequence,
        fixed_cameras=args.fixed_cameras,
        gpu_confirmed=True,
    )


def replay(args):
    from .team_frame import team_pose_record

    estimator = make_estimator(args)
    try:
        sources = (
            [args.input]
            if args.input
            else [
                args.dataset / "scenes" / c["id"] / "input.json"
                for c in json.loads((args.dataset / "manifest.json").read_text())["cases"]
            ]
        )
        failures = 0
        for source in sources:
            try:
                packet, images = load_packet(source)
                if args.camera_calibration:
                    from .capture_calibration import apply_intrinsics

                    packet = apply_intrinsics(
                        packet, json.loads(args.camera_calibration.read_text())
                    )
                value = estimator.infer(packet, images)
                robot = (
                    value["robot_pose"]
                    if args.task == "live" and args.estimator != "markers"
                    else value
                )
                team = team_pose_record(robot)
                if not robot["valid"] or value.get("map_refresh_error"):
                    failures += 1
            except (ValueError, RuntimeError) as error:
                value = {"valid": False, "reason": str(error)}
                robot = value
                team = None
                failures += 1
            result = {
                "schema": "rcr-perception-output-v1",
                "prediction": value,
                "team_pose": team,
                "source_input": str(source),
                "contact_loading_confirmed": False,
            }
            folder = args.out if args.input else args.out / source.parent.name
            save(folder / "perception.json", result)
            print(
                json.dumps({"output": str(folder / "perception.json"), "valid": robot["valid"]}),
                flush=True,
            )
    finally:
        estimator.close()
    return 1 if failures else 0


def holds(args):
    from .src.rcr_hackathon import main as infer

    args.native_generator = not args.cad_only
    args.seed_limit = 2**32
    args.hold_count = 10
    args.maximum_volumes = 3
    args.fast_verify = True
    infer(args)
    return 0


def reproduce(args):
    """Inference finishes before the expected output is read for comparison."""
    from .src.rcr_hackathon import main as infer

    manifest = json.loads((args.dataset / "manifest.json").read_text())
    records = []
    for case in manifest["cases"]:
        folder = args.dataset / case["id"]
        output = args.out / case["id"]
        config = argparse.Namespace(
            stereo=folder / "stereo.png",
            calibration=folder / "calibration.json",
            out=output,
            native_generator=True,
            seed_limit=2**32,
            hold_count=10,
            maximum_volumes=3,
            fast_verify=True,
            seed_backend=args.seed_backend,
            gpu_confirmed=True,
        )
        try:
            infer(config)
            prediction = json.loads((output / "holds.json").read_text())
            expected = json.loads((folder / "expected_holds.json").read_text())
            records.append(
                {
                    "case": case["id"],
                    "exact": prediction == expected,
                    "predicted_holds": len(prediction),
                }
            )
        except Exception as error:
            records.append({"case": case["id"], "exact": False, "error": str(error)})
    result = {
        "passed": all(r["exact"] for r in records),
        "cases": len(records),
        "records": records,
        "scope": "Replay of published benchmark examples; not a new blind evaluation",
    }
    save(args.out / "summary.json", result)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("doctor")
    p.add_argument("--robot", action="store_true")
    for name in ["holds", "reproduce"]:
        p = sub.add_parser(name)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--seed-backend", choices=["cpu", "cuda"], default="cuda")
        p.add_argument("--gpu-confirmed", action="store_true", help=argparse.SUPPRESS)
        if name == "holds":
            p.add_argument("--stereo", type=Path, required=True)
            p.add_argument("--calibration", type=Path, required=True)
            p.add_argument("--cad-only", action="store_true")
        else:
            p.add_argument("--dataset", type=Path, default=REPO / "data/perception/holds")
    for name in ["replay", "live"]:
        p = sub.add_parser(name)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--estimator", choices=["learned", "markers"], default="learned")
        p.add_argument("--task", choices=["robot", "live"], default="robot")
        p.add_argument(
            "--profile",
            choices=["fast", "accuracy", "max", "trunk"],
            default="max" if name == "replay" else "fast",
        )
        p.add_argument("--calibration", choices=["aruco", "boundary"], default="aruco")
        p.add_argument("--marker-config", type=Path)
        p.add_argument("--camera-calibration", type=Path)
        p.add_argument("--max-output-age", type=float, default=0.25)
        p.add_argument("--sequence", action="store_true", default=name == "live")
        p.add_argument("--fixed-cameras", action="store_true")
        p.add_argument("--no-socket-prior", action="store_true")
        p.add_argument("--gpu-confirmed", action="store_true", help=argparse.SUPPRESS)
        if name == "replay":
            group = p.add_mutually_exclusive_group(required=True)
            group.add_argument("--input", type=Path)
            group.add_argument("--dataset", type=Path)
        else:
            p.add_argument("--serial", action="append")
            p.add_argument("--width", type=int, default=1280)
            p.add_argument("--height", type=int, default=720)
            p.add_argument("--fps", type=int, default=30)
            p.add_argument("--count", type=int, default=0)
            p.add_argument("--telemetry-log", type=Path)
    p = sub.add_parser("capture")
    p.add_argument("--list", action="store_true")
    p.add_argument("--out", type=Path)
    p.add_argument("--serial", action="append")
    p.add_argument("--count", type=int, default=1)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--max-skew-ms", type=float, default=15)
    p.add_argument("--timeout", type=float, default=15)
    p.add_argument("--telemetry-log", type=Path)
    p = sub.add_parser("calibrate")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--columns", type=int, default=9)
    p.add_argument("--rows", type=int, default=6)
    p.add_argument(
        "--square", type=float, required=True, help="Printed chessboard square size, metres"
    )
    sub.choices["capture"].add_argument("--camera-calibration", type=Path)
    args = parser.parse_args()
    if args.command in {"replay", "live"}:
        if args.task == "live" and (args.profile == "trunk" or args.estimator == "markers"):
            parser.error(
                "Combined cup mapping needs a learned known-joint profile; "
                "use --task robot for trunk/markers"
            )
    if (
        args.command in {"holds", "reproduce", "replay", "live"}
        and not args.gpu_confirmed
        and getattr(args, "estimator", "learned") != "markers"
    ):
        return gpu_guard(args)
    if args.command == "doctor":
        return doctor(args)
    if args.command == "capture":
        from .realsense_capture import main as capture

        capture(args)
        if args.camera_calibration and not args.list:
            from .capture_calibration import apply_intrinsics

            spec = json.loads(args.camera_calibration.read_text())
            for case in json.loads((args.out / "manifest.json").read_text())["cases"]:
                path = args.out / "scenes" / case["id"] / "input.json"
                save(path, apply_intrinsics(json.loads(path.read_text()), spec))
        return 0
    if args.command == "calibrate":
        from .camera_calibration import calibrate_dataset

        return calibrate_dataset(args)
    if args.command == "live":
        from .online import live

        return live(args)
    return {"holds": holds, "reproduce": reproduce, "replay": replay}[args.command](args)
