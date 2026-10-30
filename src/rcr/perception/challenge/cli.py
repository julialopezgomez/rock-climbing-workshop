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
    import hashlib

    assets = json.loads((Path(__file__).parent / "assets/downloads.json").read_text())
    for name, spec in assets.items():

        def verify_asset(name=name, spec=spec):
            path = CACHE / "assets/rcr" / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
                raise ValueError("Asset checksum differs; rerun perception setup")
            return path.stat().st_size

        check(name, verify_asset)
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
        for folder in ["body"]:
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
    if args.task == "live":
        from .enhanced_live import EnhancedLivePerception

        return EnhancedLivePerception(
            profile=args.profile,
            calibration=args.calibration,
            socket_refinement=not args.no_socket_prior,
            gpu_confirmed=True,
            sequence=args.sequence,
            fixed_cameras=args.fixed_cameras,
        )
    if args.profile == "max":
        from .robot_pose.robust_edges import AdaptiveRobustEstimator

        cls = AdaptiveRobustEstimator
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
                robot = value["robot_pose"] if args.task == "live" else value
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


def hold_prediction(stereo, calibration_path, out, wall_width=None, wall_height=None):
    import time

    import numpy as np
    from PIL import Image

    from .src.hold_constraints import recover_scene
    from .src.hold_scene import infer
    from .src.rcr_contract import write_prediction

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for filename in ["holds.json", "prediction.json", "debug.json", "image_holds.json"]:
        (out / filename).unlink(missing_ok=True)
    image = np.asarray(Image.open(stereo).convert("RGB"))
    calibration = json.loads(Path(calibration_path).read_text())
    if image.shape[1] % 2:
        raise ValueError("Stereo PNG must contain two equal-width panels")
    width = image.shape[1] // 2
    if (width, image.shape[0]) != (calibration["image_width"], calibration["image_height"]):
        raise ValueError("Stereo dimensions differ from the declared calibration")
    if wall_width is not None or wall_height is not None:
        wall = calibration.pop("wall", {})
        calibration["wall_width_m"] = (
            wall_width
            if wall_width is not None
            else wall.get("width", calibration.get("wall_width_m", 0.8))
        )
        calibration["wall_height_m"] = (
            wall_height
            if wall_height is not None
            else wall.get("height", calibration.get("wall_height_m", 1.4))
        )
    started = time.perf_counter()
    predicted, debug = infer([image[:, :width], image[:, width:]], calibration)
    save(out / "image_holds.json", predicted)
    debug["image_fit_runtime_s"] = debug["runtime_s"]
    predicted, debug["scene_recovery"] = recover_scene(predicted, debug, calibration)
    debug["runtime_s"] = time.perf_counter() - started
    debug["timing_scope"] = "Image fitting and source-geometry scene recovery"
    volumes = [
        dict(
            name=f"Volume_{i}",
            center=dict(x=v["parameters"][0], y=v["parameters"][1]),
            baseWidth=v["parameters"][2],
            baseHeight=v["base_height"],
            depth=v["parameters"][3],
        )
        for i, v in enumerate(debug["volumes"])
    ]
    write_prediction(out, predicted, calibration, 0, debug, volumes=volumes)
    return predicted, debug


def holds(args):
    predicted, debug = hold_prediction(
        args.stereo, args.calibration, args.out, args.wall_width, args.wall_height
    )
    print(
        json.dumps(
            dict(
                output=str(args.out),
                predicted_holds=len(predicted),
                runtime_s=debug["runtime_s"],
                source_exact=False,
            )
        )
    )
    return 0


def evaluate_hold_case(task):
    dataset, output, case, wall_width, wall_height = task
    folder = Path(dataset) / case["id"]
    out = Path(output) / case["id"]
    out.mkdir(parents=True, exist_ok=True)
    for filename in ["holds.json", "prediction.json", "debug.json", "metrics.json", "failure.json"]:
        (out / filename).unlink(missing_ok=True)
    try:
        predicted, debug = hold_prediction(
            folder / "stereo.png", folder / "calibration.json", out, wall_width, wall_height
        )
        # Reference data is opened only after inference and prediction serialization.
        from .src.rcr_contract import score_holds

        expected = json.loads((folder / "expected_holds.json").read_text())
        metrics = score_holds(predicted, expected)
        strict = sum(
            m["translation_error_mm"] < 5 and m["rotation_error_deg"] < 15
            for m in metrics["matches"]
        )
        metrics.update(
            case=case["id"],
            strict_poses=strict,
            completed=True,
            runtime_s=debug["runtime_s"],
            source_exact=False,
        )
        save(out / "metrics.json", metrics)
        return metrics
    except Exception as error:
        try:
            count = len(json.loads((folder / "expected_holds.json").read_text()))
        except Exception:
            count = None
        record = dict(
            case=case["id"],
            completed=False,
            error=str(error),
            gt_holds=count,
            tp=0,
            fp=0,
            fn=count,
            strict_poses=0,
        )
        save(out / "failure.json", record)
        return record


def reproduce(args):
    import os
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    manifest = json.loads((args.dataset / "manifest.json").read_text())
    cases = [case for case in manifest["cases"] if not args.only or case["id"] in args.only]
    if not cases:
        raise ValueError("No replay cases selected")
    tasks = [
        (str(args.dataset), str(args.out), case, args.wall_width, args.wall_height)
        for case in cases
    ]
    records = []
    if args.workers == 1:
        for task in tasks:
            record = evaluate_hold_case(task)
            records.append(record)
            print(json.dumps({key: record[key] for key in ["case", "completed"]}), flush=True)
    else:
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
            for record in pool.map(evaluate_hold_case, tasks):
                records.append(record)
                print(json.dumps({key: record[key] for key in ["case", "completed"]}), flush=True)
    valid = [record for record in records if record["completed"]]
    result = dict(
        cases=len(records),
        all_inference_completed=len(valid) == len(records),
        ground_truth_holds=sum(r.get("gt_holds") or 0 for r in records),
        denominator_complete=all(r.get("gt_holds") is not None for r in records),
        matched=sum(r["tp"] for r in valid),
        false_positives=sum(r["fp"] for r in valid),
        strict_poses=sum(r["strict_poses"] for r in valid),
        records=records,
        source_exact=False,
        scope="Saved-scene evaluation; completion is distinct from pose success",
    )
    save(args.out / "summary.json", result)
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))
    return 0 if result["all_inference_completed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("doctor")
    p.add_argument("--robot", action="store_true")
    for name in ["holds", "reproduce"]:
        p = sub.add_parser(name)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--wall-width", type=float)
        p.add_argument("--wall-height", type=float)
        p.add_argument("--gpu-confirmed", action="store_true", help=argparse.SUPPRESS)
        if name == "holds":
            p.add_argument("--stereo", type=Path, required=True)
            p.add_argument("--calibration", type=Path, required=True)
        else:
            p.add_argument("--dataset", type=Path, default=REPO / "data/perception/holds")
            p.add_argument("--workers", type=int, choices=range(1, 5), default=1)
            p.add_argument("--only", nargs="*")
    for name in ["replay", "live"]:
        p = sub.add_parser(name)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument(
            "--task", choices=["robot", "live"], default="live" if name == "live" else "robot"
        )
        p.add_argument(
            "--profile",
            choices=["fast", "max"],
            default="max" if name == "replay" else "fast",
        )
        p.add_argument("--calibration", choices=["aruco", "boundary"], default="aruco")
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
    if args.command in {"holds", "reproduce", "replay", "live"} and not args.gpu_confirmed:
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
