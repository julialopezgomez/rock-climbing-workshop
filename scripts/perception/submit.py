#!/usr/bin/env python3
"""Create competition JSON files and a ZIP from stereo images, without reading labels.

The default budget is nine minutes from command startup, leaving one minute for
upload. Completed predictions are published while remaining images are running.
"""

import argparse
import datetime as dt
import fcntl
import hashlib
import importlib
import io
import json
import os
import re
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def submission_name(filename):
    stem = Path(filename).stem
    stem = re.sub(r"_cam1_(\d+)$", r"_\1", stem)
    if not re.fullmatch(r"[A-Za-z0-9_. -]+", stem) or stem in {".", ".."}:
        raise ValueError(f"Unsupported image filename: {filename}")
    return stem + ".json"


def image_members(paths):
    """Read image bytes only, even when input archives contain answer JSON."""

    def is_metadata(path):
        return path.name.startswith("._") or "__MACOSX" in path.parts

    def is_image(path):
        return path.suffix.lower() in IMAGE_SUFFIXES and not is_metadata(path)

    for path in paths:
        if path.is_dir():
            images = sorted(p for p in path.rglob("*") if p.is_file() and is_image(p))
            archives = [p for p in path.rglob("*.zip") if p.is_file() and not is_metadata(p)]
            files = sorted([*images, *archives])
            yield from image_members(files)
        elif path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                for member in sorted(archive.infolist(), key=lambda x: x.filename):
                    if not member.is_dir() and is_image(Path(member.filename)):
                        yield f"{path.name}/{member.filename}", archive.read(member)
        elif is_image(path):
            yield str(path), path.read_bytes()
        else:
            raise ValueError(f"Expected an image, ZIP or directory: {path}")


def prepare_inputs(paths, out, args):
    from PIL import Image
    from prepare_samples import calibration

    from rcr.perception.challenge.src.rcr_contract import versioned_prediction

    cases, names = [], set()
    for original, content in image_members(paths):
        name = submission_name(original)
        if name in names:
            raise ValueError(f"Duplicate submission filename: {name}; pass each image once")
        names.add(name)
        identifier = f"scene_{len(cases):04d}"
        folder = out / "work/input" / identifier
        folder.mkdir(parents=True)
        with Image.open(io.BytesIO(content)) as source:
            rgb = source.convert("RGB")
            if rgb.width % 2:
                raise ValueError(f"Stereo panels must have equal widths: {original}")
            rig = calibration(rgb.width // 2, rgb.height, args)
            rgb.save(folder / "stereo.png")
        write_json(folder / "calibration.json", rig)
        # An empty prediction keeps missing/failed images explicit.
        write_json(out / "json" / name, versioned_prediction({}, rig, 0))
        cases.append(
            dict(
                id=identifier,
                source=original,
                filename=name,
                input_sha256=hashlib.sha256(content).hexdigest(),
                status="empty_fallback",
            )
        )
    if not cases:
        raise ValueError("No stereo images found")
    write_json(out / "work/input/manifest.json", {"cases": [{"id": c["id"]} for c in cases]})
    return cases


def package(out, cases, started, state, **extra):
    """The upload ZIP contains only the requested prediction JSON files."""
    if extra.pop("native_enabled", False):
        uncertain = [
            case["filename"]
            for case in cases
            if not case.get("native_accepted")
            or case.get("native_pending")
            or (
                case.get("native_pass") == "initial"
                and (case.get("native_loss") is None or case["native_loss"] > 0.30)
            )
        ]
        extra["uncertain_cases"] = uncertain
        if state == "complete" and uncertain:
            state = "complete_with_warnings"
    temporary = out / "submission.zip.tmp"
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for case in cases:
            archive.write(out / "json" / case["filename"], case["filename"])
    temporary.replace(out / "submission.zip")
    report = dict(
        state=state,
        elapsed_s=time.monotonic() - started,
        created_utc=dt.datetime.now(dt.UTC).isoformat(),
        inferred=sum(c["status"] not in {"empty_fallback", "empty_prediction"} for c in cases),
        native_recovered=sum(bool(c.get("native_accepted")) for c in cases),
        total=len(cases),
        submission=str((out / "submission.zip").resolve()),
        cases=cases,
        **extra,
    )
    write_json(out / "manifest.json", report)
    return report


def publish(source, out, case, stage):
    from rcr.perception.challenge.src.rcr_contract import validate_holds

    if not source.exists():
        return False
    content = source.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if digest == case.get("prediction_sha256"):
        return False
    prediction = json.loads(content)
    if prediction.get("formatVersion") != 1:
        raise ValueError("The estimator must write a complete formatVersion:1 prediction")
    validate_holds(prediction["holds"])
    write_json(out / "json" / case["filename"], prediction)
    case.update(
        status=stage if prediction["holds"] else "empty_prediction",
        prediction_sha256=digest,
        holds=len(prediction["holds"]),
    )
    return True


def stop_process(process):
    """Only signal the process group created by this runner."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    until = time.monotonic() + 0.5
    while time.monotonic() < until:
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            process.wait()
            return
        time.sleep(0.02)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=2)


def run_worker(args):
    task = json.loads(args.worker.read_text())
    if "fit_seconds" in task:
        os.environ["RCR_IMAGE_FIT_SECONDS"] = str(task["fit_seconds"])
    try:
        import cv2
    except ImportError:
        pass
    else:
        cv2.setNumThreads(1)
    module, function = task["engine"].split(":", 1)
    getattr(importlib.import_module(module), function)(
        task["stereo"],
        task["calibration"],
        task["output"],
    )
    return 0


def gpu_preflight(args):
    from rcr.perception.challenge._paths import CACHE
    from rcr.perception.challenge.gpu_guard import sample_gpu

    lock_path = Path(os.environ.get("RCR_GPU_LOCK", CACHE / "gpu.lock"))
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("a+")
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "Another perception job owns the GPU lock; no inference started"
            ) from error
        samples = []
        for index in range(3):
            samples.append(sample_gpu())
            if index < 2:
                time.sleep(1)
        if (
            sum(s["utilization_percent"] for s in samples) / 3 >= 60
            or min(s["free_mib"] for s in samples) < 9000
        ):
            raise RuntimeError("GPU is busy; no inference started. Retry when it is available.")
        module, function = args.engine.split(":", 1)
        getattr(importlib.import_module(module), function)
        if getattr(args, "gpu_fallback", False):
            module, function = args.gpu_fallback_engine.split(":", 1)
            getattr(importlib.import_module(module), function)
        for dependency in ("mujoco", "scipy", "trimesh"):
            importlib.import_module(dependency)
        package = REPO / "src/rcr/perception/challenge"
        resources = json.loads((package / "assets/downloads.json").read_text())
        for name, spec in resources.items():
            path = CACHE / "assets/rcr" / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
                raise ValueError(f"Mesh checksum mismatch: {path}")
        if args.native_hold_count:
            runtime = json.loads((CACHE / "benchmark_runtime.json").read_text())
            for key in ("scanner", "node", "source", "three"):
                if not Path(runtime[key]).exists():
                    raise FileNotFoundError(f"Missing native resource: {runtime[key]}")
            scanner = package / "native/hold_seedscan.cu"
            if hashlib.sha256(scanner.read_bytes()).hexdigest() != runtime["scanner_source_sha256"]:
                raise ValueError("CUDA scanner source changed; rerun perception setup --benchmark")
        return lock
    except BaseException:
        lock.close()
        raise


def run(args, started, on_package=None):
    lock = gpu_preflight(args)
    if args.check:
        lock.close()
        print("Ready: runtime, GPU capacity and native resources checked; no images processed.")
        return 0
    if args.out.exists() and any(args.out.iterdir()):
        lock.close()
        raise FileExistsError(f"Use a new output directory: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    children = []
    cases = []
    state = "running"
    errors = []
    deadline = started + args.seconds - 5

    def launch(command, logfile):
        logfile.parent.mkdir(parents=True, exist_ok=True)
        with logfile.open("w") as log:
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=os.environ.copy(),
            )
        children.append(process)
        return process

    def changed(stage):
        report = package(
            args.out,
            cases,
            started,
            stage,
            errors=errors,
            native_enabled=bool(args.native_hold_count),
        )
        if on_package is not None:
            on_package(report)
        print(
            f"{report['inferred']}/{len(cases)} inferred; {report['elapsed_s']:.1f}s; "
            f"ZIP: {report['submission']}",
            flush=True,
        )

    def image_pass(selected, engine, directory, stage_deadline):
        pending, running, completed = list(selected), {}, []
        while (pending or running) and time.monotonic() < stage_deadline:
            while pending and len(running) < args.workers:
                case = pending.pop(0)
                source = args.out / "work/input" / case["id"]
                folder = directory / case["id"]
                task = folder / "task.json"
                write_json(
                    task,
                    dict(
                        engine=engine,
                        stereo=str(source / "stereo.png"),
                        calibration=str(source / "calibration.json"),
                        output=str(folder),
                        fit_seconds=args.case_seconds,
                    ),
                )
                process = launch(
                    [sys.executable, str(Path(__file__).resolve()), "--worker", str(task)],
                    folder / "run.log",
                )
                running[case["id"]] = (case, process, time.monotonic())
            for identifier, (case, process, launched) in list(running.items()):
                if process.poll() is None and time.monotonic() - launched > args.case_seconds:
                    stop_process(process)
                    errors.append(
                        dict(case=identifier, stage=directory.name, reason="image time limit")
                    )
                if process.poll() is None:
                    continue
                folder = directory / identifier
                try:
                    if process.returncode != 0:
                        raise RuntimeError(
                            f"estimator exit {process.returncode}; see {folder / 'run.log'}"
                        )
                    prediction = folder / "prediction.json"
                    if not prediction.exists():
                        raise RuntimeError("estimator produced no prediction")
                    if not (case.get("native_accepted") or case.get("native_pending")):
                        publish(prediction, args.out, case, directory.name)
                    completed.append(case)
                except (OSError, ValueError, RuntimeError, KeyError) as error:
                    errors.append(dict(case=identifier, stage=directory.name, reason=str(error)))
                del running[identifier]
                changed(directory.name)
            time.sleep(0.1)
        for case, process, _ in running.values():
            stop_process(process)
            errors.append(
                dict(case=case["id"], stage=directory.name, reason="image-stage deadline")
            )
        errors.extend(
            dict(case=c["id"], stage=directory.name, reason="not started before deadline")
            for c in pending
        )
        return completed

    def native_pass(selected, baseline, suffix, stage_deadline):
        if not args.native_hold_count or not selected or time.monotonic() >= stage_deadline:
            return
        native_data = args.out / ("work/native_input" + suffix)
        for case in selected:
            source = args.out / "work/input" / case["id"]
            cad = baseline / case["id"]
            target = native_data / case["id"]
            target.mkdir(parents=True, exist_ok=True)
            for name, candidate in [
                ("stereo.png", "native_stereo.png"),
                ("calibration.json", "native_calibration.json"),
            ]:
                canonical = cad / candidate
                (target / name).write_bytes(
                    (canonical if canonical.exists() else source / name).read_bytes()
                )
        write_json(native_data / "manifest.json", {"cases": [{"id": c["id"]} for c in selected]})
        refined = args.out / ("work/refined" + suffix)
        process = launch(
            [
                sys.executable,
                str(REPO / "scripts/perception/benchmark_max.py"),
                "--dataset",
                str(native_data),
                "--baseline",
                str(baseline),
                "--native-hold-count",
                str(args.native_hold_count),
                "--workers",
                str(args.workers),
                *(
                    ["--native-workers", str(args.native_workers)]
                    if getattr(args, "native_workers", None)
                    else []
                ),
                "--out",
                str(refined),
                "--gpu-confirmed",
            ],
            args.out / ("work/native" + suffix + ".log"),
        )
        last_progress = time.monotonic()
        phase = "GPU refinement" if suffix else "initial"
        print(f"Native verification ({phase}): {len(selected)} images.", flush=True)
        while True:
            updated = False
            for case in selected:
                folder = refined / case["id"]
                prediction = folder / "prediction.json"
                if not prediction.exists():
                    continue
                try:
                    diagnostic = folder / "debug.json"
                    native = (
                        json.loads(diagnostic.read_text()).get("native", {})
                        if diagnostic.exists()
                        else {}
                    )
                    accepted = native.get("accepted")
                    candidates = native.get("candidates", [])
                    loss = candidates[0].get("verification", {}).get("loss") if candidates else None
                    protected = case.get("native_accepted") or case.get("native_pending")
                    if not protected or accepted:
                        updated |= publish(prediction, args.out, case, "refined" + suffix)
                    if accepted:
                        case["native_accepted"] = True
                        case["native_loss"] = loss
                        case["native_pass"] = "gpu" if suffix else "initial"
                        case.pop("native_pending", None)
                    elif accepted is None:
                        # A streamed prediction can precede its acceptance receipt.
                        # Preserve it through a deadline until another verified result exists.
                        case["native_pending"] = True
                    elif not case.get("native_accepted"):
                        case.pop("native_pending", None)
                        case["native_accepted"] = False
                except (OSError, ValueError, KeyError) as error:
                    if case.get("refinement_error") != str(error):
                        case["refinement_error"] = str(error)
                        errors.append(
                            dict(case=case["id"], stage="native" + suffix, reason=str(error))
                        )
            if updated:
                changed("refining" + suffix)
                last_progress = time.monotonic()
            elif time.monotonic() - last_progress >= 10:
                remaining = max(0, deadline - time.monotonic())
                print(
                    f"Native verification ({phase}) running; "
                    f"{remaining:.0f}s remain in the budget.",
                    flush=True,
                )
                last_progress = time.monotonic()
            if process.poll() is not None:
                if process.returncode:
                    errors.append(
                        dict(stage="native" + suffix, reason=f"exit {process.returncode}")
                    )
                break
            if time.monotonic() >= stage_deadline:
                stop_process(process)
                errors.append(
                    dict(stage="native" + suffix, reason="deadline; retained completed estimates")
                )
                break
            time.sleep(0.2)

    try:
        cases = prepare_inputs(args.inputs, args.out, args)
        changed("prepared")
        adaptive = (
            args.native_hold_count
            and getattr(args, "gpu_fallback", False)
            and args.engine != args.gpu_fallback_engine
        )
        baseline = args.out / "work/cad"
        completed = image_pass(
            cases,
            args.engine,
            baseline,
            deadline - (args.native_reserve if args.native_hold_count else 0),
        )
        first_deadline = (
            min(deadline, time.monotonic() + 2 * args.native_reserve) if adaptive else deadline
        )
        native_pass(completed, baseline, "", first_deadline)
        if adaptive and time.monotonic() < deadline - args.native_reserve:
            uncertain = [
                case
                for case in cases
                if not case.get("native_accepted")
                or case.get("native_loss") is None
                or case["native_loss"] > 0.30
            ]
            if uncertain:
                changed("gpu_fallback")
                fallback = args.out / "work/gpu_cad"
                completed = image_pass(
                    uncertain, args.gpu_fallback_engine, fallback, deadline - args.native_reserve
                )
                native_pass(completed, fallback, "_gpu", deadline)
        state = (
            "complete"
            if all(c["status"] not in {"empty_fallback", "empty_prediction"} for c in cases)
            else "partial"
        )
        if state == "complete" and errors:
            state = "complete_with_warnings"
    except KeyboardInterrupt:
        state = "interrupted"
        errors.append(dict(reason="interrupted; retained completed predictions"))
    except Exception as error:
        state = "failed"
        errors.append(dict(reason=str(error)))
        raise
    finally:
        for process in children:
            stop_process(process)
        if cases:
            report = package(
                args.out,
                cases,
                started,
                state,
                errors=errors,
                native_enabled=bool(args.native_hold_count),
            )
            if on_package is not None:
                on_package(report)
            print(
                json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2), flush=True
            )
        lock.close()
    return (
        0
        if cases and all(c["status"] not in {"empty_fallback", "empty_prediction"} for c in cases)
        else 2
    )


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs", nargs="*", type=Path, help="ZIP files, image directories or images"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "outputs/perception/submission_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        ),
    )
    parser.add_argument(
        "--seconds", type=float, default=540, help="Total run budget; reserve upload time"
    )
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--native-hold-count", type=int, default=30)
    parser.add_argument("--no-native", action="store_true", help="Use image estimates only")
    parser.add_argument(
        "--no-gpu-fallback",
        action="store_true",
        help="Do not refit uncertain fast-frontend cases with GPU CAD",
    )
    parser.add_argument(
        "--gpu-fallback-engine",
        default="rcr.perception.challenge.photo_frontend:hold_prediction",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--native-reserve", type=float, default=90)
    parser.add_argument("--case-seconds", type=float, default=90)
    parser.add_argument(
        "--engine",
        default="rcr.perception.challenge.cpu_frontend:hold_prediction",
        help="Trusted image estimator module:function",
    )
    parser.add_argument("--wall-width", type=float, default=3.0)
    parser.add_argument("--wall-height", type=float, default=1.4)
    parser.add_argument("--camera-height", type=float, default=0.7)
    parser.add_argument("--distance", type=float, default=1.0)
    parser.add_argument("--baseline", type=float, default=0.064)
    parser.add_argument("--fov", type=float, default=74.0)
    parser.add_argument("--check", action="store_true", help="Check readiness without inference")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return run_worker(args)
    if not args.inputs and not args.check:
        parser.error("Supply image ZIP files or a directory")
    if args.seconds <= 5 or args.native_reserve < 0 or args.case_seconds <= 0:
        parser.error("Use a budget above five seconds and nonnegative time allowances")
    if (
        not 0 < args.fov < 180
        or min(args.wall_width, args.wall_height, args.distance, args.baseline) <= 0
    ):
        parser.error("Use positive metric dimensions and a field of view between 0 and 180 degrees")
    if not 1 <= args.native_hold_count <= 50:
        parser.error("Generator hold count must be between 1 and 50")
    if args.no_native:
        args.native_hold_count = None
    args.gpu_fallback = not args.no_gpu_fallback
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["PYTHONPATH"] = str(REPO / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
    if "RCR_PERCEPTION_HOME" not in os.environ and (REPO / ".perception").is_dir():
        os.environ["RCR_PERCEPTION_HOME"] = str(REPO / ".perception")
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        return run(args, started)
    except (OSError, ValueError, RuntimeError, ImportError, AttributeError) as error:
        print(f"Cannot run submission: {error}", file=sys.stderr)
        return 75


if __name__ == "__main__":
    raise SystemExit(main())
