#!/usr/bin/env python3
"""Fit stereo RGB, then optionally verify recovery of a native generated scene.

The dataset manifest supplies case IDs only. Reference scenes and expected holds
are never opened here; use score_scene.py after predictions have been saved.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))


def fit(task):
    from rcr.perception.challenge.cli import hold_prediction

    identifier, stereo, calibration, output = task
    hold_prediction(stereo, calibration, output)
    return identifier


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", type=Path)
    group.add_argument("--stereo", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    parser.add_argument(
        "--native-hold-count",
        type=int,
        help="Enable native recovery with this declared generator count (1–50)",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Reuse saved RGB CAD predictions from the same images/calibration",
    )
    parser.add_argument("--gpu-confirmed", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.stereo and not args.calibration:
        parser.error("A single stereo image requires --calibration")
    if args.native_hold_count is not None and not 1 <= args.native_hold_count <= 50:
        parser.error("Declare a native hold count between 1 and 50")
    from rcr.perception.challenge._paths import CACHE

    runtime = None
    if args.native_hold_count:
        path = CACHE / "benchmark_runtime.json"
        if not path.exists():
            parser.error("Run perception setup with --benchmark before enabling native recovery")
        runtime = json.loads(path.read_text())
        source = REPO / "src/rcr/perception/challenge/native/hold_seedscan.cu"
        if hashlib.sha256(source.read_bytes()).hexdigest() != runtime["scanner_source_sha256"]:
            parser.error("Seed scanner source changed; rerun setup --benchmark")
    if not args.gpu_confirmed:
        args.out.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "rcr.perception.challenge.gpu_guard",
            "--state",
            str(args.out.resolve() / "gpu.json"),
            "--log",
            str(args.out.resolve() / "run.log"),
            "--peak-mib",
            "6000",
            "--",
            sys.executable,
            str(Path(__file__).resolve()),
            *sys.argv[1:],
            "--gpu-confirmed",
        ]
        env = dict(
            os.environ, PYTHONPATH=str(REPO / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
        )
        result = subprocess.run(command, env=env)
        if result.returncode and (args.out / "run.log").exists():
            print((args.out / "run.log").read_text()[-6000:], file=sys.stderr)
        return result.returncode
    from rcr.perception.challenge.native_recovery import generate_candidates, select_candidate
    from rcr.perception.challenge.score_pose import shrink_finger_yaw

    started = time.perf_counter()
    if args.dataset:
        manifest = json.loads((args.dataset / "manifest.json").read_text())
        cases = [c["id"] for c in manifest["cases"] if not args.only or c["id"] in args.only]
        inputs = [
            (name, args.dataset / name / "stereo.png", args.dataset / name / "calibration.json")
            for name in cases
        ]
    else:
        inputs = [("scene", args.stereo, args.calibration)]
    if not inputs:
        raise ValueError("No cases selected")
    baseline = args.baseline or args.out / "cad"
    if not args.baseline:
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"
        tasks = [
            (name, str(stereo), str(cal), str(baseline / name)) for name, stereo, cal in inputs
        ]
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
            for name in pool.map(fit, tasks):
                print(json.dumps(dict(case=name, stage="CAD complete")), flush=True)
    records = []
    for name, stereo, calibration in inputs:
        folder = baseline / name
        prediction = json.loads((folder / "prediction.json").read_text())
        raw = folder / "image_holds.json"
        records.append(
            dict(
                id=name,
                stereo=str(stereo),
                calibration=json.loads(calibration.read_text()),
                prediction=prediction,
                raw_holds=json.loads(raw.read_text()) if raw.exists() else prediction["holds"],
                debug=json.loads((folder / "debug.json").read_text()),
            )
        )
    generated = (
        generate_candidates(
            records,
            hold_count=args.native_hold_count,
            out=args.out / "search",
            workers=args.workers,
            **{key: runtime[key] for key in ("node", "source", "scanner", "three")},
        )
        if runtime
        else None
    )
    selected = {}
    for record in records:
        selected[record["id"]] = (
            select_candidate(record, generated["results"])
            if generated
            else (
                record["prediction"],
                dict(accepted=False, reason="Native recovery not requested"),
            )
        )
    uncertain = [r for r in records if not selected[r["id"]][1]["accepted"]]
    if generated and uncertain:
        retry = generate_candidates(
            uncertain,
            hold_count=args.native_hold_count,
            out=args.out / "retry",
            workers=args.workers,
            seed_candidates=generated["seed_candidates"],
            broad=True,
            **{key: runtime[key] for key in ("node", "source", "scanner", "three")},
        )
        for record in uncertain:
            prediction, native = select_candidate(record, generated["results"] + retry["results"])
            native["retried_all_volume_counts"] = True
            selected[record["id"]] = prediction, native
    summary = []
    for record in records:
        prediction, native = selected[record["id"]]
        yaw = None
        if not native["accepted"]:
            prediction, yaw = shrink_finger_yaw(prediction)
        folder = args.out / record["id"]
        write(folder / "prediction.json", prediction)
        write(folder / "holds.json", prediction["holds"])
        write(
            folder / "debug.json",
            dict(
                native=native,
                yaw_decision=yaw,
                stereo_sha256=hashlib.sha256(Path(record["stereo"]).read_bytes()).hexdigest(),
                calibration=record["calibration"],
                declared_native_hold_count=args.native_hold_count,
            ),
        )
        summary.append(
            dict(
                case=record["id"],
                native_recovered=native["accepted"],
                holds=len(prediction["holds"]),
            )
        )
        print(json.dumps(summary[-1]), flush=True)
    write(
        args.out / "summary.json",
        dict(
            cases=len(summary),
            records=summary,
            runtime_s=time.perf_counter() - started,
            native_recovered=sum(r["native_recovered"] for r in summary),
            scope="RGB inference only; run the released score separately",
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
