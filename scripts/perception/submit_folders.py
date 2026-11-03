#!/usr/bin/env python3
"""Process real/*.zip and sim/*.zip together within one eight-minute budget."""

import argparse
import hashlib
import math
import os
import signal
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import submit


def inventory(input_root):
    """Record image members without opening supplied answer JSON files."""
    groups, internal_names = [], 0
    for category in ("real", "sim"):
        directory = input_root / category
        if not directory.is_dir():
            continue
        archives = sorted(
            p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".zip"
        )
        for path in archives:
            images, names, predictions = [], set(), set()
            with zipfile.ZipFile(path) as archive:
                for member in sorted(archive.infolist(), key=lambda item: item.filename):
                    name = PurePosixPath(member.filename)
                    if (
                        member.is_dir()
                        or name.suffix.lower() not in submit.IMAGE_SUFFIXES
                        or name.name.startswith("._")
                        or "__MACOSX" in name.parts
                    ):
                        continue
                    if name.is_absolute() or ".." in name.parts or "\\" in member.filename:
                        raise ValueError(f"Unsafe archive image path: {path}/{member.filename}")
                    prediction = str(name.with_name(submit.submission_name(name.name)))
                    if member.filename in names or prediction in predictions:
                        raise ValueError(f"Duplicate image or prediction path in {path}: {name}")
                    names.add(member.filename)
                    predictions.add(prediction)
                    internal = f"image_{internal_names:06d}"
                    internal_names += 1
                    images.append(
                        dict(
                            image=member.filename,
                            json=prediction,
                            internal_image=internal + name.suffix.lower(),
                            internal_json=internal + ".json",
                        )
                    )
            if images:
                groups.append(
                    dict(category=category, source=str(path), archive=path.name, images=images)
                )
    if not groups:
        raise ValueError("No stereo images found in input_root/real/*.zip or input_root/sim/*.zip")
    return groups


def runner_args(args, count, budget, work):
    """Share the deadline across image workers and optional native recovery."""
    native_enabled = not getattr(args, "no_native", False)
    if native_enabled:
        fit_seconds = min(args.seconds_per_sample, budget / count)
        default_engine = "rcr.perception.challenge.cpu_frontend:hold_prediction"
    else:
        batches = math.ceil(count / args.workers)
        fit_seconds = min(args.seconds_per_sample, max(0.1, (budget - 15) / batches))
        default_engine = "rcr.perception.challenge.photo_frontend:image_only_hold_prediction"
    return SimpleNamespace(
        inputs=[work / "images"],
        out=work / "inference",
        check=False,
        seconds=budget,
        workers=args.workers,
        native_workers=getattr(args, "native_workers", 8),
        native_hold_count=30 if native_enabled else None,
        native_reserve=min(90.0, budget / 4) if native_enabled else 0.0,
        case_seconds=fit_seconds,
        retry_partial=getattr(args, "retry_partial", False),
        engine=getattr(args, "engine", None) or default_engine,
        gpu_fallback=native_enabled,
        gpu_fallback_engine="rcr.perception.challenge.photo_frontend:budgeted_hold_prediction",
        wall_width=3.0,
        wall_height=1.4,
        camera_height=0.7,
        distance=1.0,
        baseline=0.064,
        fov=74.0,
    )


def prepare(groups, work, settings, deadline):
    from PIL import Image
    from prepare_samples import calibration

    from rcr.perception.challenge.src.rcr_contract import versioned_prediction

    images = work / "images"
    images.mkdir(parents=True)
    for group in groups:
        with zipfile.ZipFile(group["source"]) as archive:
            for scene in group["images"]:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Input preparation exceeded the shared deadline")
                destination = images / scene["internal_image"]
                destination.write_bytes(archive.read(scene["image"]))
                with Image.open(destination) as image:
                    if image.width % 2:
                        raise ValueError(f"Stereo panels must have equal widths: {scene['image']}")
                    rig = calibration(image.width // 2, image.height, settings)
                submit.write_json(
                    work / "fallback" / scene["internal_json"],
                    versioned_prediction({}, rig, 0),
                )
    submit.write_json(work / "scene_map.json", {"archives": groups})


def publisher(groups, out, work, started, settings):
    signatures = {}

    def publish(report, force=False):
        statuses = {case["filename"]: case for case in report.get("cases", [])}
        uncertain = set(report.get("uncertain_cases", []))
        archives = []
        for group in groups:
            predictions = []
            for scene in group["images"]:
                source = work / "inference/json" / scene["internal_json"]
                if not source.exists():
                    source = work / "fallback" / scene["internal_json"]
                predictions.append(source.read_bytes())
            signature = tuple(hashlib.sha256(value).hexdigest() for value in predictions)
            destination = out / group["category"] / group["archive"]
            if force or signatures.get(str(destination)) != signature:
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(".zip.tmp")
                with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
                    for scene, prediction in zip(group["images"], predictions, strict=True):
                        archive.write(work / "images" / scene["internal_image"], scene["image"])
                        archive.writestr(scene["json"], prediction)
                temporary.replace(destination)
                signatures[str(destination)] = signature
            archives.append(
                dict(
                    category=group["category"],
                    archive=group["archive"],
                    output=str(destination.resolve()),
                    total=len(group["images"]),
                    inferred=sum(
                        statuses.get(scene["internal_json"], {}).get("status")
                        not in {None, "empty_fallback", "empty_prediction"}
                        for scene in group["images"]
                    ),
                    uncertain_jsons=[
                        scene["json"]
                        for scene in group["images"]
                        if scene["internal_json"] in uncertain
                    ],
                )
            )
        submit.write_json(
            out / "perception_manifest.json",
            dict(
                state=report.get("state", "prepared"),
                elapsed_s=time.monotonic() - started,
                budget_s=settings.seconds,
                image_fit_timeout_s=settings.case_seconds,
                native_reserve_s=settings.native_reserve,
                native_workers=settings.native_workers,
                native_enabled=bool(settings.native_hold_count),
                engine=settings.engine,
                total=sum(g["total"] for g in archives),
                inferred=report.get("inferred", 0),
                native_recovered=report.get("native_recovered", 0),
                archives=archives,
                errors=report.get("errors", []),
            ),
        )

    return publish


def run(args, started):
    groups = inventory(args.input_root)
    count = sum(len(group["images"]) for group in groups)
    budget = min(480.0, args.seconds, args.seconds_per_sample * count)
    if budget <= 5:
        raise ValueError("The shared budget must exceed five seconds")
    work = args.out / "_perception_batch"
    collisions = [work, args.out / "perception_manifest.json"] + [
        args.out / group["category"] / group["archive"] for group in groups
    ]
    if any(path.exists() for path in collisions):
        raise FileExistsError("Delivery or work files already exist; use a fresh --out directory")
    settings = runner_args(args, count, budget, work)
    prepare(groups, work, settings, started + budget - 5)
    publish = publisher(groups, args.out, work, started, settings)
    report = dict(
        state="prepared",
        cases=[],
        uncertain_cases=[scene["internal_json"] for group in groups for scene in group["images"]],
    )

    def changed(value):
        nonlocal report
        report = value
        publish(report)

    publish(report)
    print(
        f"{count} images in {len(groups)} archives; shared budget {budget:.0f}s; "
        f"image-fit cap {settings.case_seconds:.1f}s; "
        f"native-search reserve {settings.native_reserve:.1f}s.",
        flush=True,
    )
    try:
        if time.monotonic() >= started + budget - 5:
            report = {
                **report,
                "state": "partial",
                "errors": [{"reason": "Input preparation used the inference budget"}],
            }
            return 2
        return submit.run(settings, started, on_package=changed)
    except BaseException as error:
        report = {
            **report,
            "state": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            "errors": [*report.get("errors", []), {"reason": str(error)}],
        }
        raise
    finally:
        publish(report, force=True)


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path)
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--seconds",
        type=float,
        default=480,
        help="Shared budget across all archives, capped at 480 seconds",
    )
    parser.add_argument(
        "--seconds-per-sample",
        type=float,
        default=24,
        help="Budget per image and maximum image-fit time; native search is shared",
    )
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    parser.add_argument(
        "--retry-partial",
        action="store_true",
        help="Retry unfinished image fits within the same shared deadline",
    )
    parser.add_argument(
        "--no-native",
        action="store_true",
        help="Fit images and CAD only; disable all generator and seed recovery",
    )
    parser.add_argument(
        "--engine",
        help="Image estimator as module:function; image-only CAD is selected with --no-native",
    )
    parser.add_argument(
        "--native-workers",
        type=int,
        choices=range(1, 33),
        default=8,
        help="CPU workers for candidate generation and verification; default: 8",
    )
    args = parser.parse_args()
    if args.seconds <= 5 or args.seconds_per_sample <= 0:
        parser.error("Use --seconds above five and a positive --seconds-per-sample")
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["PYTHONPATH"] = (
        str(submit.REPO / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    if "RCR_PERCEPTION_HOME" not in os.environ and (submit.REPO / ".perception").is_dir():
        os.environ["RCR_PERCEPTION_HOME"] = str(submit.REPO / ".perception")
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        return run(args, started)
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError, ImportError, AttributeError) as error:
        print(f"Cannot run grouped submission: {error}", file=sys.stderr)
        return 75


if __name__ == "__main__":
    raise SystemExit(main())
