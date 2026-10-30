"""Persistent estimation from color-only RealSense pairs with explicit timing."""

import json
import time

import numpy as np

from .realsense_capture import DualColorCapture, matching_telemetry


def live(args):
    from .cli import make_estimator, save
    from .team_frame import team_pose_record

    capture = DualColorCapture(
        serials=args.serial, width=args.width, height=args.height, fps=args.fps
    )
    estimator = make_estimator(args)
    args.out.mkdir(parents=True, exist_ok=True)
    calibration = (
        json.loads(args.camera_calibration.read_text()) if args.camera_calibration else None
    )
    count = 0
    try:
        capture.start()
        with (args.out / "predictions.jsonl").open("a") as stream:
            while args.count == 0 or count < args.count:
                frames = capture.pair()
                stamp = float(np.mean([f["timestamp_s"] for f in frames]))
                packet = {
                    "schema": "robot-input-v1",
                    "timestamp_s": stamp,
                    "clock": "host_monotonic",
                    "cameras": [],
                }
                for eye, (frame, intrinsics) in enumerate(
                    zip(frames, capture.intrinsics, strict=True)
                ):
                    packet["cameras"].append(
                        {
                            "camera_id": capture.serials[eye],
                            "timestamp_s": frame["timestamp_s"],
                            "clock_domain": "host_monotonic",
                            "width": intrinsics.width,
                            "height": intrinsics.height,
                            "K": [
                                [intrinsics.fx, 0, intrinsics.ppx],
                                [0, intrinsics.fy, intrinsics.ppy],
                                [0, 0, 1],
                            ],
                            "distortion": [0.0] * 5,
                            "distortion_model": "none",
                            "pixel_convention": "opencv_centres",
                        }
                    )
                if calibration is not None:
                    from .capture_calibration import apply_intrinsics

                    packet = apply_intrinsics(packet, calibration)
                started = time.perf_counter()
                try:
                    if args.telemetry_log:
                        packet.update(matching_telemetry(args.telemetry_log, stamp))
                    value = estimator.infer(packet, [f["rgb"] for f in frames])
                    robot = (
                        value["robot_pose"]
                        if args.task == "live" and args.estimator != "markers"
                        else value
                    )
                    record = {
                        "valid": robot["valid"],
                        "prediction": value,
                        "team_pose": team_pose_record(robot),
                    }
                except (ValueError, RuntimeError) as error:
                    record = {"valid": False, "reason": str(error), "team_pose": None}
                record.update(
                    timestamp_s=stamp,
                    processing_seconds=time.perf_counter() - started,
                    age_at_output_s=time.monotonic() - stamp,
                    contact_loading_confirmed=False,
                )
                record["fresh"] = record["age_at_output_s"] <= args.max_output_age
                if not record["fresh"]:
                    record["valid"] = False
                    record["team_pose"] = None
                    record["reason"] = "Estimate is too old for the configured output-age gate"
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                save(args.out / "latest.json", record)
                print(
                    json.dumps(
                        {
                            "frame": count,
                            "valid": record["valid"],
                            "age_s": record["age_at_output_s"],
                        }
                    ),
                    flush=True,
                )
                count += 1
    except KeyboardInterrupt:
        pass
    finally:
        capture.close()
        estimator.close()
    return 0
