"""Calibrate captured RGB pairs with a measured printed chessboard."""

import json

import cv2
import numpy as np
from PIL import Image


def calibrate_dataset(args):
    from .cli import save

    if args.square <= 0:
        raise ValueError("Square size must be a positive measured length in metres")
    points = np.zeros((args.columns * args.rows, 3), np.float32)
    points[:, :2] = np.mgrid[: args.columns, : args.rows].T.reshape(-1, 2) * args.square
    object_points, image_points, accepted, rejected = [], [[], []], [], []
    size = None
    camera_ids = None
    input_intrinsics = None
    for case in json.loads((args.dataset / "manifest.json").read_text())["cases"]:
        folder = args.dataset / "scenes" / case["id"]
        packet = json.loads((folder / "input.json").read_text())
        ids = [c["camera_id"] for c in packet["cameras"]]
        if camera_ids is not None and ids != camera_ids:
            raise ValueError("Camera IDs/order changed during calibration")
        camera_ids = ids
        input_intrinsics = [c["K"] for c in packet["cameras"]]
        found = []
        for camera in packet["cameras"]:
            gray = np.asarray(Image.open(folder / camera["rgb"]).convert("L"))
            shape = gray.shape[::-1]
            if size is not None and shape != size:
                raise ValueError("All calibration images must have identical dimensions")
            size = shape
            ok, corners = cv2.findChessboardCornersSB(
                gray,
                (args.columns, args.rows),
                flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
            )
            if not ok:
                break
            found.append(corners.astype(np.float32))
        if len(found) != 2:
            rejected.append(case["id"])
            continue
        object_points.append(points.copy())
        accepted.append(case["id"])
        for eye in range(2):
            image_points[eye].append(found[eye])
    if len(accepted) < 10:
        raise ValueError(
            f"Need at least 10 varied chessboard pairs; accepted {len(accepted)}, "
            f"rejected {len(rejected)}"
        )
    individual = [cv2.calibrateCamera(object_points, uv, size, None, None) for uv in image_points]
    rms, k0, d0, k1, d1, rotation, translation, essential, fundamental = cv2.stereoCalibrate(
        object_points,
        *image_points,
        individual[0][1],
        individual[0][2],
        individual[1][1],
        individual[1][2],
        size,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )
    result = {
        "schema": "camera-pair-calibration-v1",
        "pixel_convention": "opencv_centres",
        "units": "metres",
        "image_size": list(size),
        "square_size_m": args.square,
        "camera_ids": camera_ids,
        "input_intrinsics": input_intrinsics,
        "cameras": [
            {"K": k.tolist(), "distortion": d.ravel().tolist()} for k, d in [(k0, d0), (k1, d1)]
        ],
        "R_right_left": rotation.tolist(),
        "t_right_left_m": translation.ravel().tolist(),
        "monocular_rms_px": [float(v[0]) for v in individual],
        "stereo_rms_px": float(rms),
        "accepted_pairs": accepted,
        "rejected_pairs": rejected,
        "scope": (
            "Calibration of supplied RGB images; wall extrinsics are"
            " estimated separately from wall markers"
        ),
    }
    save(args.out / "calibration.json", result)
    print(json.dumps(result, indent=2))
    return 0
