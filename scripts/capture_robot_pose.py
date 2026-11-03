"""Capture two USB camera frames and estimate the robot trunk pose with ArUco.

Run from the workshop root with an OpenCV build that includes aruco:

  python scripts/capture_robot_pose.py --cameras INDEX_FOR_FIRST_SERIAL INDEX_FOR_SECOND_SERIAL

By default, calibration comes from data/cameras/realsense.json. Its camera
entries are ordered to match --cameras and include each device's serial number.
Use --calibration to select a different JSON file. Optional top-level
`camera_extrinsics` contains two 3x4 wall-to-camera [R|t] matrices. `image_size` is
[width, height] and is checked against the captured frames. Without supplied
extrinsics, each frame needs ID 5 and at least two wall IDs from 1-4.

The script uses the simulator's 42 mm ID 5 marker and trunk mounting offset
from robot_pose_aruco.py. Measure those constants for a different real mount.
Reprojection RMS is an image-fit check, not a ground-truth 3D pose error.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from robot_pose_aruco import estimate_robot_pose

DEFAULT_CALIBRATION = Path(__file__).resolve().parents[1] / "data/cameras/realsense.json"


def load_calibration(path):
    data = json.loads(path.read_text())
    cameras = data.get("cameras")
    if cameras is not None:
        if len(cameras) != 2 or any(not isinstance(camera, dict) for camera in cameras):
            raise ValueError("cameras must contain two camera entries")
        serials = [camera.get("serial") for camera in cameras]
        if any(not isinstance(serial, str) or not serial for serial in serials):
            raise ValueError("Each camera needs a serial number")
        if len(set(serials)) != 2:
            raise ValueError("Camera serial numbers must be different")
        matrices = [np.asarray(camera["camera_matrix"], dtype=float) for camera in cameras]
        distortion = [camera.get("distortion_coeffs") for camera in cameras]
    else:
        serials = None
        matrices = [np.asarray(value, dtype=float) for value in data["camera_matrices"]]
        distortion = data.get("distortion_coeffs", [None, None])
    if len(matrices) != 2 or any(matrix.shape != (3, 3) for matrix in matrices):
        raise ValueError("camera_matrices must contain two 3x3 matrices")
    if any(
        not np.isfinite(matrix).all() or matrix[0, 0] <= 0 or matrix[1, 1] <= 0
        for matrix in matrices
    ):
        raise ValueError("camera_matrices need finite, positive focal lengths")
    if len(distortion) != 2:
        raise ValueError("distortion_coeffs must contain two entries")
    for value in distortion:
        if value is None:
            continue
        coefficients = np.asarray(value, dtype=float)
        if (
            coefficients.shape not in ((4,), (5,), (8,), (12,), (14,))
            or not np.isfinite(coefficients).all()
        ):
            raise ValueError("distortion_coeffs must be finite OpenCV coefficient arrays")
    distortion = [None if value is None else np.asarray(value, dtype=float) for value in distortion]
    extrinsics = data.get("camera_extrinsics")
    if extrinsics is not None and len(extrinsics) != 2:
        raise ValueError("camera_extrinsics must contain two 3x4 matrices")
    size = data.get("image_size")
    if size is not None and (
        len(size) != 2 or any(not isinstance(value, int) or value <= 0 for value in size)
    ):
        raise ValueError("image_size must be [width, height] in positive pixels")
    return matrices, distortion, extrinsics, size, serials


def capture_pair(indices, size, warmup):
    cameras = []
    try:
        for index in indices:
            camera = cv2.VideoCapture(index)
            cameras.append(camera)
            if not camera.isOpened():
                raise ValueError(f"Cannot open camera {index}")
            if size is not None:
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])

        for _ in range(warmup):
            for index, camera in zip(indices, cameras, strict=True):
                if not camera.grab():
                    raise ValueError(f"Cannot grab frame from camera {index}")

        # Grab from both devices before decoding either frame to reduce skew.
        for index, camera in zip(indices, cameras, strict=True):
            if not camera.grab():
                raise ValueError(f"Cannot grab frame from camera {index}")
        frames = []
        for index, camera in zip(indices, cameras, strict=True):
            ok, frame = camera.retrieve()
            if not ok or frame is None:
                raise ValueError(f"Cannot retrieve frame from camera {index}")
            if size is not None and (frame.shape[1], frame.shape[0]) != tuple(size):
                raise ValueError(
                    f"Camera {index} returned {frame.shape[1]}x{frame.shape[0]}, "
                    f"but calibration expects {size[0]}x{size[1]}"
                )
            frames.append(frame)
        return frames
    finally:
        for camera in cameras:
            camera.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cameras", nargs=2, type=int, required=True, metavar=("LEFT", "RIGHT"))
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--save-dir", type=Path, help="Save the captured frames for inspection")
    args = parser.parse_args()
    if args.cameras[0] == args.cameras[1]:
        parser.error("Use two different camera indices")
    if args.warmup_frames < 0:
        parser.error("--warmup-frames must be nonnegative")
    try:
        matrices, distortion, extrinsics, size, serials = load_calibration(args.calibration)
        frames = capture_pair(args.cameras, size, args.warmup_frames)
        if args.save_dir is not None:
            args.save_dir.mkdir(parents=True, exist_ok=True)
            for index, frame in zip(args.cameras, frames, strict=True):
                if not cv2.imwrite(str(args.save_dir / f"camera_{index}.png"), frame):
                    raise ValueError(f"Cannot save frame from camera {index}")
        images = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
        result = estimate_robot_pose(images, matrices, distortion, extrinsics)
    except (OSError, KeyError, TypeError, ValueError, cv2.error) as error:
        parser.exit(1, f"Pose unavailable: {error}\n")
    result["frame"] = "wall: x right, y into wall, z up"
    result["camera_indices"] = args.cameras
    if serials is not None:
        result["configured_camera_serials"] = serials
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
