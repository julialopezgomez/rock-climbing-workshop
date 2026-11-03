"""Actual learned landmark inference + known-q robust pose initialization."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from ..._gpu import require_gpu
from ..configured_fit import calibrate
from ..contract import prediction
from ..keypoint_pose import estimate_generalized, estimate_keypoints
from ..model_adapter import RobotModel
from .geometry import LANDMARK_NAMES, landmarks
from .keypoint_data import image_crop


class Frontend:
    def __init__(self, weights, gpu_confirmed):
        require_gpu(gpu_confirmed)
        schema_file = Path(weights).parent / "landmarks.json"
        if (
            not schema_file.exists()
            or tuple(json.loads(schema_file.read_text())["names"]) != LANDMARK_NAMES
        ):
            raise ValueError("Checkpoint requires its matching Microban landmarks.json")
        from ultralytics import YOLO

        self.network = YOLO(str(weights))
        self.model = RobotModel()
        self.sha256 = hashlib.sha256(Path(weights).read_bytes()).hexdigest()

    def infer(self, packet, images, calibration_method="aruco", backend="opencv"):
        start = time.perf_counter()
        if len(images) != 2 or len(packet["cameras"]) != 2:
            raise ValueError("Two RGB camera observations are required")
        for image, camera in zip(images, packet["cameras"], strict=False):
            if image.shape != (camera["height"], camera["width"], 3):
                raise ValueError("RGB dimensions differ from the calibrated pixel frame")
            if abs(camera["timestamp_s"] - packet["timestamp_s"]) > 0.015:
                raise ValueError("Camera/encoder timestamp skew exceeds 15 ms")
        cameras = calibrate(images, packet["cameras"], calibration_method)
        patches, origins = [], []
        for image, camera in zip(images, cameras, strict=False):
            patch, origin, _ = image_crop(image, camera)
            # Ultralytics numpy-image interface consumes BGR.
            patches.append(np.ascontiguousarray(patch[..., ::-1]))
            origins.append(origin)
        detected = self.network.predict(patches, imgsz=384, conf=0.05, device=0, verbose=False)
        points = landmarks(self.model, packet["joints_rad"])
        observations = []
        for result, origin, camera in zip(detected, origins, cameras, strict=False):
            if result.keypoints is None or not len(result.keypoints.xy):
                continue
            selected = int(result.boxes.conf.argmax())
            pixels = result.keypoints.xy[selected].cpu().numpy() + origin
            confidence = result.keypoints.conf[selected].cpu().numpy()
            if len(pixels) != len(points):
                raise ValueError("The checkpoint landmark schema does not match Microban")
            good = np.isfinite(pixels).all(1) & (confidence >= 0.3)
            if good.any():
                observations.append(
                    dict(points_trunk=points[good], pixels=pixels[good], camera=camera)
                )
        solver = estimate_keypoints if backend == "opencv" else estimate_generalized
        pose, diagnostic = solver(observations)
        diagnostic.update(
            runtime_s=time.perf_counter() - start,
            checkpoint_sha256=self.sha256,
            landmarks_by_camera=[len(o["pixels"]) for o in observations],
            segmentation="same image-derived crop as training; no label mask",
            orientation_source="learned 2D keypoints and geometric fitting",
        )
        return prediction(
            pose,
            method="learned_keypoints_" + backend,
            timestamp_s=packet["timestamp_s"],
            joints=packet["joints_rad"],
            inputs=[
                "RGB pair",
                "intrinsics",
                "RGB wall calibration",
                "19 measured joints",
                "public CAD kinematics",
                "trained Microban landmark network",
            ],
            diagnostics=diagnostic,
        )
