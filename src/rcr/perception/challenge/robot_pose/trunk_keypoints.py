"""Dedicated high-resolution trunk landmark model; no joint readings at inference."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .configured_fit import calibrate
from .contenders.geometry import LANDMARK_NAMES, landmarks
from .contenders.keypoint_data import image_crop
from .contenders.keypoint_predict import Frontend
from .contract import prediction
from .keypoint_ensemble import INVARIANT_LANDMARKS
from .keypoint_frontend import crop_box
from .keypoint_pose import estimate_keypoints

NAMES = tuple(LANDMARK_NAMES[i] for i in INVARIANT_LANDMARKS)


class TrunkKeypoints:
    def __init__(self, whole_weights, trunk_weights, gpu_confirmed=False):
        self.whole = Frontend(whole_weights, gpu_confirmed)
        schema = json.loads((Path(trunk_weights).parent / "landmarks.json").read_text())
        if tuple(schema["names"]) != NAMES:
            raise ValueError("Trunk checkpoint landmark schema differs")
        from ultralytics import YOLO

        self.network = YOLO(str(trunk_weights))
        self.sha256 = hashlib.sha256(Path(trunk_weights).read_bytes()).hexdigest()
        self.points = landmarks(self.whole.model, np.zeros(19))[INVARIANT_LANDMARKS]

    def infer(self, packet, images, calibration_method="aruco", proposal="gray"):
        started = time.perf_counter()
        if len(images) != 2 or len(packet["cameras"]) != 2:
            raise ValueError("Two RGB observations are required")
        for image, c in zip(images, packet["cameras"], strict=False):
            if (
                image.shape != (c["height"], c["width"], 3)
                or not np.isfinite(c["timestamp_s"])
                or abs(c["timestamp_s"] - packet["timestamp_s"]) > 0.015
            ):
                raise ValueError("Image dimensions or camera timestamps differ from the packet")
        cameras = calibrate(images, packet["cameras"], calibration_method)
        patches = []
        origins = []
        if proposal == "learned":
            detections = self.whole.network.predict(
                [np.ascontiguousarray(i[..., ::-1]) for i in images],
                imgsz=1280,
                conf=0.1,
                device=0,
                verbose=False,
            )
            for image, result in zip(images, detections, strict=False):
                if not len(result.boxes):
                    raise ValueError("No whole-robot proposal")
                patch, origin = crop_box(
                    image, result.boxes.xyxy[int(result.boxes.conf.argmax())].cpu().numpy()
                )
                patches.append(patch)
                origins.append(origin)
        else:
            for image, camera in zip(images, cameras, strict=False):
                patch, origin, _ = image_crop(image, camera)
                patches.append(patch)
                origins.append(origin)
        body = self.whole.network.predict(
            [np.ascontiguousarray(p[..., ::-1]) for p in patches],
            imgsz=384,
            conf=0.05,
            device=0,
            verbose=False,
        )
        trunks = []
        trunk_origins = []
        for image, result, origin in zip(images, body, origins, strict=False):
            if result.keypoints is None or not len(result.keypoints.xy):
                raise ValueError("No trunk landmark proposal")
            selected = int(result.boxes.conf.argmax())
            uv = result.keypoints.xy[selected].cpu().numpy()[INVARIANT_LANDMARKS] + origin
            confidence = result.keypoints.conf[selected].cpu().numpy()[INVARIANT_LANDMARKS]
            good = (confidence >= 0.5) & np.isfinite(uv).all(1)
            if good.sum() < 4:
                raise ValueError("Too few invariant landmarks for the trunk crop")
            box = np.r_[uv[good].min(0), uv[good].max(0)]
            patch, where = crop_box(image, box, 0.45)
            trunks.append(np.ascontiguousarray(patch[..., ::-1]))
            trunk_origins.append(where)
        results = self.network.predict(trunks, imgsz=384, conf=0.05, device=0, verbose=False)
        observations = []
        details = []
        for result, origin, camera in zip(results, trunk_origins, cameras, strict=False):
            if result.keypoints is None or not len(result.keypoints.xy):
                continue
            selected = int(result.boxes.conf.argmax())
            uv = result.keypoints.xy[selected].cpu().numpy() + origin
            confidence = result.keypoints.conf[selected].cpu().numpy()
            good = (confidence >= 0.3) & np.isfinite(uv).all(1)
            if good.any():
                observations.append(
                    dict(points_trunk=self.points[good], pixels=uv[good], camera=camera)
                )
            details.append(
                dict(pixels=uv[good].tolist(), indices=INVARIANT_LANDMARKS[good].tolist())
            )
        pose, info = estimate_keypoints(observations)
        if info["reprojection_median_px"] > 3:
            raise ValueError("Trunk landmarks disagree across cameras")
        info.update(
            runtime_s=time.perf_counter() - started,
            checkpoint_sha256=self.sha256,
            crop_proposal=proposal,
            estimated_joints=False,
            joint_fields_read=False,
            landmarks=details,
        )
        return prediction(
            pose,
            method="dedicated_trunk_keypoints_" + proposal,
            timestamp_s=packet["timestamp_s"],
            joints=None,
            inputs=[
                "RGB pair",
                "intrinsics",
                "image wall calibration",
                "trained whole-robot crop frontend",
                "trained trunk landmarks",
                "ten invariant trunk CAD coordinates; no joint or IMU readings",
            ],
            diagnostics=info,
        )
