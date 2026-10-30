"""Encoder-free trunk landmarks from repeated crops and optional second model.

Corresponding image observations are combined before a common geometric solve;
incompatible final poses are never averaged. Model/crop choices are selected on
the original development split only.
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .configured_fit import calibrate
from .contenders.keypoint_data import image_crop
from .contract import prediction
from .keypoint_ensemble import INVARIANT_LANDMARKS, combine_landmarks
from .keypoint_frontend import crop_box
from .keypoint_pose import estimate_keypoints
from .pipeline import DEFAULT_TRUNK_WEIGHTS, DEFAULT_WEIGHTS, prepare_packet
from .resilient_pipeline import prepare_rgb
from .trunk_keypoints import NAMES, TrunkKeypoints


class TrunkEnsemble(TrunkKeypoints):
    def __init__(
        self,
        whole_weights=DEFAULT_WEIGHTS,
        trunk_weights=DEFAULT_TRUNK_WEIGHTS,
        gpu_confirmed=False,
        margins=(0.35, 0.45, 0.6),
        large_weights=None,
        rgb_correction=False,
        sequence=False,
    ):
        super().__init__(whole_weights, trunk_weights, gpu_confirmed)
        self.margins = margins
        self.networks = [self.network]
        self.hashes = [self.sha256]
        self.rgb_correction = rgb_correction
        self.sequence = sequence
        self.last_timestamp = None
        self.closed = False
        if large_weights:
            from ultralytics import YOLO

            schema = json.loads((Path(large_weights).parent / "landmarks.json").read_text())
            if tuple(schema["names"]) != NAMES:
                raise ValueError("Large checkpoint landmark schema differs")
            self.networks.append(YOLO(str(large_weights)))
            self.hashes.append(hashlib.sha256(Path(large_weights).read_bytes()).hexdigest())

    def infer(self, packet, images, calibration_method="aruco", reset=False):
        if self.closed:
            raise RuntimeError("Trunk ensemble is closed")
        started = time.perf_counter()
        packet, images, timing = prepare_packet(packet, images, need_joints=False)
        if (
            self.sequence
            and self.last_timestamp is not None
            and packet["timestamp_s"] <= self.last_timestamp
            and not reset
        ):
            raise ValueError("Sequence timestamps must increase")
        preprocessing = []
        if self.rgb_correction:
            images, preprocessing = zip(*(prepare_rgb(image) for image in images), strict=False)
            images = list(images)
        cameras = calibrate(images, packet["cameras"], calibration_method)
        patches = []
        origins = []
        for image, camera in zip(images, cameras, strict=False):
            patch, origin, _ = image_crop(image, camera)
            patches.append(np.ascontiguousarray(patch[..., ::-1]))
            origins.append(origin)
        whole = self.whole.network.predict(patches, imgsz=384, conf=0.05, device=0, verbose=False)
        trunk_images = []
        trunk_origins = []
        eyes = []
        for eye, (image, result, origin) in enumerate(zip(images, whole, origins, strict=False)):
            if result.keypoints is None or not len(result.keypoints.xy):
                raise ValueError("No trunk crop proposal")
            selected = int(result.boxes.conf.argmax())
            uv = result.keypoints.xy[selected].cpu().numpy()[INVARIANT_LANDMARKS] + origin
            confidence = result.keypoints.conf[selected].cpu().numpy()[INVARIANT_LANDMARKS]
            good = (confidence >= 0.5) & np.isfinite(uv).all(1)
            if good.sum() < 4:
                raise ValueError("Too few invariant points for a trunk crop")
            box = np.r_[uv[good].min(0), uv[good].max(0)]
            for margin in self.margins:
                patch, where = crop_box(image, box, margin)
                trunk_images.append(np.ascontiguousarray(patch[..., ::-1]))
                trunk_origins.append(where)
                eyes.append(eye)
        samples = [[], []]
        for network in self.networks:
            results = network.predict(trunk_images, imgsz=384, conf=0.05, device=0, verbose=False)
            for result, origin, eye in zip(results, trunk_origins, eyes, strict=False):
                if result.keypoints is None or not len(result.keypoints.xy):
                    continue
                selected = int(result.boxes.conf.argmax())
                samples[eye].append(
                    (
                        result.keypoints.xy[selected].cpu().numpy() + origin,
                        result.keypoints.conf[selected].cpu().numpy(),
                    )
                )
        observations = []
        details = []
        for values, camera in zip(samples, cameras, strict=False):
            if not values:
                continue
            uv, confidence, spread = combine_landmarks(values)
            good = (confidence >= 0.3) & np.isfinite(uv).all(1)
            if good.any():
                observations.append(
                    dict(points_trunk=self.points[good], pixels=uv[good], camera=camera)
                )
            details.append(
                dict(
                    pixels=uv[good].tolist(),
                    indices=INVARIANT_LANDMARKS[good].tolist(),
                    crop_spread_px=spread[good].tolist(),
                )
            )
        pose, info = estimate_keypoints(observations)
        if info["reprojection_median_px"] > 3:
            raise ValueError("Ensembled landmarks disagree across views")
        info.update(
            runtime_s=time.perf_counter() - started,
            landmarks=details,
            checkpoint_sha256=self.hashes,
            margins=list(self.margins),
            models=len(self.networks),
            joint_fields_read=False,
            estimated_joints=False,
            estimated_endpoints=False,
            timing=timing,
            rgb_preprocessing=list(preprocessing),
        )
        self.last_timestamp = packet["timestamp_s"]
        return prediction(
            pose,
            method="trunk_observation_ensemble",
            timestamp_s=packet["timestamp_s"],
            joints=None,
            diagnostics=info,
            inputs=[
                "RGB pair",
                "intrinsics",
                "image wall calibration",
                "trained trunk landmark models",
                "ten invariant CAD points; no joint/IMU readings",
            ],
        )

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
