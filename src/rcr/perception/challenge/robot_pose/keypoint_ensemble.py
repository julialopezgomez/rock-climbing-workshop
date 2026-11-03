"""Crop/resolution ensemble of Microban landmarks with geometric pose fitting.

Encoder-free root estimation uses ten landmarks whose coordinates are invariant
to every joint angle: six trunk points plus the shoulder and hip joint anchors.
"""

import time

import numpy as np

from .configured_fit import calibrate, segment_robot
from .contenders.geometry import landmarks
from .contenders.keypoint_predict import Frontend
from .contract import prediction
from .keypoint_frontend import crop_box
from .keypoint_pose import estimate_generalized, estimate_keypoints

INVARIANT_LANDMARKS = np.array([4, 5, 6, 7, 8, 9, 10, 11, 14, 15])


def combine_landmarks(samples, minimum_confidence=0.3):
    positions = np.stack([s[0] for s in samples])
    confidence = np.stack([s[1] for s in samples])
    centers = []
    weights = []
    spread = []
    for index in range(positions.shape[1]):
        good = (confidence[:, index] >= minimum_confidence) & np.isfinite(positions[:, index]).all(
            1
        )
        if not good.any():
            centers.append([0.0, 0.0])
            weights.append(0.0)
            spread.append(float("inf"))
            continue
        points = positions[good, index]
        center = np.median(points, axis=0)
        deviation = np.linalg.norm(points - center, axis=1)
        # A few very poor crops must not pull a consensus away from its mode.
        keep = deviation <= max(1.0, 3 * np.median(deviation))
        centers.append(np.mean(points[keep], axis=0))
        weights.append(float(np.median(confidence[good, index][keep])))
        spread.append(float(np.median(deviation)))
    return np.asarray(centers), np.asarray(weights), np.asarray(spread)


class EnsembleKeypoints(Frontend):
    def infer(
        self,
        packet,
        images,
        calibration_method="aruco",
        joint_mode="known",
        proposal="gray",
        resolutions=(384, 512),
        margins=(0.25, 0.3, 0.35),
        backend="opencv",
    ):
        started = time.perf_counter()
        if len(images) != 2 or len(packet["cameras"]) != 2:
            raise ValueError("Two RGB camera observations are required")
        for image, camera in zip(images, packet["cameras"], strict=False):
            if image.shape != (camera["height"], camera["width"], 3):
                raise ValueError("RGB dimensions and intrinsics differ")
            if (
                not np.isfinite(camera["timestamp_s"])
                or abs(camera["timestamp_s"] - packet["timestamp_s"]) > 0.015
            ):
                raise ValueError(
                    "Camera/observation timestamps are invalid or differ by more than15ms"
                )
        cameras = calibrate(images, packet["cameras"], calibration_method)
        boxes = [[] for _ in images]
        if proposal in {"learned", "both"}:
            detections = self.network.predict(
                [np.ascontiguousarray(im[..., ::-1]) for im in images],
                imgsz=1280,
                conf=0.1,
                device=0,
                verbose=False,
            )
            for eye, result in enumerate(detections):
                if len(result.boxes):
                    boxes[eye].append(
                        result.boxes.xyxy[int(result.boxes.conf.argmax())].cpu().numpy()
                    )
        if proposal in {"gray", "both"}:
            for eye, (image, camera) in enumerate(zip(images, cameras, strict=False)):
                try:
                    boxes[eye].append(np.asarray(segment_robot(image, camera)[1]))
                except ValueError:
                    if not boxes[eye]:
                        raise
        if any(not b for b in boxes):
            raise ValueError("No robot proposal in one camera")
        samples = [[] for _ in images]
        for resolution in resolutions:
            patches = []
            origins = []
            eyes = []
            for eye, image in enumerate(images):
                for box in boxes[eye]:
                    for margin in margins:
                        patch, origin = crop_box(image, box, margin)
                        patches.append(np.ascontiguousarray(patch[..., ::-1]))
                        origins.append(origin)
                        eyes.append(eye)
            results = self.network.predict(
                patches, imgsz=resolution, conf=0.05, device=0, verbose=False
            )
            for result, origin, eye in zip(results, origins, eyes, strict=False):
                if result.keypoints is None or not len(result.keypoints.xy):
                    continue
                index = int(result.boxes.conf.argmax())
                samples[eye].append(
                    (
                        result.keypoints.xy[index].cpu().numpy() + origin,
                        result.keypoints.conf[index].cpu().numpy(),
                    )
                )
        if joint_mode == "known":
            points = landmarks(self.model, packet["joints_rad"])
            indices = np.arange(len(points))
            joints = packet["joints_rad"]
        elif joint_mode == "unknown":
            indices = INVARIANT_LANDMARKS
            points = landmarks(self.model, np.zeros(19))[indices]
            joints = None
        else:
            raise ValueError("joint_mode must be known or unknown")
        observations = []
        diagnostics = []
        for eye, (sample, camera) in enumerate(zip(samples, cameras, strict=False)):
            if not sample:
                continue
            pixels, confidence, spread = combine_landmarks(sample)
            pixels, confidence, spread = pixels[indices], confidence[indices], spread[indices]
            good = (confidence >= 0.3) & (spread < 2.0) & np.isfinite(pixels).all(1)
            if good.any():
                observations.append(
                    dict(points_trunk=points[good], pixels=pixels[good], camera=camera)
                )
            diagnostics.append(
                dict(
                    camera_id=packet["cameras"][eye]["camera_id"],
                    variants=len(sample),
                    pixels=pixels[good].tolist(),
                    landmark_indices=indices[good].tolist(),
                    spread_median_px=float(np.median(spread[good])) if good.any() else None,
                )
            )
        pose, info = (estimate_keypoints if backend == "opencv" else estimate_generalized)(
            observations
        )
        if info["reprojection_median_px"] > 3:
            raise ValueError("Landmarks disagree with the public geometry")
        info.update(
            runtime_s=time.perf_counter() - started,
            checkpoint_sha256=self.sha256,
            ensemble_cameras=diagnostics,
            joint_mode=joint_mode,
            proposal=proposal,
            resolutions=list(resolutions),
            margins=list(margins),
            estimated_joints=False,
        )
        return prediction(
            pose,
            method=f"ensemble_keypoints_{joint_mode}_{proposal}_{backend}",
            timestamp_s=packet["timestamp_s"],
            joints=joints,
            diagnostics=info,
            inputs=[
                "RGB pair",
                "intrinsics",
                "image wall calibration",
                "trained Microban landmarks",
                "public kinematics and 19 measured joints"
                if joints is not None
                else "ten joint-invariant trunk landmarks; no joint or IMU readings",
            ],
        )
