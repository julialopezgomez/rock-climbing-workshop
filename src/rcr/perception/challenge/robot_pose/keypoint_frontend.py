"""Flexible Microban landmark frontend, including encoder-free trunk pose.

The learned proposal uses full-image detection before a high-resolution landmark
crop. It does not use the incumbent's gray-robot segmentation for that crop.
"""

import time

import numpy as np

from .configured_fit import calibrate
from .contenders.geometry import landmarks
from .contenders.keypoint_data import image_crop
from .contenders.keypoint_predict import Frontend
from .contract import prediction
from .keypoint_pose import estimate_generalized, estimate_keypoints


def crop_box(image, box, margin=0.3):
    x0, y0, x1, y1 = np.asarray(box, float)
    side = max(x1 - x0 + 1, y1 - y0 + 1) * (1 + 2 * margin)
    center = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    low = np.maximum(0, np.floor(center - side / 2)).astype(int)
    high = np.minimum([image.shape[1], image.shape[0]], np.ceil(center + side / 2) + 1).astype(int)
    if np.any(high - low < 10):
        raise ValueError("Learned robot proposal is too small")
    return image[low[1] : high[1], low[0] : high[0]].copy(), low


class RobotKeypoints(Frontend):
    def infer(
        self,
        packet,
        images,
        calibration_method="aruco",
        backend="opencv",
        joint_mode="known",
        proposal="learned",
    ):
        started = time.perf_counter()
        if len(images) != 2 or len(packet["cameras"]) != 2:
            raise ValueError("Two camera observations are required")
        for image, camera in zip(images, packet["cameras"], strict=False):
            if image.shape != (camera["height"], camera["width"], 3):
                raise ValueError("RGB and calibrated dimensions differ")
            if abs(camera["timestamp_s"] - packet["timestamp_s"]) > 0.015:
                raise ValueError("Camera/observation timestamps differ by more than 15 ms")
        cameras = calibrate(images, packet["cameras"], calibration_method)
        if proposal == "learned":
            detections = self.network.predict(
                [np.ascontiguousarray(im[..., ::-1]) for im in images],
                imgsz=1280,
                conf=0.1,
                device=0,
                verbose=False,
            )
            crops, origins = [], []
            for image, result in zip(images, detections, strict=False):
                if not len(result.boxes):
                    raise ValueError("No learned robot proposal")
                index = int(result.boxes.conf.argmax())
                patch, origin = crop_box(image, result.boxes.xyxy[index].cpu().numpy())
                crops.append(patch)
                origins.append(origin)
        else:
            crops, origins = [], []
            for image, camera in zip(images, cameras, strict=False):
                patch, origin, _ = image_crop(image, camera)
                crops.append(patch)
                origins.append(origin)
        detected = self.network.predict(
            [np.ascontiguousarray(im[..., ::-1]) for im in crops],
            imgsz=384,
            conf=0.05,
            device=0,
            verbose=False,
        )
        if joint_mode == "known":
            points = landmarks(self.model, packet["joints_rad"])
            indices = np.arange(len(points))
            joints = packet["joints_rad"]
        elif joint_mode == "unknown":
            # Only the six fixed trunk points are used. Measured or true joint
            # fields are not read in this branch, even when present in a packet.
            indices = np.arange(4, 10)
            points = landmarks(self.model, np.zeros(19))[indices]
            joints = None
        else:
            raise ValueError("joint_mode must be known or unknown")
        observations = []
        for result, origin, camera in zip(detected, origins, cameras, strict=False):
            if result.keypoints is None or not len(result.keypoints.xy):
                continue
            index = int(result.boxes.conf.argmax())
            pixels = result.keypoints.xy[index].cpu().numpy()[indices] + origin
            confidence = result.keypoints.conf[index].cpu().numpy()[indices]
            good = np.isfinite(pixels).all(1) & (confidence >= 0.3)
            if good.any():
                observations.append(
                    dict(points_trunk=points[good], pixels=pixels[good], camera=camera)
                )
        solver = estimate_keypoints if backend == "opencv" else estimate_generalized
        pose, info = solver(observations)
        if info["reprojection_median_px"] > 3:
            raise ValueError("Learned landmarks disagree with public robot geometry")
        inputs = [
            "RGB pair",
            "intrinsics",
            "image-based " + calibration_method + " calibration",
            "trained Microban keypoints",
            "public rigid trunk geometry" if joints is None else "public robot kinematics",
        ]
        if joints is not None:
            inputs.append("19 measured joint angles")
        info.update(
            runtime_s=time.perf_counter() - started,
            checkpoint_sha256=self.sha256,
            proposal=proposal,
            joint_mode=joint_mode,
            landmark_indices=indices.tolist(),
            estimated_joints=False,
            landmarks_by_camera=[len(o["pixels"]) for o in observations],
        )
        return prediction(
            pose,
            method=f"keypoints_{proposal}_{joint_mode}_{backend}",
            timestamp_s=packet["timestamp_s"],
            inputs=inputs,
            joints=joints,
            diagnostics=info,
        )
