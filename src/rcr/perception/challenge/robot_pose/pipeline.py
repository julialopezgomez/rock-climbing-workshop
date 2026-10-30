"""Configured keypoint/M3T tracking with persistent geometry and image recovery.

The selected static refinement wraps this tracker in robust_edges.py.
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np

from rcr.perception.challenge.src.camera_io import CameraFrame, validate_pair

from .._paths import WEIGHTS
from .configured_fit import calibrate, estimate
from .contract import prediction
from .frame_conventions import apply
from .keypoint_frontend import RobotKeypoints
from .m3t_tracker import DEFAULT_CACHE, M3TTracker
from .model_adapter import JOINTS, RobotRenderer

DEFAULT_WEIGHTS = WEIGHTS / "body/best.pt"


def normal_packet(observation, frames, clock_domain):
    """Convert normal simulator telemetry and captured RGB calibration to input.

    Only observation q/time are consumed. True simulator transforms and contact
    flags are deliberately outside this adapter.
    """
    q = np.asarray(observation["q"], float)
    if q.shape != (19,) or not np.isfinite(q).all():
        raise ValueError("Normal q must contain the 19 released joints")
    return dict(
        schema="robot-input-v1",
        timestamp_s=float(observation["time"]),
        clock=clock_domain,
        joints_rad=dict(zip(JOINTS, q.tolist(), strict=False)),
        cameras=[
            dict(
                camera_id=f.camera_id,
                K=np.asarray(f.K).tolist(),
                width=f.rgb.shape[1],
                height=f.rgb.shape[0],
                distortion=np.asarray(f.distortion).tolist(),
                distortion_model=f.distortion_model,
                timestamp_s=f.timestamp_s,
                clock_domain=f.clock_domain,
            )
            for f in frames
        ],
    )


def prepare_packet(packet, images, need_joints=True):
    if packet.get("schema") != "robot-input-v1":
        raise ValueError("Expected robot-input-v1")
    if not np.isfinite(packet["timestamp_s"]) or not packet.get("clock"):
        raise ValueError("A finite observation timestamp and common clock domain are required")
    if len(images) != 2 or len(packet["cameras"]) != 2:
        raise ValueError("Two RGB camera observations are required")
    if need_joints:
        if not isinstance(packet.get("joints_rad"), dict) or set(packet["joints_rad"]) != set(
            JOINTS
        ):
            raise ValueError("Configured robot pose requires all 19 named measured joint angles")
        q = np.asarray([packet["joints_rad"][name] for name in JOINTS], float)
        if not np.isfinite(q).all():
            raise ValueError("Joint readings must be finite radians")
        joint_stamp = packet.get("joint_timestamp_s", packet["timestamp_s"])
        if not np.isfinite(joint_stamp) or abs(joint_stamp - packet["timestamp_s"]) > 0.015:
            raise ValueError("Joint readings are not synchronized to the observation")
    frames = []
    for image, camera in zip(images, packet["cameras"], strict=False):
        if camera.get("pixel_convention", "opencv_centres") != "opencv_centres":
            raise ValueError("Convert intrinsics to OpenCV pixel centres once before inference")
        if image.shape != (camera["height"], camera["width"], 3):
            raise ValueError("RGB and calibrated image dimensions differ")
        stamp = camera["timestamp_s"]
        if not np.isfinite(stamp) or abs(stamp - packet["timestamp_s"]) > 0.015:
            raise ValueError("Camera/encoder timestamp skew exceeds15ms")
        frames.append(
            CameraFrame(
                camera_id=camera["camera_id"],
                rgb=image,
                K=np.asarray(camera["K"], float),
                distortion=np.asarray(camera.get("distortion", []), float),
                timestamp_s=stamp,
                clock_domain=camera.get("clock_domain", packet["clock"]),
                distortion_model=camera.get("distortion_model", "opencv_brown_conrady"),
            ).validate()
        )
    timing = validate_pair(*frames, max_skew_s=0.015)
    if timing["clock_domain"] != packet["clock"]:
        raise ValueError("Camera and encoder clocks differ")
    corrected = []
    cameras = []
    for frame in frames:
        image = frame.rgb
        if np.any(frame.distortion):
            image = cv2.undistort(image, frame.K, frame.distortion, None, frame.K)
        corrected.append(image)
        cameras.append(
            dict(
                camera_id=frame.camera_id,
                K=frame.K.tolist(),
                width=image.shape[1],
                height=image.shape[0],
                distortion=[0.0] * 5,
                distortion_model="none",
                timestamp_s=frame.timestamp_s,
            )
        )
    # Explicit allowlist: no true camera/root pose, depth, IMU or contact field
    # from an extended packet is passed to either estimator.
    cleaned = dict(
        schema="robot-input-v1",
        timestamp_s=float(packet["timestamp_s"]),
        clock=packet["clock"],
        cameras=cameras,
    )
    if need_joints:
        cleaned["joints_rad"] = {n: float(packet["joints_rad"][n]) for n in JOINTS}
    return cleaned, corrected, timing


class RobotPoseEstimator:
    def __init__(
        self,
        *,
        profile="fast",
        calibration="aruco",
        weights=DEFAULT_WEIGHTS,
        sequence=False,
        refresh_frames=30,
        cache=DEFAULT_CACHE,
        gpu_confirmed=False,
        correspondence_iterations=None,
        update_iterations=2,
        fixed_cameras=False,
    ):
        if profile != "fast":
            raise ValueError("Use AdaptiveRobustEstimator for the selected max refinement")
        if calibration not in {"aruco", "boundary"}:
            raise ValueError("Unknown image calibration method")
        if refresh_frames < 1:
            raise ValueError("refresh_frames must be positive")
        started = time.perf_counter()
        self.profile = profile
        self.calibration = calibration
        self.sequence = sequence
        self.refresh_frames = refresh_frames
        self.cache = Path(cache)
        self.gpu_confirmed = gpu_confirmed
        self.correspondence_iterations = (
            5 if correspondence_iterations is None else correspondence_iterations
        )
        self.update_iterations = update_iterations
        self.fixed_cameras = fixed_cameras
        self.cached_cameras = None
        self.calibration_signature = None
        self.frontend = RobotKeypoints(weights, gpu_confirmed)
        self.tracker = self._tracker()
        self.refiner = None
        self.renderer = None
        if sequence:
            from .differentiable_fit import DifferentiableCAD

            self.refiner = DifferentiableCAD(model=self.frontend.model, gpu_confirmed=gpu_confirmed)
        self.index = 0
        self.last_timestamp = None
        self.closed = False
        self.setup_seconds = time.perf_counter() - started

    def _tracker(self):
        return M3TTracker(
            model=self.frontend.model,
            cache=self.cache,
            gpu_confirmed=self.gpu_confirmed,
            correspondence_iterations=self.correspondence_iterations,
            update_iterations=self.update_iterations,
        )

    def _fallback(self, images, cameras, joints, initial=None):
        if self.renderer is None:
            self.renderer = RobotRenderer(side=224)
        return estimate(images, cameras, joints, self.renderer, initial=initial)

    def infer(self, packet, images, *, reset=False):
        if self.closed:
            raise RuntimeError("RobotPoseEstimator is closed")
        started = time.perf_counter()
        packet, images, timing = prepare_packet(packet, images)
        timestamp = packet["timestamp_s"]
        if (
            self.sequence
            and self.last_timestamp is not None
            and timestamp <= self.last_timestamp
            and not reset
        ):
            raise ValueError(
                "Sequence timestamps must increase; explicitly reset for a new sequence"
            )
        signature = json.dumps(
            [{k: v for k, v in c.items() if k != "timestamp_s"} for c in packet["cameras"]],
            sort_keys=True,
        )
        metadata_changed = (
            self.calibration_signature is not None and signature != self.calibration_signature
        )
        if metadata_changed:
            self.tracker.close()
            self.tracker = self._tracker()
        use_cache = (
            self.sequence
            and self.fixed_cameras
            and self.cached_cameras is not None
            and not reset
            and not metadata_changed
        )
        cameras = (
            self.cached_cameras
            if use_cache
            else calibrate(images, packet["cameras"], self.calibration)
        )
        if self.fixed_cameras:
            self.cached_cameras = cameras
        self.calibration_signature = signature
        gap = self.last_timestamp is not None and timestamp - self.last_timestamp > 0.4
        reacquire = (
            not self.sequence
            or self.index == 0
            or reset
            or gap
            or metadata_changed
            or self.index % self.refresh_frames == 0
        )
        stages = {}
        initial = None
        inputs = [
            "RGB pair",
            "intrinsics",
            "image-based " + self.calibration + " wall calibration",
            "19 synchronized measured joint angles",
            "public climbing CAD",
        ]
        if reacquire:
            try:
                value = self.frontend.infer(
                    packet, images, self.calibration, "opencv", "known", "learned"
                )
                initial = np.asarray(value["T_wall_trunk"])
                stages["keypoints"] = value["diagnostics"]
                inputs.append("trained Microban keypoint initialization")
            except (ValueError, RuntimeError, cv2.error) as error:
                initial, info = self._fallback(images, cameras, packet["joints_rad"])
                stages["fallback_initialization"] = dict(info, keypoint_failure=str(error))
                inputs.append("autonomous RGB multi-start CAD initialization")
            # Accurate initialization establishes cleaner region histograms for
            # sustained tracking; static fast inference keeps its cheaper path.
            if self.profile == "fast" and self.sequence:
                initial, stages["initial_CAD"] = self.refiner.fit(
                    images, cameras, packet["joints_rad"], initial
                )
        try:
            pose, stages["m3t"] = self.tracker.update(
                images, cameras, packet["joints_rad"], initial=initial, reset=reacquire
            )
        except (ValueError, RuntimeError, TimeoutError, BrokenPipeError) as error:
            self.tracker.close()
            self.tracker = self._tracker()
            initial, info = self._fallback(images, cameras, packet["joints_rad"], initial=initial)
            stages["tracker_recovery"] = dict(info, error=str(error))
            pose, stages["m3t"] = self.tracker.update(
                images, cameras, packet["joints_rad"], initial=initial, reset=True
            )
        endpoints = {
            name: apply(pose, np.asarray([p]))[0].tolist()
            for name, p in self.frontend.model.endpoints(packet["joints_rad"]).items()
        }
        diagnostics = dict(
            stages=stages,
            runtime_s=time.perf_counter() - started,
            setup_seconds=self.setup_seconds,
            setup_included_in_runtime=False,
            profile=self.profile,
            sequence=self.sequence,
            frame_index=self.index,
            reacquired=bool(reacquire),
            timing=timing,
            fixed_camera_rig=self.fixed_cameras,
            used_in_sequence_calibration_cache=bool(use_cache),
            camera_metadata_changed=bool(metadata_changed),
            calibration=[c["calibration"] for c in cameras],
            endpoints_wall_m=endpoints,
            native_depth_used=False,
            imu_used=False,
            contact_truth_used=False,
            scope="Local live Microban estimator; no official offline robot schema or score",
        )
        value = prediction(
            pose,
            method="microban_" + self.profile + "_" + self.calibration,
            timestamp_s=timestamp,
            inputs=inputs,
            joints=packet["joints_rad"],
            diagnostics=diagnostics,
        )
        self.last_timestamp = timestamp
        self.index += 1
        return value

    def close(self):
        if self.closed:
            return
        self.tracker.close()
        if self.renderer is not None:
            self.renderer.close()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
