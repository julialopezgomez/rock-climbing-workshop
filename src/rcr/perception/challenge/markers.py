"""Measured marker-to-trunk geometry and joint two-camera reprojection fitting."""

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .robot_pose.contract import prediction
from .robot_pose.frame_conventions import validate_transform
from .src.calibration import calibrate_wall


class MarkerEstimator:
    def __init__(self, config):
        if not config.get("measured", False):
            raise ValueError(
                "Set measured=true only after measuring marker size and T_trunk_marker"
            )
        self.size = float(config["code_size_m"])
        if not 0 < self.size < 1:
            raise ValueError("Marker code size must be in metres")
        self.identifier = int(config.get("marker_id", 5))
        self.t_trunk_marker = validate_transform(config["T_trunk_marker"])
        self.wall = config.get("wall", {})
        if "marker_corners_wall_m" in self.wall:
            self.wall["marker_corners_wall_m"] = {
                int(k): v for k, v in self.wall["marker_corners_wall_m"].items()
            }
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), parameters
        )
        half = self.size / 2
        self.points = np.array(
            [[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]]
        )

    def infer(self, packet, images):
        from .robot_pose.pipeline import prepare_packet

        packet, images, _ = prepare_packet(packet, images, need_joints=False)
        observations, candidates = [], []
        for image, camera in zip(images, packet["cameras"], strict=True):
            k = np.asarray(camera["K"], float)
            dist = np.asarray(camera.get("distortion", [0] * 5), float)
            image = cv2.undistort(image, k, dist) if np.any(dist) else image
            # Default dimensions match the released board; overrides are measured
            # physical dimensions.
            try:
                wall = calibrate_wall(image, k, **self.wall)
            except ValueError:
                continue
            corners, ids, _ = self.detector.detectMarkers(image)
            if ids is None or self.identifier not in ids.ravel():
                continue
            uv = corners[list(ids.ravel()).index(self.identifier)].reshape(4, 2)
            result = cv2.solvePnPGeneric(self.points, uv, k, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            for rv, tv in zip(result[1], result[2], strict=True):
                t = np.eye(4)
                t[:3, :3] = cv2.Rodrigues(rv)[0]
                t[:3, 3] = tv.ravel()
                candidates.append(
                    np.linalg.inv(wall.T_camera_wall) @ t @ np.linalg.inv(self.t_trunk_marker)
                )
            observations.append((k, wall.T_camera_wall, uv))
        if not candidates:
            raise ValueError("Robot marker and wall markers were not jointly observed")
        points = self.points @ self.t_trunk_marker[:3, :3].T + self.t_trunk_marker[:3, 3]

        def residual(parameters):
            world = points @ Rotation.from_rotvec(parameters[:3]).as_matrix().T + parameters[3:]
            result = []
            for k, tc, uv in observations:
                camera = world @ tc[:3, :3].T + tc[:3, 3]
                if np.any(camera[:, 2] <= 0):
                    return np.full(8 * len(observations), 1e6)
                pixel = camera @ k.T
                result.extend((pixel[:, :2] / pixel[:, 2:] - uv).ravel())
            return np.asarray(result)

        fits = [
            least_squares(
                residual,
                np.r_[Rotation.from_matrix(t[:3, :3]).as_rotvec(), t[:3, 3]],
                loss="soft_l1",
                max_nfev=200,
            )
            for t in candidates
        ]
        fit = min(fits, key=lambda f: np.mean(residual(f.x) ** 2))
        rms = float(np.sqrt(np.mean(residual(fit.x) ** 2)))
        if not fit.success or rms > 3:
            raise ValueError(f"Marker fit rejected: {rms:.2f}px reprojection RMS")
        t = np.eye(4)
        t[:3, :3] = Rotation.from_rotvec(fit.x[:3]).as_matrix()
        t[:3, 3] = fit.x[3:]
        return prediction(
            t,
            method="measured_marker_multiview",
            timestamp_s=packet["timestamp_s"],
            joints=None,
            diagnostics={"reprojection_rms_px": rms, "views": len(observations)},
            inputs=[
                "RGB",
                "measured code size",
                "measured marker-to-trunk transform",
                "wall marker geometry",
            ],
        )

    def close(self):
        pass
