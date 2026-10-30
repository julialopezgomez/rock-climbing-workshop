"""Input contract shared by simulator replay and future two-RealSense capture.

Optical camera axes: X right, Y down, Z forward. Geometry/depth in metres;
T_camera_wall maps metric wall coordinates into optical camera coordinates.
Native depth is an optional research input, with provenance kept explicit.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CameraFrame:
    camera_id: str
    rgb: np.ndarray
    K: np.ndarray
    distortion: np.ndarray
    timestamp_s: float
    clock_domain: str
    T_camera_wall: np.ndarray | None = None
    depth_m: np.ndarray | None = None
    depth_source: str | None = None
    depth_aligned_to_rgb: bool = False
    distortion_model: str = "opencv_brown_conrady"

    def validate(self):
        if self.rgb.dtype != np.uint8 or self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
            raise ValueError("RGB must be uint8 HxWx3 in RGB channel order")
        K = np.asarray(self.K)
        if K.shape != (3, 3) or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0:
            raise ValueError("Invalid pinhole intrinsics")
        if not np.allclose(K[2], [0, 0, 1]):
            raise ValueError("Intrinsics must have homogeneous last row [0,0,1]")
        if not self.camera_id or not self.clock_domain or not np.isfinite(self.timestamp_s):
            raise ValueError("Camera identity and finite timestamp with clock domain are required")
        distortion = np.asarray(self.distortion)
        if self.distortion_model not in {"opencv_brown_conrady", "none"}:
            raise ValueError(
                "Convert this camera distortion model to OpenCV Brown-Conrady before use"
            )
        if (
            distortion.ndim != 1
            or len(distortion) not in {0, 4, 5, 8, 12, 14}
            or not np.isfinite(distortion).all()
        ):
            raise ValueError("Lens distortion must be a finite OpenCV coefficient vector")
        if self.distortion_model == "none" and np.any(distortion):
            raise ValueError("Nonzero distortion contradicts the none model")
        if self.T_camera_wall is not None:
            T = np.asarray(self.T_camera_wall)
            if T.shape != (4, 4) or not np.isfinite(T).all() or not np.allclose(T[3], [0, 0, 0, 1]):
                raise ValueError("Invalid metric camera transform")
            if not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-5) or not np.isclose(
                np.linalg.det(T[:3, :3]), 1, atol=1e-5
            ):
                raise ValueError("Camera rotation must be proper orthonormal")
        if self.depth_m is not None:
            if self.depth_m.shape != self.rgb.shape[:2] or not self.depth_aligned_to_rgb:
                raise ValueError("Depth must be explicitly aligned to RGB at the same resolution")
            if not np.issubdtype(self.depth_m.dtype, np.floating):
                raise ValueError(
                    "Depth must be floating point metres; convert device units explicitly"
                )
            if self.depth_source not in {"simulator_oracle", "rgb_stereo", "native_rgbd_research"}:
                raise ValueError("Explicit depth provenance required")
            finite = self.depth_m[np.isfinite(self.depth_m)]
            if (finite < 0).any():
                raise ValueError("Negative depth is invalid; use zero or NaN for missing pixels")
        return self


def validate_pair(left, right, max_skew_s=0.02):
    left.validate()
    right.validate()
    if left.camera_id == right.camera_id:
        raise ValueError("Stereo requires distinct camera identities")
    if left.clock_domain != right.clock_domain:
        raise ValueError("Cannot compare independent device clocks without synchronization")
    skew = abs(left.timestamp_s - right.timestamp_s)
    if skew > max_skew_s:
        raise ValueError(f"Pair skew {skew:.4f}s exceeds {max_skew_s:.4f}s")
    return {"skew_s": skew, "clock_domain": left.clock_domain}


def load_sim_pair(scene: Path, include_oracle_depth=False):
    from PIL import Image

    meta = json.loads((scene / "metadata.json").read_text())
    frames = []
    for name in ["cam_left", "cam_right"]:
        c = meta["cameras"][name]
        frames.append(
            CameraFrame(
                name,
                np.array(Image.open(scene / name / "rgb.png")),
                np.array(c["K"]),
                np.array(c["distortion"]),
                c["timestamp_s"],
                "simulator",
                np.array(c["T_camera_wall"]),
                np.load(scene / name / "depth.npy") if include_oracle_depth else None,
                "simulator_oracle" if include_oracle_depth else None,
                include_oracle_depth,
            ).validate()
        )
    validate_pair(*frames)
    return tuple(frames)
