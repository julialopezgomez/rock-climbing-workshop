"""Convert real M3T image estimates to the shared contract, never FK checks."""

import numpy as np
from scipy.spatial.transform import Rotation


def normalize_float32_pose(value):
    pose = np.asarray(value, float).copy()
    if (
        pose.shape != (4, 4)
        or not np.isfinite(pose).all()
        or not np.allclose(pose[3], [0, 0, 0, 1])
    ):
        raise ValueError("M3T returned an invalid homogeneous matrix")
    error = float(np.linalg.norm(pose[:3, :3].T @ pose[:3, :3] - np.eye(3)))
    if error > 1e-4 or np.linalg.det(pose[:3, :3]) < 0.9998:
        raise ValueError("M3T rotation error exceeds float32 rounding tolerance")
    pose[:3, :3] = Rotation.from_matrix(pose[:3, :3]).as_matrix()
    return pose, error
