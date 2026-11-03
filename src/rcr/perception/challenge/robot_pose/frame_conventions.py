"""T_A_B maps B coordinates into A. SI units, OpenCV cameras, WXYZ JSON."""

import numpy as np
from scipy.spatial.transform import Rotation

R_WALL_WORLD = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])


def transform(rotation=None, translation=None):
    t = np.eye(4)
    if rotation is not None:
        t[:3, :3] = rotation
    if translation is not None:
        t[:3, 3] = translation
    return validate_transform(t)


def validate_transform(t):
    t = np.asarray(t, float)
    if t.shape != (4, 4) or not np.isfinite(t).all():
        raise ValueError("Transform must be finite 4x4")
    r = t[:3, :3]
    if (
        not np.allclose(t[3], [0, 0, 0, 1])
        or not np.allclose(r.T @ r, np.eye(3), atol=1e-6)
        or np.linalg.det(r) < 0.999999
    ):
        raise ValueError("Transform must contain a proper rotation")
    return t


def from_wxyz(q):
    q = np.asarray(q, float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q) - 1) > 0.01:
        raise ValueError("Expected normalized finite WXYZ quaternion")
    return Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()


def to_wxyz(r):
    q = Rotation.from_matrix(r).as_quat()[[3, 0, 1, 2]]
    return q if q[0] >= 0 else -q


def apply(t, points):
    t = validate_transform(t)
    return np.asarray(points) @ t[:3, :3].T + t[:3, 3]


def project(points_wall, camera):
    import cv2

    t = np.asarray(camera["T_camera_wall"])
    return cv2.projectPoints(
        np.asarray(points_wall, float),
        Rotation.from_matrix(t[:3, :3]).as_rotvec(),
        t[:3, 3],
        np.asarray(camera["K"], float),
        np.asarray(camera.get("distortion", [0] * 5), float),
    )[0].reshape(-1, 2)


def crop_intrinsics(k, origin, scale=1.0):
    k = np.asarray(k, float).copy()
    k[0, 2] -= origin[0]
    k[1, 2] -= origin[1]
    k[:2] *= scale
    return k


def pose_errors(estimate, truth):
    e, g = validate_transform(estimate), validate_transform(truth)
    return dict(
        translation_mm=float(np.linalg.norm(e[:3, 3] - g[:3, 3]) * 1000),
        rotation_deg=float(np.degrees(Rotation.from_matrix(e[:3, :3] @ g[:3, :3].T).magnitude())),
    )
