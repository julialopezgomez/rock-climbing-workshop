"""Four public hold meshes and calibrated scene projection helpers."""

from __future__ import annotations

import os

from .._paths import CACHE

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def normalized_mesh(kind="microban_hold"):
    names = {
        "microban_hold": "microban_hold.stl",
        "cube_hold": "cube_hold.stl",
        "sidepull_hold": "sidehpull_hold.stl",
        "fingerpocket_hold": "fingerpocket_hold.stl",
    }
    name = names[kind]
    mesh = trimesh.load(CACHE / "assets/rcr" / name, force="mesh")
    v = np.asarray(mesh.vertices) @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0.0]]).T
    lo, hi = v.min(0), v.max(0)
    v[:, :2] -= (lo[:2] + hi[:2]) / 2
    v[:, 2] -= lo[2]
    return trimesh.Trimesh(v, mesh.faces, process=False)


def _calibration(calibration):
    cameras = calibration["cameras"]
    K = [np.array(cameras[c]["K"], float) for c in ["left", "right"]]
    T = [np.array(cameras[c]["T_camera_scene"], float) for c in ["left", "right"]]
    return K, T


def _project(point, K, T):
    p = T[:3, :3] @ point + T[:3, 3]
    return (K @ p)[:2] / p[2], float(p[2])


def _rotation(normal, yaw):
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    z = np.array([0.0, 0, 1])
    cross = np.cross(z, n)
    c = np.clip(z @ n, -1, 1)
    if np.linalg.norm(cross) < 1e-10:
        align = np.eye(3) if c > 0 else Rotation.from_rotvec([np.pi, 0, 0]).as_matrix()
    else:
        align = Rotation.from_rotvec(cross / np.linalg.norm(cross) * np.arccos(c)).as_matrix()
    return align @ Rotation.from_rotvec([0, 0, yaw]).as_matrix()
