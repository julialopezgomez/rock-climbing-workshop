"""Geometry and output helpers shared by the CPU examples.

Transforms map model coordinates to camera-0 coordinates. Lengths are metres;
image coordinates are OpenCV pixel centres (x right, y down).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

plt.rcParams.update({"font.family": "serif"})


def transform_points(points, rotation, translation):
    return np.asarray(points) @ rotation.T + translation


def project(points_camera, intrinsics):
    if np.any(points_camera[:, 2] <= 0):
        raise ValueError("All projected points must be in front of the camera")
    homogeneous = points_camera @ intrinsics.T
    return homogeneous[:, :2] / homogeneous[:, 2:]


def pose_errors(rotation, translation, reference_rotation, reference_translation):
    return {
        "position_error_mm": float(1000 * np.linalg.norm(translation - reference_translation)),
        "rotation_error_deg": float(
            np.rad2deg(Rotation.from_matrix(rotation @ reference_rotation.T).magnitude())
        ),
    }


def write_results(folder, result, figure):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    figure.savefig(folder / "figure.png", dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps(result, indent=2, allow_nan=False))
