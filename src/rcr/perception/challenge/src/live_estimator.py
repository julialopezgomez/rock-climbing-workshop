"""Metric frame-transform serialization shared by live cup mapping."""

import numpy as np


def frame_transform(child, point, stamp):
    return {
        "parent_frame": "wall",
        "child_frame": child,
        "translation_m": np.asarray(point).tolist(),
        "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
        "timestamp_s": stamp,
    }
