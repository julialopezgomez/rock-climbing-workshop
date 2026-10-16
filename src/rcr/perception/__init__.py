"""Perception (laptop): camera images -> wall model + robot pose. See README.md.

Until a v0 exists, the pipeline loads a hand-written wall model with rcr.io.load_wall.
"""

import numpy as np

from rcr.types import Pose, WallModel


def wall_model(images: list[np.ndarray]) -> WallModel:
    """RGB images from the two wall cameras -> holds and start stance, in the wall frame."""
    raise NotImplementedError


def robot_pose(images: list[np.ndarray]) -> Pose:
    """RGB images from the two wall cameras -> robot base pose in the wall frame."""
    raise NotImplementedError
