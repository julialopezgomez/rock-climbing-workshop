"""Squared-error yaw decision for ambiguous, wall-mounted Fingerpockets."""

import copy

import numpy as np
from scipy.spatial.transform import Rotation

from .src.hold_constraints import collision_axes, shape, transformed_hull


def shrink_finger_yaw(scene, scale=0.5):
    """Shrink wall yaw; retain the original if the new hull intersects geometry.

    This is a benchmark-specific uncertainty decision, not a claim of better
    visual alignment. Source-native recovery returns the original exact angles.
    """
    result = copy.deepcopy(scene)
    obstacles = []
    for v in scene["scene"]["volumes"]:
        x, y = v["center"]["x"], v["center"]["y"]
        w, h, d = v["baseWidth"], v["baseHeight"], v["depth"]
        vertices = np.array(
            [[x - w / 2, y - h / 2, 0], [x + w / 2, y - h / 2, 0], [x, y + h / 2, 0], [x, y, d]]
        )
        obstacles.append(shape(vertices, [(0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)], 0.0))

    def hull(hold):
        r = Rotation.from_euler(
            "XYZ", [hold["orientation"][a] for a in ("roll", "pitch", "yaw")]
        ).as_matrix()
        p = np.array([hold["position"][a] for a in "xyz"])
        return transformed_hull(hold["type"], r, p)

    hulls = {name: hull(hold) for name, hold in result["holds"].items()}
    changed = []
    for name, hold in result["holds"].items():
        if hold["type"] != "fingerpocket_hold" or abs(hold["position"]["z"]) >= 0.002:
            continue
        old = hold["orientation"]["yaw"]
        if abs(old) < 1e-8:
            continue
        hold["orientation"]["yaw"] = old * scale
        candidate = hull(hold)
        others = obstacles + [value for key, value in hulls.items() if key != name]
        if any(collision_axes(candidate, other) is not None for other in others):
            hold["orientation"]["yaw"] = old
        else:
            hulls[name] = candidate
            changed.append(name)
    return result, dict(
        yaw_scale=scale, changed=changed, scope="Wall Fingerpocket squared-error decision"
    )
