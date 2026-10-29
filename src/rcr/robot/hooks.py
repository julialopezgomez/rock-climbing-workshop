"""The climbing ends: the balls of the climbing model that go into the holds, read from the model.

The robot's climbing variant (microban_model, robot_climb.urdf / .xml; see rcr.robot.microban) has
climbing forearms and feet, each ending in a ball. Everything here is read from the model files on
first use, nothing is typed in [Microban]:

    link   the link the ball is on: the parent of the URDF's fixed joint `<limb>_ball_frame`
    ball   the ball's centre in that link's frame [m]: that joint's origin; the limb's contact
           point, which rests in a hold's spherical socket
    BALL_RADIUS   [m] the radius of the MJCF's ball geoms `<limb>_ball_coll` (all four the same)

One thing is ours, for the optimiser only:

    neck   the direction the forearm or foot comes into the ball, in the link's frame (unit). The
           optimiser's first solve lowers each ball into its socket from above along it
           (rcr.optimiser.transition.NECK_CONE). [Ours, measured on the model: from its collision
           pieces 10-20 mm from the ball centre to the centre, averaged]

The organisers' sim has the same balls (tests/test_optimiser.py checks it).

Pi-safe: stdlib and numpy (the model files are read only when HOOKS or BALL_RADIUS is asked for).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

LIMBS = ("right_hand", "left_hand", "right_foot", "left_foot")  # as rcr.types.LIMBS

# [Ours] see `neck` above
NECK: dict[str, tuple[float, float, float]] = {
    "right_hand": (0.79, 0.61, 0.0),
    "left_hand": (-0.79, -0.62, 0.0),
    "right_foot": (0.0, -0.99, -0.16),
    "left_foot": (0.0, -0.99, -0.16),
}


@dataclass(frozen=True)
class Hook:
    link: str  # the model's link the ball is on (a URDF link, an MJCF body)
    ball: tuple[float, float, float]  # [m] the ball's centre, in that link's frame
    neck: tuple[float, float, float]  # the forearm's or foot's direction into the ball [Ours]


def ball_centre(hook: Hook) -> np.ndarray:
    """[m] the ball centre, the limb's contact point, in the hook's link frame."""
    return np.array(hook.ball, dtype=float)


@lru_cache(maxsize=1)
def _read() -> tuple[dict[str, Hook], float]:
    from rcr.robot import microban

    joints = {j.get("name"): j for j in ET.parse(microban.urdf_path()).getroot().iter("joint")}
    hooks = {}
    for limb in LIMBS:
        joint = joints[f"{limb}_ball_frame"]
        # the frame's origin is the centre (its rotation does not matter for a sphere)
        xyz = tuple(float(v) for v in joint.find("origin").get("xyz").split())
        hooks[limb] = Hook(joint.find("parent").get("link"), xyz, NECK[limb])
    geoms = {g.get("name"): g for g in ET.parse(microban.mjcf_path()).getroot().iter("geom")}
    radii = {float(geoms[f"{limb}_ball_coll"].get("size").split()[0]) for limb in LIMBS}
    if len(radii) != 1:
        raise ValueError(f"the balls differ in radius: {sorted(radii)}")
    return hooks, radii.pop()


def __getattr__(name: str):
    """HOOKS (limb -> Hook) and BALL_RADIUS [m], read from the model on first use."""
    if name == "HOOKS":
        return _read()[0]
    if name == "BALL_RADIUS":
        return _read()[1]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
