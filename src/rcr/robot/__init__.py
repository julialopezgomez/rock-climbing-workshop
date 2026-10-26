"""Robot: the real-robot backend, Microban. See README.md.

The only code here is our climbing hooks (hooks.py): what we add to Microban's robot, shared by
the optimiser and the sim. Pi-safe.
"""

from rcr.robot.hooks import (
    BALL_INSET,
    BALL_RADIUS,
    FINGER_RADIUS,
    HOOKS,
    MASS,
    SEGMENT_MASS_SHARE,
    Hook,
    ball_centre,
    finger_points,
    masses,
)

__all__ = [
    "BALL_INSET",
    "BALL_RADIUS",
    "FINGER_RADIUS",
    "HOOKS",
    "MASS",
    "SEGMENT_MASS_SHARE",
    "Hook",
    "ball_centre",
    "finger_points",
    "masses",
]
