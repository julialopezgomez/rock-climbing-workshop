"""The climbing hooks: everything we add to Microban's robot [Ours].

Microban as published has no climbing limbs; the organisers will publish theirs [Rules §2.5].
Until then each hand and foot carries the hook below, and every part of our code (the optimiser's
model, collision check and contact point, and the sim's robot) takes it from here, so this file is
the one place to change when the real limbs are known.

A hook, in the frame of the Microban link it is fixed to:

    start ──segment 0──┐
                       │ segment 1 (each a unit direction and a length)
                       ●  ball: past the last segment's end by BALL_INSET ball radii

The ball centre is the limb's contact point: it rests in a hold's spherical socket.

To change when the real limbs are known (measure them, then edit the numbers below):
    the shape     HOOKS: which link, where on it (start), the finger segments
    the size      BALL_RADIUS, FINGER_RADIUS, BALL_INSET
    the mass      MASS, SEGMENT_MASS_SHARE
Then run the tests: they check the optimiser and the sim agree, and replan (the motions depend on
the hook). Elsewhere, also guesses that depend on the real limbs and holds:
    rcr.optimiser.transition.TransitionConfig: normals and mu (how a hold pushes on a ball)
    rcr.optimiser.collision.SOCKET_RADIUS, rcr.sim CupHold: the hold's socket

Pi-safe: numpy only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --- Sizes and mass (the same for all four hooks) ------------------------------------------------
BALL_RADIUS = 0.010  # [m] the ball at the end, what sits in a hold's socket
FINGER_RADIUS = 0.0045  # [m] the finger segments, round bars (capsules)
BALL_INSET = 0.6  # the ball centre lies this many ball radii past the finger's end
MASS = 0.008  # [kg] per hook
SEGMENT_MASS_SHARE = 0.2  # of MASS per finger segment, at its middle; the rest is in the ball


@dataclass(frozen=True)
class Hook:
    link: str  # the Microban link it is fixed to (a URDF link, an MJCF body)
    start: tuple[float, float, float]  # [m] where the finger starts, in that link's frame
    segments: tuple[tuple[tuple[float, float, float], float], ...]  # (unit direction, length [m])


# --- The four hooks -------------------------------------------------------------------------------
# Directions were found by forward kinematics with the robot facing the wall [Ours]:
#   hands: past the end of the forearm, then bent towards the wall with the arm raised overhead
#          (the climbing pose; chosen with the arms hanging, the hook flips away from the wall);
#   feet:  past the toe, then bent down out of the sole, so the ball drops into a socket from
#          above and the foot plate stays clear of the hold's rim (foot frame: z to the toe,
#          -y down).
# Lengths read against the forearm (~0.020 m wide) in the organisers' video [Ours].
HOOKS: dict[str, Hook] = {
    "right_hand": Hook(
        link="radius",
        start=(0.0, -0.067, 0.0135),
        segments=(((0.0, -1.0, 0.0), 0.008), ((1.0, 0.0, 0.0), 0.018)),
    ),
    "left_hand": Hook(
        link="radius_2",
        start=(0.0, 0.067, -0.0135),
        segments=(((0.0, 1.0, 0.0), 0.008), ((-1.0, 0.0, 0.0), 0.018)),
    ),
    "right_foot": Hook(
        link="foot",
        start=(0.0, -0.0217, 0.082),
        segments=(((0.0, 0.0, 1.0), 0.006), ((0.0, -1.0, 0.0), 0.022)),
    ),
    "left_foot": Hook(
        link="foot_2",
        start=(0.0, -0.0217, 0.082),
        segments=(((0.0, 0.0, 1.0), 0.006), ((0.0, -1.0, 0.0), 0.022)),
    ),
}


def finger_points(hook: Hook) -> list[np.ndarray]:
    """[m] the finger's start and the end of each segment, in the hook's link frame."""
    points = [np.array(hook.start, dtype=float)]
    for direction, length in hook.segments:
        points.append(points[-1] + np.array(direction, dtype=float) * length)
    return points


def ball_centre(hook: Hook) -> np.ndarray:
    """[m] the ball centre, the limb's contact point, in the hook's link frame."""
    last_direction = np.array(hook.segments[-1][0], dtype=float)
    return finger_points(hook)[-1] + last_direction * BALL_RADIUS * BALL_INSET


def masses(hook: Hook) -> list[tuple[float, np.ndarray]]:
    """(mass [kg], position [m] in the link frame): each finger segment's middle, then the ball."""
    points = finger_points(hook)
    out = [
        (MASS * SEGMENT_MASS_SHARE, (a + b) / 2) for a, b in zip(points, points[1:], strict=False)
    ]
    out.append((MASS * (1 - SEGMENT_MASS_SHARE * len(hook.segments)), ball_centre(hook)))
    return out
