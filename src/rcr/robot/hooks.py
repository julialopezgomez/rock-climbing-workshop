"""The climbing ends: the balls of the climbing model that go into the holds [Microban].

The robot's climbing variant (microban_model, robot_climb.urdf / .xml; see rcr.robot.microban) has
climbing forearms and feet, each ending in a ball. Every part of our code (the optimiser's
model, collision check and contact point, and the sim) takes the balls from here:

    link   the model's link (URDF) / body (MJCF) the ball is on
    ball   the ball's centre in that link's frame [m]: the limb's contact point, which rests in
           a hold's spherical socket
    neck   the direction the finger comes into the ball, in that frame (unit): the optimiser
           lowers the ball into a socket along it, from above [Ours, measured on the model: from
           the model's collision pieces 10-20 mm from the ball centre to the centre, averaged]

The forearms and feet themselves, with their masses and collision shapes, are in the model. The
values below are the model's (frames right_hand_ball_frame, ..., geoms *_ball_coll, at
microban_model 8e42ef7); a test checks them against the model, so a new model that moves the
balls fails it instead of going unnoticed.

Elsewhere, guesses that depend on the real limbs and holds:
    rcr.optimiser.transition.TransitionConfig: normals and mu (how a hold pushes on a ball)
    rcr.optimiser.collision.SOCKET_RADIUS, rcr.sim CupHold: the hold's socket

Pi-safe: numpy only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BALL_RADIUS = 0.0125  # [m] every ball (*_ball_coll size) [Microban]


@dataclass(frozen=True)
class Hook:
    link: str  # the model's link the ball is on (a URDF link, an MJCF body)
    ball: tuple[float, float, float]  # [m] the ball's centre, in that link's frame
    neck: tuple[float, float, float]  # the finger's direction into the ball, in that frame


HOOKS: dict[str, Hook] = {
    "right_hand": Hook("radius_right", ball=(0.03, -0.0715, 0.0135), neck=(0.79, 0.61, 0.0)),
    "left_hand": Hook("radius_left", ball=(-0.03, 0.0715, -0.0135), neck=(-0.79, -0.62, 0.0)),
    "right_foot": Hook("foot_climb", ball=(0.0, -0.0065, 0.1095), neck=(0.0, -0.99, -0.16)),
    "left_foot": Hook("foot_climb_2", ball=(0.0, -0.0065, 0.1095), neck=(0.0, -0.99, -0.16)),
}


def ball_centre(hook: Hook) -> np.ndarray:
    """[m] the ball centre, the limb's contact point, in the hook's link frame."""
    return np.array(hook.ball, dtype=float)
