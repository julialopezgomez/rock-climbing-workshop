"""Residual tracking of the optimiser's trajectory [Ours]: the numbers both sides must agree on.

Controller.step (robot) and the training env (laptop, scripts/train_controller.py) both call these
functions, so the policy sees the same observation and its action means the same targets in
training and on the robot. Change them only together with a retrain.

Pi-safe: numpy only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rcr.types import MICROBAN_JOINTS, Trajectory

# The policy's vector order. The sim (climbing_sim.JOINTS) uses another order: convert by name.
JOINTS = MICROBAN_JOINTS
KP_PRELOAD = 400.0  # kp_fw the optimiser pre-loads `positions` for [optimiser KP_DEFAULT]
# Default kp_fw (P only) [simulator README: the gain for learned policies]. A policy saves the one
# it was trained at, and the robot must run its climbing move at that gain.
KP_FW = 125.0
# Residual action [Ours, as residual policy learning, Silver et al. 2018]: targets = feedforward +
# ALPHA * action, action in [-1, 1], so ALPHA is the largest correction (rad) and action 0 plays
# the reference back. Each policy saves the ALPHA it used, and ACTION, which Controller checks.
ALPHA = 0.2
ACTION = "residual"
LOOKAHEAD_S = 0.1  # [s] the policy also sees the error to where the reference will be then [Ours]
OBS_SIZE = 4 * len(JOINTS) + 6  # error to the reference now and ahead, q, dq, gravity, gyro


@dataclass
class Reference:
    """The trajectory as arrays in JOINTS order."""

    times: np.ndarray  # (N,) [s]
    feedforward: np.ndarray  # (N, 19) [rad] targets pre-loaded for the servos' kp_fw
    planned: np.ndarray  # (N, 19) [rad] where the joints should be


def reference(trajectory: Trajectory, kp_fw: float = KP_FW) -> Reference:
    """Reorder the joints and rescale the pre-load from KP_PRELOAD to kp_fw: the pre-load is
    tau / kp, so at a softer gain it grows by KP_PRELOAD / kp_fw. Without planned_positions (the
    robot's part sent by rcr.link), `positions` are played back as they are."""
    if len(trajectory.times) < 2:
        raise ValueError("the trajectory needs at least two samples")
    cols = [trajectory.joint_names.index(n) for n in JOINTS]
    positions = np.asarray(trajectory.positions, dtype=float)[:, cols]
    if trajectory.planned_positions is None:
        return Reference(np.asarray(trajectory.times, dtype=float), positions, positions)
    planned = np.asarray(trajectory.planned_positions, dtype=float)[:, cols]
    feedforward = planned + (KP_PRELOAD / kp_fw) * (positions - planned)
    return Reference(np.asarray(trajectory.times, dtype=float), feedforward, planned)


def at(ref: Reference, t: float) -> tuple[np.ndarray, np.ndarray]:
    """(feedforward, planned) at time t, linear between samples, held before the first and after
    the last."""
    k = int(np.clip(np.searchsorted(ref.times, t, side="right") - 1, 0, len(ref.times) - 2))
    s = float(np.clip((t - ref.times[k]) / (ref.times[k + 1] - ref.times[k]), 0.0, 1.0))
    ff = ref.feedforward[k] + s * (ref.feedforward[k + 1] - ref.feedforward[k])
    planned = ref.planned[k] + s * (ref.planned[k + 1] - ref.planned[k])
    return ff, planned


def gravity(quat) -> np.ndarray:
    """Gravity direction in the trunk frame from its orientation (w, x, y, z): R^T (0, 0, -1)."""
    w, x, y, z = quat
    return -np.array([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)])


def observation(planned, planned_ahead, q, dq, quat, gyro) -> np.ndarray:
    """What the policy sees: the error to the reference now and LOOKAHEAD_S ahead, then what the
    robot measures."""
    err, ahead = np.asarray(planned) - np.asarray(q), np.asarray(planned_ahead) - np.asarray(q)
    return np.concatenate([err, ahead, q, dq, gravity(quat), gyro]).astype(np.float32)


def targets(feedforward, action, alpha, low, high) -> np.ndarray:
    """Joint targets: the feedforward plus alpha (rad) times the action clipped to [-1, 1], within
    the joint ranges [low, high]. Action 0 is playback."""
    return np.clip(np.asarray(feedforward) + alpha * np.clip(action, -1.0, 1.0), low, high)
