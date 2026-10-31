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
KP_FW = 125.0  # kp_fw we run the servos at [simulator README: the gain for learned policies]
# The policy's share of the targets [Ours]: 0 plays the reference back, 1 lets the policy choose
# every target. A curriculum raises it as the policy learns; each policy saves the one it used.
ALPHA = 0.1
OBS_SIZE = 3 * len(JOINTS) + 6  # error to the reference, q, dq, gravity, gyro


@dataclass
class Reference:
    """The trajectory as arrays in JOINTS order."""

    times: np.ndarray  # (N,) [s]
    feedforward: np.ndarray  # (N, 19) [rad] targets pre-loaded for KP_FW
    planned: np.ndarray  # (N, 19) [rad] where the joints should be


def reference(trajectory: Trajectory) -> Reference:
    """Reorder the joints and rescale the pre-load from KP_PRELOAD to KP_FW: the pre-load is
    tau / kp, so at a softer gain it grows by KP_PRELOAD / KP_FW. Without planned_positions (the
    robot's part sent by rcr.link), `positions` are played back as they are."""
    if len(trajectory.times) < 2:
        raise ValueError("the trajectory needs at least two samples")
    cols = [trajectory.joint_names.index(n) for n in JOINTS]
    positions = np.asarray(trajectory.positions, dtype=float)[:, cols]
    if trajectory.planned_positions is None:
        return Reference(np.asarray(trajectory.times, dtype=float), positions, positions)
    planned = np.asarray(trajectory.planned_positions, dtype=float)[:, cols]
    feedforward = planned + (KP_PRELOAD / KP_FW) * (positions - planned)
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


def observation(planned, q, dq, quat, gyro) -> np.ndarray:
    """What the policy sees: the error to the reference, then what the robot measures."""
    err = np.asarray(planned) - np.asarray(q)
    return np.concatenate([err, q, dq, gravity(quat), gyro]).astype(np.float32)


def targets(feedforward, action, alpha, low, high) -> np.ndarray:
    """Joint targets: (1 - alpha) of the feedforward plus alpha of the policy's own targets, the
    action (clipped to [-1, 1]) mapped onto the joint ranges [low, high]."""
    own = low + (np.clip(action, -1.0, 1.0) + 1.0) / 2.0 * (high - low)
    return np.clip((1.0 - alpha) * np.asarray(feedforward) + alpha * own, low, high)
