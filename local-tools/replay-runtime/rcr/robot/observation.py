"""Microban's observation and targets, to and from our types [Microban].

A Microban move's step(obs) and the organisers' simulator's step() / observation() use the same
dict (third_party/microban/docs/moves.md): q, dq (the 19 joints, in microban_model.JOINTS order),
quat (trunk, w x y z), gyro (trunk frame), time, ... And both take the targets as 19 angles in that
order. The order is passed in (`joints`), so this file needs no Microban import.

Pi-safe: stdlib only.
"""

from __future__ import annotations

from collections.abc import Sequence

from rcr.types import JointTargets, Pose, RobotState


def to_state(obs: dict, joints: Sequence[str], t0: float = 0.0, base: Pose | None = None):
    """RobotState from an observation; time counts from t0, base is the trunk if known."""
    return RobotState(
        time=float(obs["time"]) - t0,
        joint_positions={n: float(v) for n, v in zip(joints, obs["q"], strict=True)},
        joint_velocities={n: float(v) for n, v in zip(joints, obs["dq"], strict=True)},
        imu_quat=tuple(float(v) for v in obs["quat"]),
        gyro=tuple(float(v) for v in obs["gyro"]),
        base_pose=base,
    )


def to_targets(targets: JointTargets, joints: Sequence[str], last: Sequence[float]) -> list:
    """The 19 angles in `joints` order; a joint absent from `targets` keeps its `last` target, as
    for a Microban move. Raises KeyError on a joint name Microban does not have."""
    unknown = set(targets.positions) - set(joints)
    if unknown:
        raise KeyError(f"unknown joints {sorted(unknown)}")
    return [float(targets.positions.get(n, v)) for n, v in zip(joints, last, strict=True)]
