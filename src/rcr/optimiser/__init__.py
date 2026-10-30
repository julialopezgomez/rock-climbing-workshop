"""Optimiser (laptop): contact plan -> full trajectory. See README.md.

Method (D3): quasi-static direct transcription, one NLP per transition (transition.py), solved
with CasADi + IPOPT, on kinematics written in CasADi from Microban's URDF (robot_model.py).
casadi is imported only when a solve runs, so importing rcr.optimiser needs no extra group.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rcr.types import MICROBAN_JOINTS, ContactPlan, Trajectory, WallModel

if TYPE_CHECKING:
    from rcr.optimiser.transition import TransitionConfig, TransitionResult

# Servo stiffness, to pre-load each target against its expected sag (q_cmd = q + tau / kp).
KP_DEFAULT = 400  # [register units] the servos' P gain [Microban constants.py]
KP_GAIN_PRM = 0.0022  # [N·m/rad per register unit] for the XL330 [Microban constants.py]
SERVO_KP = KP_DEFAULT * KP_GAIN_PRM  # [N·m/rad], about 0.88


def optimise(wall: WallModel, plan: ContactPlan) -> Trajectory:
    """The full joint trajectory the robot executes for this plan."""
    from rcr.optimiser.transition import TransitionConfig

    cfg = TransitionConfig()
    return to_trajectory(solve_plan(wall, plan, cfg), cfg)


def solve_plan(
    wall: WallModel, plan: ContactPlan, cfg: TransitionConfig | None = None
) -> list[TransitionResult]:
    """Solve every transition of the plan in order, each starting where the previous ended."""
    from rcr.optimiser import transition as tr

    cfg = cfg or tr.TransitionConfig()
    return list(iter_plan(wall, plan, start_pose(wall, plan, cfg), cfg))


def start_pose(wall: WallModel, plan: ContactPlan, cfg: TransitionConfig | None = None, near=None):
    """The plan's first configuration: a static pose of its first stance (validates the plan).
    near: a configuration (NQ) to start from (a reference's first), and stay close to with
    w_track."""
    from dataclasses import replace

    from rcr.optimiser import transition as tr

    cfg = cfg or tr.TransitionConfig()
    validate_plan(wall, plan)
    if near is not None:
        stay = max(cfg.w_stay, cfg.w_track)
        return tr.static_pose(wall, plan.stances[0], near, replace(cfg, w_stay=stay)).q[0]
    return tr.stance_pose(wall, plan.stances[0], cfg)


def iter_plan(
    wall: WallModel, plan: ContactPlan, q, cfg: TransitionConfig | None = None, references=None
):
    """Yield each transition of the plan as it is solved, the first starting at q, each next one
    where the previous ended. references: per transition, the (N + 1) x NQ configurations to
    follow (with cfg.w_track; rcr.optimiser.reference)."""
    from rcr.optimiser import transition as tr

    cfg = cfg or tr.TransitionConfig()
    for i, (a, b) in enumerate(zip(plan.stances, plan.stances[1:], strict=False)):
        reference = None if references is None else references[i]
        result = tr.solve_transition(wall, a, b, q, cfg, index=i, reference=reference)
        q = result.q[-1]
        yield result


def validate_plan(wall: WallModel, plan: ContactPlan) -> None:
    """ValueError, naming the stance, unless the plan is for this wall, has at least two
    stances, all on known holds, and exactly one limb changes between consecutive stances.
    Also unless every hold id is unique: holds are found by id, so a repeated id would hide a
    hold."""
    from collections import Counter

    from rcr.optimiser import transition as tr

    repeated = sorted(i for i, n in Counter(h.id for h in wall.holds).items() if n > 1)
    if repeated:
        raise ValueError(f"wall {wall.id!r} has repeated hold ids: {repeated}")
    if plan.wall_id != wall.id:
        raise ValueError(f"the plan is for wall {plan.wall_id!r}, not {wall.id!r}")
    if len(plan.stances) < 2:
        raise ValueError("the plan needs at least two stances to have a transition")
    for i, stance in enumerate(plan.stances):
        tr.check_holds(wall, stance, i)
    for i, (a, b) in enumerate(zip(plan.stances, plan.stances[1:], strict=False)):
        tr.moving_limb(a, b, i)


def to_trajectory(results: list[TransitionResult], cfg: TransitionConfig) -> Trajectory:
    """Chain the transitions' samples (each starts on the previous one's last), time them, and
    pre-load each motor target against its sag: q_cmd = q + tau / SERVO_KP. The planned angles
    and the trunk poses go along, for replaying the plan exactly in the sim."""
    import numpy as np

    times, qs, rows, start = [], [], [], 0.0
    for i, result in enumerate(results):
        for k in range(0 if i == 0 else 1, len(result.q)):
            times.append(start + sample_time(k, cfg))
            qs.append(result.q[k])
            rows.append(result.q[k][7:] + result.tau[k] / SERVO_KP)
        start += sample_time(len(result.q) - 1, cfg)

    def table(rows):
        return [[float(v) for v in row] for row in np.array(rows)]

    return Trajectory(
        joint_names=list(MICROBAN_JOINTS),
        times=[float(t) for t in times],
        positions=table(rows),
        planned_positions=table([q[7:] for q in qs]),
        trunk_poses=table([q[:7] for q in qs]),
    )


def sample_time(k: int, cfg: TransitionConfig) -> float:
    """[s] time of sample k from the start of its transition: phase A's K steps share
    phase_a_duration, phase B's N - K steps share phase_b_duration."""
    n, k_release = cfg.n_samples, cfg.release_sample
    if k <= k_release:
        return cfg.phase_a_duration * k / k_release
    return cfg.phase_a_duration + cfg.phase_b_duration * (k - k_release) / (n - k_release)
