"""Optimiser (laptop): contact plan -> full trajectory; owns feasibility. See README.md.

Method (D3): quasi-static direct transcription, one NLP per transition (transition.py), solved
with CasADi + IPOPT, on kinematics written in CasADi from Microban's URDF (robot_model.py).
casadi is imported only when a solve runs, so importing rcr.optimiser needs no extra group.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rcr.types import MICROBAN_JOINTS, ContactPlan, Feasibility, Stance, Trajectory, WallModel

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
    validate_plan(wall, plan)
    first = plan.stances[0]
    q = tr.solve_static_pose(wall, first, tr.initial_guess(wall, first, cfg), cfg)
    results = []
    for i, (a, b) in enumerate(zip(plan.stances, plan.stances[1:], strict=False)):
        results.append(tr.solve_transition(wall, a, b, q, cfg, index=i))
        q = results[-1].q[-1]
    return results


def validate_plan(wall: WallModel, plan: ContactPlan) -> None:
    """ValueError, naming the stance, unless the plan is for this wall, has at least two
    stances, all on known holds, and exactly one limb changes between consecutive stances."""
    from rcr.optimiser import transition as tr

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
    pre-load each motor target against its sag: q_cmd = q + tau / SERVO_KP."""
    import numpy as np

    times, rows, start = [], [], 0.0
    for i, result in enumerate(results):
        for k in range(0 if i == 0 else 1, len(result.q)):
            times.append(start + sample_time(k, cfg))
            rows.append(result.q[k][7:] + result.tau[k] / SERVO_KP)
        start += cfg.phase_a_duration + cfg.phase_b_duration
    return Trajectory(
        joint_names=list(MICROBAN_JOINTS),
        times=[float(t) for t in times],
        positions=[[float(v) for v in row] for row in np.array(rows)],
    )


def sample_time(k: int, cfg: TransitionConfig) -> float:
    """[s] time of sample k from the start of its transition: phase A's K steps share
    phase_a_duration, phase B's N - K steps share phase_b_duration."""
    n, k_release = cfg.n_samples, cfg.release_sample
    if k <= k_release:
        return cfg.phase_a_duration * k / k_release
    return cfg.phase_a_duration + cfg.phase_b_duration * (k - k_release) / (n - k_release)


def feasibility(wall: WallModel, a: Stance, b: Stance) -> Feasibility:
    """Can the robot go from stance a to stance b? Passed to the planner as FeasibilityFn.

    Not implemented yet; transition.solve_static_pose is a cheap first check (see README).
    """
    raise NotImplementedError
