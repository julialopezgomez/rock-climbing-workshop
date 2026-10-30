"""Optimiser (laptop): contact plan -> full trajectory; owns feasibility. See README.md.

Method (D3): quasi-static direct transcription, one NLP per transition (transition.py), solved
with CasADi + IPOPT, on kinematics written in CasADi from Microban's URDF (robot_model.py).
casadi is imported only when a solve runs, so importing rcr.optimiser.v0 needs no extra group.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rcr.types import MICROBAN_JOINTS, ContactPlan, Feasibility, Stance, Trajectory, WallModel

if TYPE_CHECKING:
    from rcr.optimiser.v0.transition import TransitionConfig, TransitionResult

# Servo stiffness, to pre-load each target against its expected sag (q_cmd = q + tau / kp).
KP_DEFAULT = 400  # [register units] the servos' P gain [Microban constants.py]
KP_GAIN_PRM = 0.0022  # [N·m/rad per register unit] for the XL330 [Microban constants.py]
SERVO_KP = KP_DEFAULT * KP_GAIN_PRM  # [N·m/rad], about 0.88


def optimise(wall: WallModel, plan: ContactPlan) -> Trajectory:
    """The full joint trajectory the robot executes for this plan."""
    from rcr.optimiser.v0.transition import TransitionConfig

    cfg = TransitionConfig()
    return to_trajectory(solve_plan(wall, plan, cfg), cfg)


DEFAULT = "default"  # cache argument: move_cache.default_cache (none unless RCR_MOVE_CACHE is set)


def solve_plan(
    wall: WallModel, plan: ContactPlan, cfg: TransitionConfig | None = None, cache=DEFAULT
) -> list[TransitionResult]:
    """Solve every transition of the plan in order, each starting where the previous ended
    (with cfg.settle: each from its stance's own pose; stored moves come from the cache)."""
    from rcr.optimiser.v0 import transition as tr

    cfg = cfg or tr.TransitionConfig()
    return list(iter_plan(wall, plan, start_pose(wall, plan, cfg), cfg, cache))


def start_pose(wall: WallModel, plan: ContactPlan, cfg: TransitionConfig | None = None):
    """The plan's first configuration: a static pose of its first stance (validates the plan)."""
    from rcr.optimiser.v0 import transition as tr

    cfg = cfg or tr.TransitionConfig()
    validate_plan(wall, plan)
    return tr.stance_pose(wall, plan.stances[0], cfg)


def iter_plan(
    wall: WallModel, plan: ContactPlan, q, cfg: TransitionConfig | None = None, cache=DEFAULT
):
    """Yield each transition of the plan as it is solved, the first starting at q, each next one
    where the previous ended. A move starts from the same limb's last solved motion when the holds
    are laid out alike (move_cache.same_layout): on a regular wall, most moves repeat a row up.
    With cfg.settle, each move is tr.solve_move instead: from stance a's own pose to stance b's,
    the same every run (q, from start_pose, is stance 0's pose). Moves are reused from and
    stored in `cache`: by default move_cache.default_cache (none unless RCR_MOVE_CACHE is set);
    None for none."""
    from rcr.optimiser.v0 import transition as tr

    cfg = cfg or tr.TransitionConfig()
    if cfg.settle:  # each move from stance a's own pose to stance b's: q is stance 0's pose
        from rcr.optimiser.v0.move_cache import default_cache, solve_move_cached

        if cache == DEFAULT:
            cache = default_cache()

        for i, (a, b) in enumerate(zip(plan.stances, plan.stances[1:], strict=False)):
            if cache is None:
                yield tr.solve_move(wall, a, b, cfg, index=i)
            else:
                yield solve_move_cached(wall, a, b, cfg, cache, index=i)[0]
        return
    from rcr.optimiser.v0 import move_cache as mc

    solved = {}  # limb -> (layout, start point, configurations) of its last move
    for i, (a, b) in enumerate(zip(plan.stances, plan.stances[1:], strict=False)):
        moving, here = tr.moving_limb(a, b, i), mc.layout(wall, a, b, cfg)
        start = tr.contact_point(wall, a.contacts[moving], cfg)
        last, seed = solved.get(moving), None
        if last is not None and mc.same_layout(last[0], here):
            seed = mc.shifted_seed(q, last[2], start - last[1], cfg)
        result = tr.solve_transition(wall, a, b, q, cfg, index=i, seed=seed)
        solved[moving] = (here, start, result.q)
        q = result.q[-1]
        yield result


def validate_plan(wall: WallModel, plan: ContactPlan) -> None:
    """ValueError, naming the stance, unless the plan is for this wall, has at least two
    stances, all on known holds, and exactly one limb changes between consecutive stances.
    Also unless every hold id is unique: the collision check keys holds by id, so a repeated id
    would hide a hold from it."""
    from collections import Counter

    from rcr.optimiser.v0 import transition as tr

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
    phase_a_duration, phase B's N - K steps share phase_b_duration, and a settle's N steps after
    them (cfg.settle) share settle_duration."""
    n, k_release = cfg.n_samples, cfg.release_sample
    if k > n:
        return cfg.phase_a_duration + cfg.phase_b_duration + cfg.settle_duration * (k - n) / n
    if k <= k_release:
        return cfg.phase_a_duration * k / k_release
    return cfg.phase_a_duration + cfg.phase_b_duration * (k - k_release) / (n - k_release)


def feasibility(wall: WallModel, a: Stance, b: Stance) -> Feasibility:
    """Can the robot go from stance a to stance b? Passed to the planner as FeasibilityFn.

    Cheap checks at the move's edges (edge_checks.py, ~1-3 s, cached), not the full transition:
    failing means it cannot be done; passing means the full solve is worth trying.
    """
    from rcr.optimiser.v0.edge_checks import feasibility as check

    return check(wall, a, b)
