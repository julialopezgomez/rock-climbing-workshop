"""Collision avoidance for the optimiser's problems (transition.py). Off by default for now:
TransitionConfig.avoid_collisions (the runner's --collision), and the two-step way into a socket,
approach_from_above (--approach). Without them the solver (transition.py) never calls in here.

What it adds to a problem, all [Ours]:
    the walls points    the static pose's points out of the wall (wall_margin)
    mid-air             the moving ball out from the wall in the middle of phase B (clearance)
    steering            the first solve only: every hook in a socket lowered in from above (its
                        neck within NECK_CONE of up), the moving ball LIFT above its sockets just
                        after it lets go and just before it arrives
    approach            with approach_from_above: LIFT above its new socket at N-2, SEAT_LIFT
                        above it within SEAT_CONE at N-1, in every solve (a short last drop)
    clearance slots     room for linearised clearances, filled by the collision loop

The collision loop (solve_clear): every robot shape keeps its clearance from every hold mesh and
the wall throughout the motion, except a ball resting in its own socket; a hook may touch, never
enter, its own holds. Exact distances (collision.py) are linearised and re-solved in a trust
region until nothing is violated (sequential convex programming, TrajOpt). Only the first solve is
steered, so it starts in the right basin; the answer is decided by the exact clearances alone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import casadi as ca
import numpy as np

from rcr.optimiser.collision import Check, check_trajectory
from rcr.optimiser.robot_model import NV, RobotModel, load_robot_model
from rcr.robot import hooks
from rcr.types import LIMBS, Limb, Stance, WallModel

if TYPE_CHECKING:
    from rcr.optimiser.transition import TransitionConfig, _Problem, _Solution

# The hold's socket opens upwards: a ball leaves it only near vertical (~10 deg, the organisers'
# sim, wall.py). The first solve keeps each ball's neck within NECK_CONE of up, so it starts with
# the balls lowered in from above; the exact clearances then decide [Ours].
NECK_CONE = np.radians(45.0)
NECK_PROBE = 0.01  # [m] how far back from the ball along its neck (hooks.Hook.neck) it is measured
# Likewise the moving ball starts its path above its sockets: on the sample after it lets go and
# the one before it arrives, at least LIFT above its resting point, inside the opening cone [Ours].
LIFT = 0.035  # [m]: the hold's top is 18.2 mm above the resting ball, + its 12.5 mm radius + 4 mm
# With approach_from_above, the sample before the ball arrives is closer and narrower: SEAT_LIFT
# above its resting point within SEAT_CONE of vertical, so the last interval is a short drop that
# stays inside the socket's mouth. The socket is a tube, not a funnel: 5 mm above the resting
# point the ball touches its rim ~1 mm off its axis, 10 mm above ~3 mm off [Ours, measured on the
# sim's hold mesh]; 10 mm x tan 10 deg = 1.8 mm. 10 deg: a ball can only leave a socket within
# ~10 deg of vertical [the organisers' sim, wall.py EXIT_BLOCK].
SEAT_LIFT = 0.010  # [m]
SEAT_CONE = np.radians(10.0)

OPEN_TRUST_REGION = 10.0  # [rad, m] effectively no trust region
MIN_TRUST_RADIUS = 1e-5  # [rad, m] stop shrinking: the linearisation cannot improve further
SCP_STEP_TOL = 1e-4  # [rad, m] converged when a clear step moves dq less than this ...
SCP_MERIT_TOL = 1e-3  # ... or improves the merit by less than this fraction (0.1 %)
CLEAR_TOL = 1e-5  # [m] summed violation counted as clear (0.01 mm, numerical noise)


# --- Constraints added to the problems -----------------------------------------------------------


def out_of_the_wall(opti: ca.Opti, model: RobotModel, q, cfg: TransitionConfig) -> None:
    """No point of the robot's collision shapes is in the wall (y <= -wall_margin)."""
    opti.subject_to(model.body_points(q)[1, :] <= -cfg.wall_margin)


def neck_rise(model: RobotModel, q_ref, dq, limb: str):
    """[m] how far the hook's neck rises over NECK_PROBE back along it from the
    ball: NECK_PROBE when it points straight up, cos(angle) * NECK_PROBE in general."""
    hook = hooks.HOOKS[limb]
    ball = hooks.ball_centre(hook)
    neck = np.array(hook.neck, dtype=float)
    back = ball - neck / np.linalg.norm(neck) * NECK_PROBE
    point = model.point_jacobian[hook.link]
    return point(q_ref, dq, back)[0][2] - point(q_ref, dq, ball)[0][2]


def neck_from_above(opti: ca.Opti, model: RobotModel, q_ref, dq, limb: str, neck_min) -> None:
    """The hook's neck rises at least neck_min[limb] (see neck_bounds): lowered in from above."""
    opti.subject_to(neck_rise(model, q_ref, dq, limb) >= neck_min[LIMBS.index(limb)])


def neck_bounds(
    model: RobotModel,
    q_ref: np.ndarray,
    seeding: bool,
    fixed_start: bool,
    free: tuple = (),
    end: np.ndarray | None = None,
):
    """neck_min for each limb. Seeding: within NECK_CONE of up, or as far as the fixed start
    already is if it is beyond (a transition cannot move its first sample), and as far as a
    fixed end pose (`end`) is. Otherwise, and for the `free` limbs (hooks not in a socket), no
    bound."""
    if not seeding:
        return np.full(len(LIMBS), -NECK_PROBE)
    want = np.cos(NECK_CONE) * NECK_PROBE
    if not fixed_start:
        bounds = np.full(len(LIMBS), want)
    else:
        now = [float(neck_rise(model, q_ref, np.zeros(NV), limb)) for limb in LIMBS]
        bounds = np.minimum(want, np.array(now) - 1e-6)
    if end is not None:  # nor can it move a pinned last sample
        at_end = [float(neck_rise(model, end, np.zeros(NV), limb)) for limb in LIMBS]
        bounds = np.minimum(bounds, np.array(at_end) - 1e-6)
    bounds[[LIMBS.index(limb) for limb in free]] = -NECK_PROBE
    return bounds


def above_socket(opti: ca.Opti, rel, on, lift: float = LIFT, cone: float = NECK_CONE) -> None:
    """With on = 1 the ball (rel: from its resting point in a socket) is at least `lift` above
    it, within `cone` of vertical; with on = 0 this is always true (within a metre)."""
    opti.subject_to(rel[2] >= on * lift - (1 - on))
    opti.subject_to(rel[0] ** 2 + rel[1] ** 2 <= np.tan(cone) ** 2 * rel[2] ** 2 + (1 - on))


def mid_air(opti: ca.Opti, balls, m: int, k: int, cfg: TransitionConfig) -> None:
    """The moving ball (column m of balls) out from the wall on the middle samples of phase B,
    K+2 .. N-2 (N-3 with approach_from_above), so it can leave and enter its cups."""
    last_out = cfg.n_samples - (3 if cfg.approach_from_above else 2)
    if cfg.release_sample + 2 <= k <= last_out:
        opti.subject_to(balls[1, m] <= -cfg.clearance)


def approach(opti: ca.Opti, balls, m, holds, target, k: int, cfg: TransitionConfig, on) -> None:
    """The moving ball above its sockets at sample k: just out of its old one (K+1), and before
    its new one (N-1; or N-2 and, closer and narrower, N-1 with approach_from_above). `on` is the
    problem's parameter: 1 in the steered first solve (and always with approach_from_above)."""
    n, k_release = cfg.n_samples, cfg.release_sample
    if k == k_release + 1:
        above_socket(opti, balls[:, m] - holds[:, m], on)
    if k == n - 1 and not cfg.approach_from_above:
        above_socket(opti, balls[:, m] - target, on)
    if cfg.approach_from_above and k == n - 2:
        above_socket(opti, balls[:, m] - target, on)
    if cfg.approach_from_above and k == n - 1:
        above_socket(opti, balls[:, m] - target, on, SEAT_LIFT, SEAT_CONE)


def clearance_slots(opti: ca.Opti, dq, cfg: TransitionConfig):
    """Room for linearised clearances, cfg.collision_slots per interval between samples:
        A dq[:, i] + B dq[:, i+1] + s >= rhs,  s >= 0,   cost + w_collision sum(s)
    with A, B, rhs set before each solve (see linearise). The slack keeps every subproblem
    solvable; its weight makes violating a clearance cost more than any torque saving."""
    samples, m = dq.shape[1], cfg.collision_slots
    slots, penalty = [], 0
    for i in range(max(1, samples - 1)):
        a, rhs, slack = opti.parameter(m, NV), opti.parameter(m), opti.variable(m)
        lhs = a @ dq[:, i] + slack
        b = None
        if samples > 1:
            b = opti.parameter(m, NV)
            lhs = lhs + b @ dq[:, i + 1]
        opti.subject_to(lhs >= rhs)
        opti.subject_to(slack >= 0)
        slots.append((a, b, rhs, slack))
        penalty = penalty + cfg.w_collision * ca.sum1(slack)
    centre, radius = opti.parameter(NV, samples), opti.parameter()
    opti.subject_to(opti.bounded(centre - radius, dq, centre + radius))
    return slots, penalty, centre, radius


# --- The collision loop --------------------------------------------------------------------------


def transition_contacts(a: Stance, b: Stance, moving: Limb, cfg: TransitionConfig):
    """Per sample: the hold each limb's ball rests in (the moving one: its old hold up to K,
    nothing in the air, its new hold at N)."""
    n, k_release = cfg.n_samples, cfg.release_sample
    out = []
    for k in range(n + 1):
        contacts = dict(a.contacts)
        if k > k_release:
            contacts[moving] = b.contacts[moving] if k == n else None
        out.append(contacts)
    return out


def own_holds(a: Stance, b: Stance) -> dict[str, set[str]]:
    """The holds each limb's hook may touch during a -> b: its own, old and new."""
    return {limb: {a.contacts[limb], b.contacts[limb]} for limb in LIMBS}


def checker(wall: WallModel, q_ref, contacts, own, cfg: TransitionConfig):
    """check(dq) -> the clearances of the motion integrate(q_ref, dq) (collision.py); None
    without cfg.avoid_collisions (nothing is checked)."""
    if not cfg.avoid_collisions:
        return None
    model, holds = load_robot_model(), {h.id: h for h in wall.holds}

    def check(dq: np.ndarray) -> list[Check]:
        return check_trajectory(model, holds, q_ref, dq, contacts, own, cfg.collision_clearance)

    return check


def linearise(p: _Problem, checks: list[Check], dq_ref: np.ndarray, cfg: TransitionConfig):
    """Fill each interval's slots with its tightest clearances, linearised at dq_ref:
    value + grad_a (dq_i - dq_ref_i) + grad_b (dq_i+1 - dq_ref_i+1) >= required + aim."""
    m, by_interval = cfg.collision_slots, {}
    for check in checks:
        by_interval.setdefault(check.interval, []).append(check)
    for i, (a, b, rhs, _) in enumerate(p.slots):
        chosen = sorted(by_interval.get(i, []), key=lambda c: c.gap)[:m]
        rows_a, rows_b, values = np.zeros((m, NV)), np.zeros((m, NV)), np.full(m, -1.0)
        for j, c in enumerate(chosen):
            rows_a[j], rows_b[j] = c.grad_a, c.grad_b
            values[j] = c.required + cfg.collision_clearance.aim - c.value + c.grad_a @ dq_ref[:, i]
            if b is not None:
                values[j] += c.grad_b @ dq_ref[:, i + 1]
        p.opti.set_value(a, rows_a)
        p.opti.set_value(rhs, values)  # unused slots: 0 >= -1, always true
        if b is not None:
            p.opti.set_value(b, rows_b)


def violation(checks: list[Check]) -> float:
    """[m] summed amount by which clearances are violated (0 when the motion is clear)."""
    return float(sum(max(0.0, -c.gap) for c in checks))


def solve_clear(
    p: _Problem,
    what: str,
    cfg: TransitionConfig,
    check,
    strict: bool = True,
    seed: np.ndarray | None = None,
) -> _Solution:
    """Solve, then make the motion clear of the holds and the wall by sequential convex
    programming [TrajOpt, Schulman et al. 2014]:
        1. solve without clearances, but with every hook in a socket lowered in from above
           (NECK_CONE): a balanced, feasible start in the right basin;
        2. measure every clearance exactly along the motion (check(dq) -> collision.Check list);
        3. linearise the tightest into the slots and re-solve within a trust region;
        4. keep the step if the exact merit (objective + w_collision * violation) improves as
           the linear model promised; grow or shrink the trust region; repeat.
    With a `seed` (a solved motion reused, already in the right basin), step 1 is replaced by a
    first SCP step from it: its clearances linearised, within the trust region, unsteered.
    OptimisationError if the motion cannot be made clear (strict), else the least violating."""
    from rcr.optimiser.transition import (
        FIRST_SOLVE_PATIENCE,
        OptimisationError,
        _read,
        solve_opti,
    )

    model, q_ref = load_robot_model(), np.array(p.opti.value(p.q_ref)).ravel()
    fixed_start = len(p.q) > 1  # a transition's first sample is its given start
    steer = seed is None
    if seed is not None:
        linearise(p, check(seed), seed, cfg)
        p.opti.set_value(p.centre, seed)
        p.opti.set_value(p.radius, cfg.trust_radius)
    else:
        linearise(p, [], np.zeros((NV, len(p.q))), cfg)  # no clearances yet
        p.opti.set_value(p.centre, np.zeros((NV, len(p.q))))
        p.opti.set_value(p.radius, OPEN_TRUST_REGION)
    p.opti.set_value(
        p.neck_min, neck_bounds(model, q_ref, steer, fixed_start, p.free_neck, p.end_q)
    )
    if p.approach is not None:
        p.opti.set_value(p.approach, 1.0 if steer or cfg.approach_from_above else 0.0)
    best = _read(p, *solve_opti(p.opti, what, FIRST_SOLVE_PATIENCE))
    p.opti.set_value(
        p.neck_min, neck_bounds(model, q_ref, False, fixed_start, p.free_neck, p.end_q)
    )
    if p.approach is not None:
        p.opti.set_value(p.approach, 1.0 if cfg.approach_from_above else 0.0)
    checks = check(best.dq)
    violated, radius = violation(checks), cfg.trust_radius
    polished = 0
    for _ in range(cfg.scp_iterations):
        if violated <= CLEAR_TOL:
            if polished >= cfg.scp_polish:
                break  # clear, and the cost has had its chance to come down
            polished += 1
        linearise(p, checks, best.dq, cfg)
        p.opti.set_value(p.centre, best.dq)
        p.opti.set_value(p.radius, radius)
        p.opti.set_initial(p.opti.x, best.variables)
        try:
            step = _read(p, *solve_opti(p.opti, what))
        except OptimisationError:
            radius /= 4  # the linear model is poor here: a smaller step
            if radius < MIN_TRUST_RADIUS:
                break
            continue
        step_checks = check(step.dq)
        step_violation = violation(step_checks)
        merit = best.objective + cfg.w_collision * violated
        promised = merit - (step.objective + cfg.w_collision * step.slack)
        achieved = merit - (step.objective + cfg.w_collision * step_violation)
        if achieved > 0.1 * max(promised, 0.0) and achieved >= 0.0:
            moved = float(np.abs(step.dq - best.dq).max())
            best, checks, violated = step, step_checks, step_violation
            if achieved > 0.75 * promised:
                radius = min(4 * radius, OPEN_TRUST_REGION)
            if violated <= CLEAR_TOL and (moved < SCP_STEP_TOL or achieved < SCP_MERIT_TOL * merit):
                break
        else:
            radius /= 4
            if radius < MIN_TRUST_RADIUS:
                break
    if strict and violated > CLEAR_TOL:
        worst = min(checks, key=lambda c: c.gap)
        raise OptimisationError(
            f"{what}: not clear of the holds and wall: {worst.label} is {-worst.gap * 1e3:.2f} mm "
            f"short of its clearance (interval {worst.interval}, {worst.tau:.2f} along)"
        )
    return best
