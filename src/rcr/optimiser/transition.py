"""Quasi-static transitions: one small NLP (CasADi Opti + IPOPT) per change of hold.

Quasi-static: no inertia, every sample k is a statically balanced pose. Per sample:
    dq_k   the configuration, as a tangent vector around the transition's start q0:
           q_k = integrate(q0, dq_k), so the trunk quaternion stays a unit quaternion
    f_k^c  the force [N] that hold c exerts on the robot's ball, one per limb in contact
Torques are not variables. The 19 motor rows of the static balance give them,
    tau_k = g(q_k) - sum_c J_c(q_k)^T f_k^c                                   [N·m]
and the 6 trunk rows must balance:
    sum_c f_k^c + m g = 0   and   sum_c (x_c - com) x f_k^c = 0   (moments about the CoM).

Two phases, samples k = 0..N:
    A (k = 0..K): four contacts; the body shifts until the moving limb carries nothing at K.
    B (k = K..N): three contacts; the moving limb travels and reaches its new hold at N.
Cost: w_tau sum ||tau_k||^2 + w_smooth sum ||dq_{k+1} - dq_k||^2.

Clear of the holds and the wall (collision.py, _solve_clear): every robot shape keeps its
clearance from every hold mesh and the wall throughout the motion, except a ball resting in its
own socket; a hook may touch, never enter, its own holds. Exact distances are linearised and
re-solved in a trust region until nothing is violated (sequential convex programming, TrajOpt).
Only the first solve is steered (hooks lowered into sockets from above, NECK_CONE and LIFT), so
it starts in the right basin; the answer is decided by the exact clearances alone.

Each problem is built once and re-solved with new numbers (its start, holds and target are
CasADi parameters): building the derivatives costs ~3x more than solving. A transition's
structure depends only on which limb moves, so there are at most four, plus the static pose.

Wall frame (INTERFACES.md): x right, z up, y into the wall, the robot at y < 0, gravity -z.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import casadi as ca
import numpy as np

from rcr.optimiser import robot_model as rm
from rcr.optimiser.collision import Check, Clearance, check_trajectory
from rcr.optimiser.robot_model import GRAVITY, NJ, NQ, NV, RobotModel, load_robot_model
from rcr.types import LIMBS, Limb, Stance, WallModel

SERVO_TORQUE_LIMIT = 0.64  # [N·m] 1.75 A x 0.366 N·m/A [Microban constants.py]
GUESS_STANDOFF = -0.10  # [m] trunk y in the first static pose's guess, in front of the wall [Ours]
FACING_WALL = (np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4))  # (w,x,y,z): trunk front along +y
# The hold's socket opens upwards: a 9 mm finger leaves the resting ball only within ~60 deg of
# straight up (measured on hold.stl). The first solve keeps each hook's neck within NECK_CONE of
# up, so it starts with the hooks lowered in from above; the exact clearances then decide [Ours].
NECK_CONE = np.radians(45.0)
NECK_PROBE = 0.01  # [m] how far back along the hook's last segment its direction is measured
# Likewise the moving ball starts its path above its sockets: on the sample after it lets go and
# the one before it arrives, at least LIFT above its resting point, inside the opening cone [Ours].
LIFT = 0.035  # [m]: the hold's top is 20.7 mm above the resting ball, + its 10 mm radius + 4 mm


class OptimisationError(RuntimeError):
    """IPOPT did not find a solution; the message names the problem, status and violation."""


@dataclass
class TransitionConfig:
    n_samples: int = 12  # N: samples k = 0..N per transition [Ours]
    release_sample: int = 4  # K: end of phase A, where the moving limb's force reaches 0 [Ours]
    tau_limit: float = 0.7 * SERVO_TORQUE_LIMIT  # [N·m] 70 % of the servo limit [Ours]
    # Friction pyramid per limb: contact normal (the direction the hold pushes the ball) and mu.
    # Feet: the wall's normal, out of the wall (-y). Hands: up (+z), the hold supports the hand.
    # Both are placeholders until the real holds and climbing hands are published [Ours].
    normals: dict[str, tuple[float, float, float]] = field(
        default_factory=lambda: {
            "right_hand": (0.0, 0.0, 1.0),
            "left_hand": (0.0, 0.0, 1.0),
            "right_foot": (0.0, -1.0, 0.0),
            "left_foot": (0.0, -1.0, 0.0),
        }
    )
    mu: dict[str, float] = field(default_factory=lambda: dict.fromkeys(LIMBS, 0.8))  # [Ours]
    # The moving limb keeps this far out from the wall face (y <= -clearance) in the middle of
    # phase B, i.e. on samples K+2..N-2, so it can leave and enter its cups [Ours].
    # 0.06 m: the hold sticks out 43 mm [hold.stl], the ball is 10 mm in radius, + 7 mm, so the
    # ball can pass in front of a hold (the task's first value, 0.04 m, could not).
    clearance: float = 0.06  # [m]
    # Every point of Microban's collision shapes stays this far out of the wall face, in every
    # sample (y <= -wall_margin). Without it the trunk can sink into the wall [Ours].
    wall_margin: float = 0.002  # [m]
    # Ball centre relative to Hold.position (the bolt midpoint on the wall face): the bottom of
    # the hold's spherical socket, 26 mm out and 14 mm below the bolts [Rules Figure 3], lowered
    # by the socket/ball clearance (12.7 - 10 mm). The same point as rcr.sim's grasp point.
    contact_offset: tuple[float, float, float] = (0.0, -0.026, -0.0167)  # [m]
    # How far the trunk may move from the solve's reference configuration (a static pose's
    # guess, or a transition's start): keeps the exponential map away from wrapping at pi.
    max_trunk_shift: float = 0.3  # [m] per axis [Ours]
    max_trunk_turn: float = 1.2  # [rad] per axis of the rotation vector [Ours]
    w_tau: float = 1.0  # cost weight on torques^2 [1/(N·m)^2]
    w_smooth: float = 1.0  # cost weight on (dq_{k+1} - dq_k)^2 [1/rad^2, 1/m^2]
    w_stay: float = 1e-3  # static pose only: weight on dq^2, keeps it near the guess [Ours]
    # Clear of the holds and the wall throughout the motion (collision.py), by sequential convex
    # programming on the exact clearances: see _solve_clear.
    avoid_collisions: bool = True
    collision_clearance: Clearance = field(default_factory=Clearance)
    collision_slots: int = 40  # linearised clearances per interval between samples [Ours]
    w_collision: float = 1e3  # [1/m] penalty on violated clearance (an exact penalty) [Ours]
    trust_radius: float = 0.05  # [rad, m] first trust region per component of dq [Ours]
    scp_iterations: int = 30  # most re-solves for the clearances [Ours]
    scp_polish: int = 3  # once clear, at most this many more re-solves to lower the cost [Ours]
    phase_a_duration: float = 1.0  # [s] for phase A, chosen after solving (quasi-static) [Ours]
    phase_b_duration: float = 2.0  # [s] for phase B [Ours]
    ipopt: dict = field(
        default_factory=lambda: {"print_level": 0, "max_iter": 1000, "tol": 1e-6, "sb": "yes"}
    )


@dataclass
class TransitionResult:
    q: np.ndarray  # (N+1) x NQ configurations
    dq: np.ndarray  # NV x (N+1) the tangent vectors at the start: q[k] = integrate(q[0], dq[:, k])
    tau: np.ndarray  # (N+1) x 19 motor torques [N·m]
    forces: dict[str, np.ndarray]  # limb -> (N+1) x 3 hold forces [N]; 0 where not in contact
    moving: str  # the limb that changed hold
    solve_time: float  # [s] IPOPT wall time
    iterations: int


# --- Inputs -------------------------------------------------------------------------------------


def moving_limb(a: Stance, b: Stance, index: int) -> Limb:
    """The one limb whose hold differs between a and b. ValueError naming `index` otherwise."""
    changed = [limb for limb in LIMBS if a.contacts[limb] != b.contacts[limb]]
    if len(changed) != 1:
        raise ValueError(
            f"transition {index}: exactly one limb must change hold, but {changed or 'none'} do"
        )
    return changed[0]


def check_holds(wall: WallModel, stance: Stance, index: int) -> None:
    """Every limb is on a hold, and every hold exists in the wall. ValueError otherwise."""
    known = {h.id for h in wall.holds}
    for limb in LIMBS:
        hold = stance.contacts[limb]
        if hold is None:
            raise ValueError(f"stance {index}: {limb} is free; every stance needs four holds")
        if hold not in known:
            raise ValueError(f"stance {index}: {limb} is on {hold!r}, which is not in the wall")


def contact_point(wall: WallModel, hold_id: str, cfg: TransitionConfig) -> np.ndarray:
    """[m] where a ball centre sits on this hold, in the wall frame."""
    hold = next(h for h in wall.holds if h.id == hold_id)
    return np.array(hold.position) + np.array(cfg.contact_offset)


def contact_points(wall: WallModel, stance: Stance, cfg: TransitionConfig) -> np.ndarray:
    """3x4 contact points of a stance, one column per limb in LIMBS order."""
    return np.array([contact_point(wall, stance.contacts[limb], cfg) for limb in LIMBS]).T


# --- Physics of one sample ----------------------------------------------------------------------


def static_balance(model: RobotModel, q, forces: dict[str, ca.MX]):
    """(trunk wrench residual (6), motor torques (19)) for configuration q and hold forces.

    The residual is the net force and the net moment about the CoM; it is zero in balance.
    """
    balls, com = model.limb_positions(q), model.com(q)
    net_force = ca.DM([0.0, 0.0, -model.mass * GRAVITY])
    net_moment = ca.DM.zeros(3)
    tau = model.gravity_torques(q)
    for limb, force in forces.items():
        i = LIMBS.index(limb)
        net_force = net_force + force
        net_moment = net_moment + ca.cross(balls[:, i] - com, force)
        tau = tau - model.limb_jacobian(q, limb).T @ force
    return ca.vertcat(net_force, net_moment), tau


def friction_pyramid(opti: ca.Opti, force, normal, mu: float) -> None:
    """The hold can only push along its normal, with tangential force at most mu times that."""
    n = np.array(normal, dtype=float) / np.linalg.norm(normal)
    t1 = np.cross(n, [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0])
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    pushing = ca.dot(force, n)
    opti.subject_to(pushing >= 0)
    for t in (t1, t2):  # |f.t| <= mu f.n, as two smooth (linear) inequalities
        opti.subject_to(opti.bounded(-mu * pushing, ca.dot(force, t), mu * pushing))


def joint_and_torque_limits(opti: ca.Opti, model: RobotModel, q, tau, cfg: TransitionConfig):
    opti.subject_to(opti.bounded(model.lower, q[7:], model.upper))
    opti.subject_to(opti.bounded(-cfg.tau_limit, tau, cfg.tau_limit))


def trunk_bounds(opti: ca.Opti, dq, cfg: TransitionConfig) -> None:
    """The trunk stays within max_trunk_shift / max_trunk_turn of the reference configuration."""
    opti.subject_to(opti.bounded(-cfg.max_trunk_shift, dq[0:3], cfg.max_trunk_shift))
    opti.subject_to(opti.bounded(-cfg.max_trunk_turn, dq[3:6], cfg.max_trunk_turn))


def out_of_the_wall(opti: ca.Opti, model: RobotModel, q, cfg: TransitionConfig) -> None:
    """No point of the robot's collision shapes is in the wall (y <= -wall_margin)."""
    opti.subject_to(model.body_points(q)[1, :] <= -cfg.wall_margin)


def balanced_pose(opti, model, dq, q_ref, forces, cfg):
    """Constrain one sample to a statically balanced, allowed pose. Returns (q, tau)."""
    trunk_bounds(opti, dq, cfg)
    q = model.integrate(q_ref, dq)
    wrench, tau = static_balance(model, q, forces)
    opti.subject_to(wrench == 0)
    joint_and_torque_limits(opti, model, q, tau, cfg)
    out_of_the_wall(opti, model, q, cfg)
    return q, tau


# --- The problems, built once -------------------------------------------------------------------


@dataclass
class _Problem:
    """One NLP, built once and re-solved with new parameter values."""

    opti: ca.Opti
    dq: ca.MX  # NV x samples
    forces: dict[str, ca.MX]  # limb -> 3 x samples
    q: list  # per sample, NQ expressions
    tau: list  # per sample, 19 expressions
    objective: ca.MX  # the cost without the collision penalty
    neck_min: ca.MX  # parameter 4: the least rise of each hook's neck, LIMBS order (seed only)
    q_ref: ca.MX  # parameter NQ: the configuration dq is measured from
    holds: ca.MX  # parameter 3x4: the contact points of the start stance, LIMBS order
    slots: list  # per interval: (A, B, rhs, slack) for the linearised clearances, see below
    centre: ca.MX  # parameter NV x samples: the trust region's centre
    radius: ca.MX  # parameter: the trust region's half-width on every component of dq
    target: ca.MX | None = None  # parameter 3: the moving limb's new contact point
    approach: ca.MX | None = None  # parameter, 1 or 0: the moving ball above its sockets (seed)


_problems: dict[tuple, _Problem] = {}


def _cached(key: tuple, build) -> _Problem:
    if key not in _problems:
        _problems[key] = build()
    return _problems[key]


def above_socket(opti: ca.Opti, rel, on) -> None:
    """With on = 1 the ball (rel: from its resting point in a socket) is at least LIFT above it,
    inside the socket's opening cone; with on = 0 this is always true (within a metre)."""
    opti.subject_to(rel[2] >= on * LIFT - (1 - on))
    opti.subject_to(rel[0] ** 2 + rel[1] ** 2 <= np.tan(NECK_CONE) ** 2 * rel[2] ** 2 + (1 - on))


def neck_rise(model: RobotModel, q_ref, dq, limb: str):
    """[m] how far the hook's neck rises over NECK_PROBE back along its last segment from the
    ball: NECK_PROBE when it points straight up, cos(angle) * NECK_PROBE in general."""
    hook = rm.HOOKS[limb]
    ball = rm.hook_ball_centre(hook)
    back = ball - np.array(hook.segments[-1][0], dtype=float) * NECK_PROBE
    point = model.point_jacobian[hook.link]
    return point(q_ref, dq, back)[0][2] - point(q_ref, dq, ball)[0][2]


def neck_from_above(opti: ca.Opti, model: RobotModel, q_ref, dq, limb: str, neck_min) -> None:
    """The hook's neck rises at least neck_min[limb] (see _neck_bounds): lowered in from above."""
    opti.subject_to(neck_rise(model, q_ref, dq, limb) >= neck_min[LIMBS.index(limb)])


def _neck_bounds(model: RobotModel, q_ref: np.ndarray, seeding: bool, fixed_start: bool):
    """neck_min for each limb. Seeding: within NECK_CONE of up, or as far as the fixed start
    already is if it is beyond (a transition cannot move its first sample). Otherwise no bound."""
    if not seeding:
        return np.full(len(LIMBS), -NECK_PROBE)
    want = np.cos(NECK_CONE) * NECK_PROBE
    if not fixed_start:
        return np.full(len(LIMBS), want)
    now = [float(neck_rise(model, q_ref, np.zeros(NV), limb)) for limb in LIMBS]
    return np.minimum(want, np.array(now) - 1e-6)


def clearance_slots(opti: ca.Opti, dq, cfg: TransitionConfig):
    """Room for linearised clearances, cfg.collision_slots per interval between samples:
        A dq[:, i] + B dq[:, i+1] + s >= rhs,  s >= 0,   cost + w_collision sum(s)
    with A, B, rhs set before each solve (see _linearise). The slack keeps every subproblem
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


def static_pose_problem(cfg: TransitionConfig) -> _Problem:
    """One balanced pose with all four balls on their holds; cost tau^2 + w_stay dq^2."""

    def build() -> _Problem:
        model, opti = load_robot_model(), ca.Opti()
        q_ref, holds = opti.parameter(NQ), opti.parameter(3, len(LIMBS))
        dq = opti.variable(NV, 1)
        forces = {limb: opti.variable(3, 1) for limb in LIMBS}
        neck_min = opti.parameter(len(LIMBS))
        q, tau = balanced_pose(opti, model, dq[:, 0], q_ref, forces, cfg)
        opti.subject_to(model.limb_positions(q) == holds)
        for limb in LIMBS:
            friction_pyramid(opti, forces[limb], cfg.normals[limb], cfg.mu[limb])
            neck_from_above(opti, model, q_ref, dq[:, 0], limb, neck_min)
        objective = cfg.w_tau * ca.sumsqr(tau) + cfg.w_stay * ca.sumsqr(dq)
        slots, penalty, centre, radius = clearance_slots(opti, dq, cfg)
        opti.minimize(objective + penalty)
        _set_solver(opti, cfg)
        return _Problem(
            opti, dq, forces, [q], [tau], objective, neck_min, q_ref, holds, slots, centre, radius
        )

    return _cached(("static", repr(cfg)), build)


def transition_problem(moving: Limb, cfg: TransitionConfig) -> _Problem:
    """The two-phase transition of one moving limb; see the module docstring."""
    n, k_release = cfg.n_samples, cfg.release_sample
    if not 0 < k_release < n - 2:
        raise ValueError(f"need 0 < release_sample < n_samples - 2, got {k_release}, {n}")

    def build() -> _Problem:
        model, opti = load_robot_model(), ca.Opti()
        q0, holds, target = opti.parameter(NQ), opti.parameter(3, len(LIMBS)), opti.parameter(3)
        dq = opti.variable(NV, n + 1)
        forces = {limb: opti.variable(3, n + 1) for limb in LIMBS}
        opti.subject_to(dq[:, 0] == 0)
        neck_min, approach = opti.parameter(len(LIMBS)), opti.parameter()
        m = LIMBS.index(moving)
        qs, taus = [], []
        for k in range(n + 1):
            sample = {limb: forces[limb][:, k] for limb in LIMBS}
            q, tau = balanced_pose(opti, model, dq[:, k], q0, sample, cfg)
            balls = model.limb_positions(q)
            for i, limb in enumerate(LIMBS):
                if limb != moving or k <= k_release or k == n:  # its ball is in a socket
                    neck_from_above(opti, model, q0, dq[:, k], limb, neck_min)
                if limb != moving or k <= k_release:
                    opti.subject_to(balls[:, i] == holds[:, i])  # on its hold of the start stance
                if limb != moving or k < k_release:
                    friction_pyramid(opti, sample[limb], cfg.normals[limb], cfg.mu[limb])
                else:
                    opti.subject_to(sample[limb] == 0)  # f_K^moving = 0, and nothing after
            if k_release + 2 <= k <= n - 2:
                opti.subject_to(balls[1, m] <= -cfg.clearance)  # away from the wall in mid-air
            if k == n:
                opti.subject_to(balls[:, m] == target)  # the moving limb arrives
            if k in (k_release + 1, n - 1):  # seed only: just out of / just before its sockets
                above_socket(opti, balls[:, m] - (holds[:, m] if k < n - 1 else target), approach)
            qs.append(q)
            taus.append(tau)
        smooth = sum(ca.sumsqr(dq[:, k + 1] - dq[:, k]) for k in range(n))
        objective = cfg.w_tau * sum(ca.sumsqr(t) for t in taus) + cfg.w_smooth * smooth
        slots, penalty, centre, radius = clearance_slots(opti, dq, cfg)
        opti.minimize(objective + penalty)
        _set_solver(opti, cfg)
        return _Problem(
            opti,
            dq,
            forces,
            qs,
            taus,
            objective,
            neck_min,
            q0,
            holds,
            slots,
            centre,
            radius,
            target,
            approach,
        )

    return _cached(("transition", moving, repr(cfg)), build)


def _set_solver(opti: ca.Opti, cfg: TransitionConfig) -> None:
    opti.solver("ipopt", {"print_time": False, "expand": True}, cfg.ipopt)


# --- Solving ------------------------------------------------------------------------------------


def solve_opti(opti: ca.Opti, what: str) -> tuple[ca.OptiSol, float]:
    """Run IPOPT: (solution, wall time [s]). OptimisationError with the status and the
    largest constraint violation if it fails."""
    start = time.perf_counter()
    try:
        return opti.solve(), time.perf_counter() - start
    except RuntimeError:
        g = np.atleast_1d(opti.debug.value(opti.g))
        lbg = np.atleast_1d(opti.debug.value(opti.lbg))
        ubg = np.atleast_1d(opti.debug.value(opti.ubg))
        violation = float(np.max(np.concatenate([lbg - g, g - ubg, [0.0]])))
        status = opti.debug.stats().get("return_status", "unknown")
        raise OptimisationError(
            f"{what}: IPOPT {status}, largest constraint violation {violation:.2e}"
        ) from None


@dataclass
class _Solution:
    """What one accepted solve gave, read out of CasADi."""

    dq: np.ndarray  # NV x samples
    q: np.ndarray  # samples x NQ
    tau: np.ndarray  # samples x 19
    forces: dict[str, np.ndarray]  # limb -> samples x 3
    objective: float
    slack: float  # [m] summed slack of the linearised clearances
    variables: np.ndarray  # all of the NLP's variables, to warm start the next solve
    solve_time: float
    iterations: int


def _read(p: _Problem, sol: ca.OptiSol, solve_time: float) -> _Solution:
    return _Solution(
        dq=np.array(sol.value(p.dq)).reshape(NV, -1),
        q=np.array([np.array(sol.value(q)).ravel() for q in p.q]),
        tau=np.array([np.array(sol.value(t)).ravel() for t in p.tau]),
        forces={limb: np.array(sol.value(p.forces[limb])).reshape(3, -1).T for limb in LIMBS},
        objective=float(sol.value(p.objective)),
        slack=float(sum(np.sum(sol.value(slot[3])) for slot in p.slots)),
        variables=np.array(sol.value(p.opti.x)).ravel(),
        solve_time=solve_time,
        iterations=int(sol.stats().get("iter_count", 0)),
    )


def _linearise(p: _Problem, checks: list[Check], dq_ref: np.ndarray, cfg: TransitionConfig):
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


def _violation(checks: list[Check]) -> float:
    """[m] summed amount by which clearances are violated (0 when the motion is clear)."""
    return float(sum(max(0.0, -c.gap) for c in checks))


def _solve_clear(p: _Problem, what: str, cfg: TransitionConfig, check) -> _Solution:
    """Solve, then make the motion clear of the holds and the wall by sequential convex
    programming [TrajOpt, Schulman et al. 2014]:
        1. solve without clearances, but with every hook in a socket lowered in from above
           (NECK_CONE): a balanced, feasible start in the right basin;
        2. measure every clearance exactly along the motion (check(dq) -> collision.Check list);
        3. linearise the tightest into the slots and re-solve within a trust region;
        4. keep the step if the exact merit (objective + w_collision * violation) improves as
           the linear model promised; grow or shrink the trust region; repeat.
    OptimisationError if the motion cannot be made clear."""
    _linearise(p, [], np.zeros((NV, len(p.q))), cfg)  # no clearances yet
    p.opti.set_value(p.centre, np.zeros((NV, len(p.q))))
    p.opti.set_value(p.radius, OPEN_TRUST_REGION)
    model, q_ref = load_robot_model(), np.array(p.opti.value(p.q_ref)).ravel()
    fixed_start = len(p.q) > 1  # a transition's first sample is its given start
    p.opti.set_value(p.neck_min, _neck_bounds(model, q_ref, cfg.avoid_collisions, fixed_start))
    if p.approach is not None:
        p.opti.set_value(p.approach, 1.0 if cfg.avoid_collisions else 0.0)
    best = _read(p, *solve_opti(p.opti, what))
    if not cfg.avoid_collisions:
        return best
    p.opti.set_value(p.neck_min, _neck_bounds(model, q_ref, False, fixed_start))  # exact only now
    if p.approach is not None:
        p.opti.set_value(p.approach, 0.0)
    checks = check(best.dq)
    violation, radius = _violation(checks), cfg.trust_radius
    polished = 0
    for _ in range(cfg.scp_iterations):
        if violation <= CLEAR_TOL:
            if polished >= cfg.scp_polish:
                break  # clear, and the cost has had its chance to come down
            polished += 1
        _linearise(p, checks, best.dq, cfg)
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
        step_violation = _violation(step_checks)
        merit = best.objective + cfg.w_collision * violation
        promised = merit - (step.objective + cfg.w_collision * step.slack)
        achieved = merit - (step.objective + cfg.w_collision * step_violation)
        if achieved > 0.1 * max(promised, 0.0) and achieved >= 0.0:
            moved = float(np.abs(step.dq - best.dq).max())
            best, checks, violation = step, step_checks, step_violation
            if achieved > 0.75 * promised:
                radius = min(4 * radius, OPEN_TRUST_REGION)
            if violation <= CLEAR_TOL and (
                moved < SCP_STEP_TOL or achieved < SCP_MERIT_TOL * merit
            ):
                break
        else:
            radius /= 4
            if radius < MIN_TRUST_RADIUS:
                break
    if violation > CLEAR_TOL:
        worst = min(checks, key=lambda c: c.gap)
        raise OptimisationError(
            f"{what}: not clear of the holds and wall: {worst.label} is {-worst.gap * 1e3:.2f} mm "
            f"short of its clearance (interval {worst.interval}, {worst.tau:.2f} along)"
        )
    return best


OPEN_TRUST_REGION = 10.0  # [rad, m] effectively no trust region
MIN_TRUST_RADIUS = 1e-5  # [rad, m] stop shrinking: the linearisation cannot improve further
SCP_STEP_TOL = 1e-4  # [rad, m] converged when a clear step moves dq less than this ...
SCP_MERIT_TOL = 1e-3  # ... or improves the merit by less than this fraction (0.1 %)
CLEAR_TOL = 1e-5  # [m] summed violation counted as clear (0.01 mm, numerical noise)


def _checker(wall: WallModel, q_ref, contacts, own, cfg: TransitionConfig):
    """check(dq) -> the clearances of the motion integrate(q_ref, dq) (collision.py)."""
    model, holds = load_robot_model(), {h.id: h for h in wall.holds}

    def check(dq: np.ndarray) -> list[Check]:
        return check_trajectory(model, holds, q_ref, dq, contacts, own, cfg.collision_clearance)

    return check


def solve_static_pose(
    wall: WallModel,
    stance: Stance,
    q_guess: np.ndarray,
    cfg: TransitionConfig | None = None,
    index: int = 0,
) -> np.ndarray:
    """A statically balanced configuration (NQ) with every ball on its hold, clear of the holds
    and the wall.

    The same constraints as one sample of a transition, all four limbs in contact; cost
    w_tau ||tau||^2 + w_stay ||dq||^2, dq measured from q_guess. Cheap enough to serve as a
    feasibility check.
    """
    return static_pose(wall, stance, q_guess, cfg or TransitionConfig(), index).q[0]


def static_pose(wall, stance, q_ref, cfg: TransitionConfig, index: int = 0) -> _Solution:
    """The full solution of solve_static_pose (one sample): q, tau, forces, dq from q_ref."""
    check_holds(wall, stance, index)
    p = static_pose_problem(cfg)
    p.opti.set_value(p.q_ref, q_ref)
    p.opti.set_value(p.holds, contact_points(wall, stance, cfg))
    p.opti.set_initial(p.dq, np.zeros((NV, 1)))
    for limb in LIMBS:
        p.opti.set_initial(p.forces[limb], weight_share(len(LIMBS)))
    contacts = [dict(stance.contacts)]
    own = {limb: {stance.contacts[limb]} for limb in LIMBS}
    check = _checker(wall, q_ref, contacts, own, cfg)
    return _solve_clear(p, f"static pose for stance {index}", cfg, check)


def solve_transition(
    wall: WallModel,
    a: Stance,
    b: Stance,
    q0: np.ndarray,
    cfg: TransitionConfig | None = None,
    index: int = 0,
) -> TransitionResult:
    """Move the one limb that changes from its hold in `a` to its hold in `b`, starting at q0.

    q0 must have every ball on its hold of `a` (e.g. from solve_static_pose or the previous
    transition). Only the moving limb's final point is fixed, not the final configuration.
    The motion stays clear of the holds and the wall throughout (cfg.avoid_collisions).
    """
    cfg = cfg or TransitionConfig()
    check_holds(wall, a, index)
    check_holds(wall, b, index + 1)
    moving = moving_limb(a, b, index)
    what = f"transition {index} ({moving} to {b.contacts[moving]})"
    p = transition_problem(moving, cfg)
    p.opti.set_value(p.q_ref, q0)
    p.opti.set_value(p.holds, contact_points(wall, a, cfg))
    p.opti.set_value(p.target, contact_point(wall, b.contacts[moving], cfg))
    try:
        _warm_start(p, wall, b, q0, moving, cfg, index)
    except OptimisationError as e:
        raise OptimisationError(f"{what}: the robot cannot hold stance {index + 1} ({e})") from e
    check = _checker(wall, q0, transition_contacts(a, b, moving, cfg), _own_holds(a, b), cfg)
    s = _solve_clear(p, what, cfg, check)
    return TransitionResult(
        q=s.q, dq=s.dq, tau=s.tau, forces=s.forces, moving=moving,
        solve_time=s.solve_time, iterations=s.iterations,
    )  # fmt: skip


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


def _own_holds(a: Stance, b: Stance) -> dict[str, set[str]]:
    """The holds each limb's hook may touch during a -> b: its own, old and new."""
    return {limb: {a.contacts[limb], b.contacts[limb]} for limb in LIMBS}


def _warm_start(p: _Problem, wall, b, q0, moving, cfg, index) -> None:
    """dq: still in phase A, then a straight line (in the tangent space at q0) to a static pose
    of stance b; forces: the weight shared over the limbs carrying it."""
    dq_end = static_pose(wall, b, q0, cfg, index + 1).dq[:, 0]
    n, k_release = cfg.n_samples, cfg.release_sample
    for k in range(n + 1):
        s = max(0.0, (k - k_release) / (n - k_release))
        p.opti.set_initial(p.dq[:, k], s * dq_end)
        carrying = [limb for limb in LIMBS if limb != moving or k < k_release]
        for limb in LIMBS:
            share = weight_share(len(carrying)) if limb in carrying else np.zeros(3)
            p.opti.set_initial(p.forces[limb][:, k], share)


def weight_share(n_limbs: int) -> np.ndarray:
    """[N] warm start for one hold's force: an equal share of the robot's weight, upwards."""
    return np.array([0.0, 0.0, load_robot_model().mass * GRAVITY / n_limbs])


def initial_guess(wall: WallModel, stance: Stance, cfg: TransitionConfig | None = None):
    """A rough configuration to start solve_static_pose from: the trunk in front of the
    stance's holds, upright and facing the wall, all motors at zero [Ours]."""
    cfg = cfg or TransitionConfig()
    points = contact_points(wall, stance, cfg)
    model = load_robot_model()
    q = np.zeros(7 + NJ)
    q[0:3] = [points[0].mean(), GUESS_STANDOFF, points[2].mean()]
    q[3:7] = FACING_WALL
    q[7:] = np.clip(0.0, model.lower, model.upper)
    return q
