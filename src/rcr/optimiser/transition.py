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
Midpoints: halfway between consecutive samples (the joints interpolated, as the controller
will), the same balance, friction and limits apply, with their own hold forces, and the held balls
stay within midpoint_play of their holds (midpoint_poses).
Cost: w_tau sum ||tau||^2 (samples and midpoints) + w_smooth sum ||dq_{k+1} - dq_k||^2.

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
from dataclasses import dataclass, field, replace

import casadi as ca
import numpy as np

from rcr.optimiser import robot_model as rm
from rcr.optimiser.collision import (
    HOLD_MESH_OFFSET,
    SOCKET_RADIUS,
    Check,
    Clearance,
    check_trajectory,
)
from rcr.optimiser.robot_model import GRAVITY, NJ, NQ, NV, RobotModel, load_robot_model
from rcr.robot import hooks
from rcr.types import LIMBS, MICROBAN_JOINTS, Limb, Stance, WallModel

# [N·m] the current limit times BAM's XL330 torque constant: 1.75 A x 0.346 N·m/A, where the
# organisers' sim clips each motor [Microban constants.py BAM_MAX_CURRENT, microban_sim]
SERVO_TORQUE_LIMIT = 0.605
GUESS_STANDOFF = -0.10  # [m] trunk y in the first static pose's guess, in front of the wall [Ours]
FACING_WALL = (np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4))  # (w,x,y,z): trunk front along +y
# The hold's socket opens upwards: a ball leaves it only near vertical (~10 deg, the organisers'
# sim, wall.py). The first solve keeps each ball's neck within NECK_CONE of up, so it starts with
# the balls lowered in from above; the exact clearances then decide [Ours].
NECK_CONE = np.radians(45.0)
NECK_PROBE = 0.01  # [m] how far back from the ball along its neck (hooks.Hook.neck) it is measured
# Likewise the moving ball starts its path above its sockets: on the sample after it lets go and
# the one before it arrives, at least LIFT above its resting point, inside the opening cone [Ours].
LIFT = 0.035  # [m]: the hold's top is 18.2 mm above the resting ball, + its 12.5 mm radius + 4 mm

# The reference posture's leg joints: from a hand-designed pose on the example's start stance
# (legs mirrored, knees bent 20 deg, spine 25 deg, hips 100 mm off the wall, clear of the holds;
# peak torque 0.15 N·m), rounded. Left leg; the right mirrors it, the roll and yaw joints with the
# opposite sign (checked on the model: the feet then mirror exactly) [Ours].
NOMINAL_LEFT_LEG = {"hip_yaw": 6.0, "hip_roll": 14.0, "hip_pitch": -63.0, "knee": 20.0,
                    "ankle_pitch": 26.0, "ankle_roll": 2.0}  # [deg]  # fmt: skip
MIRROR_SIGN = {"hip_yaw": -1, "hip_roll": -1, "hip_pitch": 1, "knee": 1, "ankle_pitch": 1,
               "ankle_roll": -1}  # fmt: skip
NOMINAL_LEGS = {
    f"{side}_{joint}": float(np.radians(angle * (1 if side == "left" else MIRROR_SIGN[joint])))
    for side in ("left", "right")
    for joint, angle in NOMINAL_LEFT_LEG.items()
}


class OptimisationError(RuntimeError):
    """IPOPT did not find a solution; the message names the problem, status and violation."""


@dataclass
class TransitionConfig:
    n_samples: int = 12  # N: samples k = 0..N per transition [Ours]
    release_sample: int = 4  # K: end of phase A, where the moving limb's force reaches 0 [Ours]
    tau_limit: float = 0.7 * SERVO_TORQUE_LIMIT  # [N·m] 70 % of the servo limit [Ours]
    # Friction pyramid per limb: contact normal (the direction the hold pushes the ball) and mu.
    # Up (+z) for all four: the hold's socket opens upwards and holds a resting ball up, a foot's
    # as much as a hand's [hold.stl]. (Feet pushed only out from the wall, -y, made the robot
    # lean out and hang from its arms.) A simplification of the ball in its spherical socket,
    # which can push in more directions [Ours].
    normals: dict[str, tuple[float, float, float]] = field(
        default_factory=lambda: dict.fromkeys(LIMBS, (0.0, 0.0, 1.0))
    )
    # Below the organisers' sim, where ball on hold is max(ball 1.0, hold 0.5) = 1.0 [Ours]
    mu: dict[str, float] = field(default_factory=lambda: dict.fromkeys(LIMBS, 0.8))
    # The moving limb keeps this far out from the wall face (y <= -clearance) in the middle of
    # phase B, i.e. on samples K+2..N-2, so it can leave and enter its cups [Ours].
    # 0.06 m: the hold sticks out 43 mm [hold.stl], the ball is 12.5 mm in radius, + 4.5 mm, so
    # the ball can pass in front of a hold (the task's first value, 0.04 m, could not).
    clearance: float = 0.06  # [m]
    # Every point of Microban's collision shapes stays this far out of the wall face, in every
    # sample (y <= -wall_margin). Without it the trunk can sink into the wall [Ours].
    wall_margin: float = 0.002  # [m]
    # That constraint in a transition's NLP: off, as the exact check covers the wall too, and
    # its 7200 rows (288 points x 25 poses) were 70 % of the NLP. The static pose keeps it [Ours].
    wall_points: bool = False
    # Ball centre relative to Hold.position (the bolt midpoint on the wall face): the bottom of
    # the hold's spherical socket, 26 mm out and 14 mm below the bolts [Rules Figure 3], lowered
    # by the socket/ball clearance (12.7 - 12.5 mm). The same point as rcr.sim's grasp point.
    # Follows the model's ball (rcr.robot.hooks.BALL_RADIUS): (0, -0.026, -0.0142) today.
    contact_offset: tuple[float, float, float] = (
        HOLD_MESH_OFFSET[0],
        HOLD_MESH_OFFSET[1],
        HOLD_MESH_OFFSET[2] - (SOCKET_RADIUS - hooks.BALL_RADIUS),
    )  # [m]
    # How far the trunk may move from the solve's reference configuration (a static pose's
    # guess, or a transition's start): keeps the exponential map away from wrapping at pi.
    max_trunk_shift: float = 0.3  # [m] per axis [Ours]
    max_trunk_turn: float = 1.2  # [rad] per axis of the rotation vector [Ours]
    w_tau: float = 1.0  # cost weight on torques^2 [1/(N·m)^2]
    w_smooth: float = 1.0  # cost weight on (dq_{k+1} - dq_k)^2 [1/rad^2, 1/m^2]
    w_stay: float = 1e-3  # static pose only: weight on dq^2, keeps it near the guess [Ours]
    # A reference posture, preferred softly at every sample and midpoint (posture()), as FARO
    # regularises towards a nominal pose: (theta - theta_nom)^T W (theta - theta_nom) on the leg
    # joints (all six per leg, mirrored), plus the spine leaning spine_target from vertical.
    # Without it the least-torque pose crouches (knees at 135 deg on the example's start) and the
    # legs twist. The arms (the hand hooks are not mirror images) and the trunk's position are
    # left to the holds [Ours].
    nominal: dict[str, float] = field(default_factory=lambda: dict(NOMINAL_LEGS))  # [rad]
    spine_target: float = float(np.radians(25.0))  # [rad]
    w_nominal: float = 0.3  # weight on (theta - theta_nom)^2 per leg joint [1/rad^2]
    w_spine: float = 2.0  # weight on (spine_z - cos(spine_target))^2
    # Clear of the holds and the wall throughout the motion (collision.py), by sequential convex
    # programming on the exact clearances: see _solve_clear.
    avoid_collisions: bool = True
    collision_clearance: Clearance = field(default_factory=Clearance)
    collision_slots: int = 40  # linearised clearances per interval between samples [Ours]
    w_collision: float = 1e3  # [1/m] penalty on violated clearance (an exact penalty) [Ours]
    trust_radius: float = 0.05  # [rad, m] first trust region per component of dq [Ours]
    # If a transition cannot be made clear, SCP is run once more from this smaller first trust
    # region: it can stall a fraction of a millimetre short where a ball leaves its socket [Ours].
    retry_trust_radius: float = 0.02  # [rad, m]
    scp_iterations: int = 30  # most re-solves for the clearances [Ours]
    scp_polish: int = 0  # once clear, at most this many more re-solves to lower the cost [Ours]
    # Also hold the limits at the midpoint of every interval (the joints are interpolated there):
    # balls on their holds, balance, friction and torque limits, with their own hold forces.
    # Without it the plan met the limits only at the samples: +8 % torque and 4 mm of ball drift
    # between them on the test climb [Ours].
    midpoints: bool = True
    # How far a held ball may be from its hold at a midpoint: interpolated joints cannot keep a
    # closed chain exactly on four points, and the ball has 2.7 mm of play in its socket [Ours].
    midpoint_play: float = 0.001  # [m]
    # Every move starts at stance a's own pose and ends at stance b's (stance_pose: balanced,
    # clear, solved from a fresh guess): the move itself ends freely, then a settle with all four
    # balls held brings the robot to stance b's pose (solve_move). A move then depends on its
    # stances and the wall only, never on the moves before it, so its result is the same every
    # run and can be reused and cached [Ours, after FARO's per-mode poses]. On the example plan:
    # 34 of 35 moves, against 21 of 35 when the move itself had to end at stance b's pose.
    # Off: each move starts where the previous one ended, and depends on how the robot got there
    # (the example plan's full re-solve then failed at move 9 after a far hold changed).
    settle: bool = True
    settle_duration: float = 1.0  # [s] for the settle, after phases A and B [Ours]
    phase_a_duration: float = 1.0  # [s] for phase A, chosen after solving (quasi-static) [Ours]
    phase_b_duration: float = 2.0  # [s] for phase B [Ours]
    ipopt: dict = field(
        # mu_init 1e-3: each SCP round starts next to its answer (warm start), so a small first
        # barrier saves iterations. max_iter 150: a round that needs more is a poor step; it is
        # rejected (or kept if feasible) and SCP shrinks its trust region, which was faster
        # than letting it run to 1000 on the example's hand moves [Ours].
        default_factory=lambda: {
            "print_level": 0,
            "max_iter": 150,
            "tol": 1e-6,
            "mu_init": 1e-3,
            "sb": "yes",
        }  # fmt: skip
    )


@dataclass
class TransitionResult:
    # S samples: N + 1, or 2N + 1 with its settle (solve_move); times from sample_time
    q: np.ndarray  # S x NQ configurations
    dq: np.ndarray  # NV x S the tangent vectors at the start: q[k] = integrate(q[0], dq[:, k])
    tau: np.ndarray  # S x 19 motor torques [N·m]
    forces: dict[str, np.ndarray]  # limb -> S x 3 hold forces [N]; 0 where not in contact
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
    if cfg.wall_points:
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
    mid_forces: dict | None = None  # limb -> 3 x N hold forces at the interval midpoints
    # static pose only: parameter 4, 1 if that limb's hold carries load, 0 if its force is zero
    loaded: ca.MX | None = None
    free_neck: tuple = ()  # limbs whose hook is not in a socket: no neck bound for them
    end_dq: ca.MX | None = None  # parameter NV: a settle's last sample's dq
    end_q: np.ndarray | None = None  # the pose end_dq leads to (NQ), for the neck bounds


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


def posture(q, cfg: TransitionConfig):
    """Soft preference for the reference posture: the leg joints near cfg.nominal, and the spine
    leaning spine_target from vertical. spine_z, the vertical component of the trunk's own z axis,
    is 1 - 2 (x^2 + y^2) for its quaternion (w, x, y, z)."""
    joint = {name: q[7 + i] for i, name in enumerate(MICROBAN_JOINTS)}
    legs = sum((joint[name] - angle) ** 2 for name, angle in cfg.nominal.items())
    spine_z = 1 - 2 * (q[4] ** 2 + q[5] ** 2)
    return cfg.w_nominal * legs + cfg.w_spine * (spine_z - np.cos(cfg.spine_target)) ** 2


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
    """The hook's neck rises at least neck_min[limb] (see _neck_bounds): lowered in from above."""
    opti.subject_to(neck_rise(model, q_ref, dq, limb) >= neck_min[LIMBS.index(limb)])


def _neck_bounds(
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
        neck_min, loaded = opti.parameter(len(LIMBS)), opti.parameter(len(LIMBS))
        # one pose: the wall points are cheap here, and without them the solve can fail
        q, tau = balanced_pose(opti, model, dq[:, 0], q_ref, forces, replace(cfg, wall_points=True))
        opti.subject_to(model.limb_positions(q) == holds)
        for i, limb in enumerate(LIMBS):
            friction_pyramid(opti, forces[limb], cfg.normals[limb], cfg.mu[limb])
            neck_from_above(opti, model, q_ref, dq[:, 0], limb, neck_min)
            # unloaded (0): no push, so no force at all (friction pyramid); loaded (1): a bound
            # that never binds. One plain row: `(1 - loaded) f == 0` gave 0 == 0 rows that
            # changed IPOPT's path even with every limb loaded.
            push = ca.dot(forces[limb], np.array(cfg.normals[limb], dtype=float))
            opti.subject_to(push <= loaded[i] * MAX_HOLD_FORCE)
        objective = cfg.w_tau * ca.sumsqr(tau) + cfg.w_stay * ca.sumsqr(dq) + posture(q, cfg)
        slots, penalty, centre, radius = clearance_slots(opti, dq, cfg)
        opti.minimize(objective + penalty)
        _set_solver(opti, cfg, cold=True)
        return _Problem(
            opti, dq, forces, [q], [tau], objective, neck_min, q_ref, holds, slots, centre, radius,
            loaded=loaded,
        )  # fmt: skip

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
        mid_forces, mid_taus, mid_qs = midpoint_poses(opti, model, dq, q0, holds, moving, cfg)
        smooth = sum(ca.sumsqr(dq[:, k + 1] - dq[:, k]) for k in range(n))
        torque = sum(ca.sumsqr(t) for t in taus + mid_taus)
        stance = sum(posture(q, cfg) for q in qs + mid_qs)
        objective = cfg.w_tau * torque + cfg.w_smooth * smooth + stance
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
            mid_forces,
        )

    return _cached(("transition", moving, repr(cfg)), build)


# [rad, m] how far a settle's last sample may be from its end pose, per component of dq: a bound,
# not an equality, as the pose also meets the balls' equalities (to IPOPT's tolerance) [Ours]
END_POSE_BAND = 1e-4
_stance_poses: dict[tuple, np.ndarray] = {}


def stance_pose(wall: WallModel, stance: Stance, cfg: TransitionConfig, index: int = 0):
    """The stance's own pose (NQ): balanced, clear, solved from a fresh guess (initial_guess),
    so it depends on the stance and the wall only. The plan starts at stance 0's; with
    cfg.settle every move starts at its stance a's and ends at its stance b's. Cached per wall,
    stance and settings."""
    cfg = replace(cfg, settle=False)  # the pose itself does not depend on it
    key = (
        tuple((h.id, tuple(h.position)) for h in wall.holds),
        tuple(stance.contacts[limb] for limb in LIMBS),
        repr(cfg),
    )
    if key not in _stance_poses:
        guess = initial_guess(wall, stance, cfg)
        _stance_poses[key] = static_pose(wall, stance, guess, cfg, index).q[0]
    return _stance_poses[key].copy()


def settle_problem(cfg: TransitionConfig) -> _Problem:
    """A settle: all four balls stay on their holds while the robot moves from its start to a
    given end pose (end_dq), balanced at every sample and midpoint [Ours, after FARO]. Used to
    bring a move's free end to stance b's own pose (stance_pose)."""
    n = cfg.n_samples

    def build() -> _Problem:
        model, opti = load_robot_model(), ca.Opti()
        q0, holds, end_dq = opti.parameter(NQ), opti.parameter(3, len(LIMBS)), opti.parameter(NV)
        dq = opti.variable(NV, n + 1)
        forces = {limb: opti.variable(3, n + 1) for limb in LIMBS}
        opti.subject_to(dq[:, 0] == 0)
        opti.subject_to(opti.bounded(end_dq - END_POSE_BAND, dq[:, n], end_dq + END_POSE_BAND))
        neck_min = opti.parameter(len(LIMBS))
        qs, taus = [], []
        for k in range(n + 1):
            sample = {limb: forces[limb][:, k] for limb in LIMBS}
            q, tau = balanced_pose(opti, model, dq[:, k], q0, sample, cfg)
            opti.subject_to(model.limb_positions(q) == holds)
            for limb in LIMBS:
                neck_from_above(opti, model, q0, dq[:, k], limb, neck_min)
                friction_pyramid(opti, sample[limb], cfg.normals[limb], cfg.mu[limb])
            qs.append(q)
            taus.append(tau)
        mid_forces, mid_taus, mid_qs = midpoint_poses(opti, model, dq, q0, holds, None, cfg)
        smooth = sum(ca.sumsqr(dq[:, k + 1] - dq[:, k]) for k in range(n))
        torque = sum(ca.sumsqr(t) for t in taus + mid_taus)
        stance = sum(posture(q, cfg) for q in qs + mid_qs)
        objective = cfg.w_tau * torque + cfg.w_smooth * smooth + stance
        slots, penalty, centre, radius = clearance_slots(opti, dq, cfg)
        opti.minimize(objective + penalty)
        _set_solver(opti, cfg)
        return _Problem(
            opti, dq, forces, qs, taus, objective, neck_min, q0, holds, slots, centre, radius,
            mid_forces=mid_forces, end_dq=end_dq,
        )  # fmt: skip

    return _cached(("settle", repr(cfg)), build)


def solve_settle(
    wall: WallModel,
    stance: Stance,
    q0: np.ndarray,
    q_end: np.ndarray,
    cfg: TransitionConfig,
    index: int = 0,
) -> TransitionResult:
    """Move from q0 to q_end with all four balls on the holds of `stance` (settle_problem),
    clear of the holds, the wall and itself. OptimisationError if it cannot."""
    p = settle_problem(cfg)
    n = cfg.n_samples
    p.opti.set_value(p.q_ref, q0)
    p.opti.set_value(p.holds, contact_points(wall, stance, cfg))
    end = rm.difference(q0, q_end)
    p.opti.set_value(p.end_dq, end)
    p.end_q = q_end
    _fresh(p)
    for k in range(n + 1):
        p.opti.set_initial(p.dq[:, k], end * k / n)
        for limb in LIMBS:
            p.opti.set_initial(p.forces[limb][:, k], weight_share(len(LIMBS)))
    if p.mid_forces is not None:
        for limb in LIMBS:
            p.opti.set_initial(p.mid_forces[limb], np.tile(weight_share(len(LIMBS)), (n, 1)).T)
    own = {limb: {stance.contacts[limb]} for limb in LIMBS}
    check = _checker(wall, q0, [dict(stance.contacts)] * (n + 1), own, cfg)
    s = _solve_clear(p, f"settle into stance {index}", cfg, check)
    return TransitionResult(
        q=s.q, dq=s.dq, tau=s.tau, forces=s.forces, moving="",
        solve_time=s.solve_time, iterations=s.iterations,
    )  # fmt: skip


def midpoint_poses(opti: ca.Opti, model: RobotModel, dq, q0, holds, moving: Limb | None, cfg):
    """The pose halfway between consecutive samples (dq interpolated, as the controller will),
    held to the same rules as a sample: balance, friction, torque and joint limits, each midpoint
    with its own hold forces; held balls within cfg.midpoint_play of their holds. A limb counts as
    held if it is in its old socket at both ends of the interval. Returns (forces, torques,
    poses); nothing if cfg.midpoints is off."""
    if not cfg.midpoints:
        return None, [], []
    n, k_release = cfg.n_samples, cfg.release_sample
    forces = {limb: opti.variable(3, n) for limb in LIMBS}
    taus, qs = [], []
    for k in range(n):
        at = {limb: forces[limb][:, k] for limb in LIMBS}
        q, tau = balanced_pose(opti, model, (dq[:, k] + dq[:, k + 1]) / 2, q0, at, cfg)
        balls = model.limb_positions(q)
        for i, limb in enumerate(LIMBS):
            if limb != moving or k + 1 <= k_release:
                # relative to the play: in m^2, IPOPT's tolerance alone allowed 3 mm, not 1 mm
                drift = (balls[:, i] - holds[:, i]) / cfg.midpoint_play
                opti.subject_to(ca.sumsqr(drift) <= 1)
                friction_pyramid(opti, at[limb], cfg.normals[limb], cfg.mu[limb])
            else:
                opti.subject_to(at[limb] == 0)  # in the air
        taus.append(tau)
        qs.append(q)
    return forces, taus, qs


def _set_solver(opti: ca.Opti, cfg: TransitionConfig, cold: bool = False) -> None:
    """cold: the problem starts from a rough guess (the static pose), not next to its answer, so
    IPOPT's own barrier start and iteration budget: with cfg's small mu_init it found a start pose
    with a joint 3.8 rad from the usual one, from which the example's first move cannot be made
    clear [Ours]."""
    options = cfg.ipopt
    if cold:
        options = {k: v for k, v in options.items() if k != "mu_init"} | {"max_iter": 1000}
    opti.solver("ipopt", {"print_time": False, "expand": True}, options)


# --- Solving ------------------------------------------------------------------------------------


def solve_opti(opti: ca.Opti, what: str, patience: int = 0) -> tuple[ca.OptiSol, float]:
    """Run IPOPT: (solution, wall time [s]).

    If IPOPT runs out of iterations, it continues from where it stopped up to `patience` more
    times (for a solve that starts far from its answer, e.g. a transition's first). If it still
    runs out at a point that satisfies every constraint (to FEASIBLE_TOL), that point is
    returned: it is a valid motion, only not certified optimal, and the collision loop's own
    acceptance test decides what is kept. Otherwise OptimisationError with the status and the
    largest constraint violation.
    """
    start = time.perf_counter()
    for attempt in range(patience + 1):
        try:
            return opti.solve(), time.perf_counter() - start
        except RuntimeError:
            status = opti.debug.stats().get("return_status", "unknown")
            g = np.atleast_1d(opti.debug.value(opti.g))
            lbg = np.atleast_1d(opti.debug.value(opti.lbg))
            ubg = np.atleast_1d(opti.debug.value(opti.ubg))
            violation = float(np.max(np.concatenate([lbg - g, g - ubg, [0.0]])))
            if status != "Maximum_Iterations_Exceeded" or attempt == patience:
                break
            x, lam_g = opti.debug.value(opti.x), opti.debug.value(opti.lam_g)  # read both first:
            opti.set_initial(opti.x, x)  # set_initial discards the solve
            opti.set_initial(opti.lam_g, lam_g)  # carry on from where it stopped
    if status == "Maximum_Iterations_Exceeded" and violation <= FEASIBLE_TOL:
        return opti.debug, time.perf_counter() - start
    raise OptimisationError(f"{what}: IPOPT {status}, largest constraint violation {violation:.2e}")


# A point this close to every constraint counts as feasible. Not looser: some constraints are
# squared distances (m^2), and 1e-4 let a held ball drift 1.13 mm against its 1 mm [Ours].
MAX_HOLD_FORCE = 100.0  # [N] ~10x the robot's weight: a loaded hold's bound never binds
FEASIBLE_TOL = 1e-5  # [m, N, N·m, m^2]
# The first solve of the collision loop starts far from its answer: it may continue this many
# times past max_iter (150 each, ~1000 in all); the rounds after it start next to theirs [Ours].
FIRST_SOLVE_PATIENCE = 6


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


def _solve_clear(
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
    model, q_ref = load_robot_model(), np.array(p.opti.value(p.q_ref)).ravel()
    fixed_start = len(p.q) > 1  # a transition's first sample is its given start
    steer = cfg.avoid_collisions and seed is None
    if seed is not None and cfg.avoid_collisions:
        _linearise(p, check(seed), seed, cfg)
        p.opti.set_value(p.centre, seed)
        p.opti.set_value(p.radius, cfg.trust_radius)
    else:
        _linearise(p, [], np.zeros((NV, len(p.q))), cfg)  # no clearances yet
        p.opti.set_value(p.centre, np.zeros((NV, len(p.q))))
        p.opti.set_value(p.radius, OPEN_TRUST_REGION)
    p.opti.set_value(
        p.neck_min, _neck_bounds(model, q_ref, steer, fixed_start, p.free_neck, p.end_q)
    )
    if p.approach is not None:
        p.opti.set_value(p.approach, 1.0 if steer else 0.0)
    best = _read(p, *solve_opti(p.opti, what, FIRST_SOLVE_PATIENCE))
    if not cfg.avoid_collisions:
        return best
    p.opti.set_value(
        p.neck_min, _neck_bounds(model, q_ref, False, fixed_start, p.free_neck, p.end_q)
    )
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
    if strict and violation > CLEAR_TOL:
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


def static_pose(
    wall,
    stance,
    q_ref,
    cfg: TransitionConfig,
    index: int = 0,
    strict: bool = True,
    unloaded: Limb | None = None,
    away: tuple[float, float, float] | None = None,
    what: str = "",
) -> _Solution:
    """The full solution of solve_static_pose (one sample): q, tau, forces, dq from q_ref.
    strict=False returns a balanced pose even if it is not clear (for a warm start).
    unloaded: that limb's ball carries nothing (the other three carry the robot); with `away`
    it is also moved by that offset [m] out of its socket. These are the poses at the edges of
    a move (edge_checks.py)."""
    check_holds(wall, stance, index)
    p = static_pose_problem(cfg)
    holds = contact_points(wall, stance, cfg)
    loaded = np.ones(len(LIMBS))
    contacts, own = dict(stance.contacts), {limb: {stance.contacts[limb]} for limb in LIMBS}
    p.free_neck = ()
    if unloaded is not None:
        loaded[LIMBS.index(unloaded)] = 0.0
        if away is not None:
            holds[:, LIMBS.index(unloaded)] += np.asarray(away, dtype=float)
            contacts[unloaded] = None  # in the air: its own hold may be touched, not entered
            p.free_neck = (unloaded,)
    p.opti.set_value(p.q_ref, q_ref)
    p.opti.set_value(p.holds, holds)
    p.opti.set_value(p.loaded, loaded)
    _fresh(p)
    p.opti.set_initial(p.dq, np.zeros((NV, 1)))
    for limb, carries in zip(LIMBS, loaded, strict=True):
        share = weight_share(int(loaded.sum())) if carries else np.zeros(3)
        p.opti.set_initial(p.forces[limb], share)
    check = _checker(wall, q_ref, [contacts], own, cfg)
    return _solve_clear(p, what or f"static pose for stance {index}", cfg, check, strict)


def solve_transition(
    wall: WallModel,
    a: Stance,
    b: Stance,
    q0: np.ndarray,
    cfg: TransitionConfig | None = None,
    index: int = 0,
    seed: np.ndarray | None = None,
    rescue: bool = True,
) -> TransitionResult:
    """Move the one limb that changes from its hold in `a` to its hold in `b`, starting at q0.

    q0 must have every ball on its hold of `a` (e.g. from solve_static_pose or the previous
    transition). Only the moving limb's final point is fixed, not the final configuration
    (solve_move adds the settle into stance b's own pose).
    The motion stays clear of the holds and the wall throughout (cfg.avoid_collisions).
    seed: a solved transition's dq (NV x samples) to start from, e.g. the same limb's last move
    on holds laid out alike (see same_layout); if that fails, the usual start is tried.
    rescue: if every start fails, solve the move coarsely (coarse()) and start from that path.
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
    check = _checker(wall, q0, transition_contacts(a, b, moving, cfg), _own_holds(a, b), cfg)
    s = None
    if seed is not None and seed.shape == (NV, cfg.n_samples + 1):
        _warm_start(p, wall, b, q0, moving, cfg, index, seed)
        try:
            s = _solve_clear(p, what, cfg, check, seed=seed)
        except OptimisationError:
            pass  # the reused motion does not fit here: start as usual
    if s is None:
        try:
            _warm_start(p, wall, b, q0, moving, cfg, index)
        except OptimisationError as e:
            raise OptimisationError(
                f"{what}: the robot cannot hold stance {index + 1} ({e})"
            ) from e
        try:
            s = _solve_clear(p, what, cfg, check)
        except OptimisationError as e:
            try:
                s = _solve_clear(p, what, replace(cfg, trust_radius=cfg.retry_trust_radius), check)
            except OptimisationError:
                if not (rescue and _halves(cfg)):
                    raise
                s = _rescue(wall, a, b, q0, cfg, index, p, check, what, e)
    return TransitionResult(
        q=s.q, dq=s.dq, tau=s.tau, forces=s.forces, moving=moving,
        solve_time=s.solve_time, iterations=s.iterations,
    )  # fmt: skip


def solve_move(
    wall: WallModel, a: Stance, b: Stance, cfg: TransitionConfig, index: int = 0
) -> TransitionResult:
    """One move of a plan with cfg.settle: from stance a's own pose (stance_pose), the move
    (solve_transition, free end, no reused seed), then the settle into stance b's own pose
    (solve_settle). Depends on the stances and the wall only. Returns both as one result: the
    move's samples, then the settle's; the last sample is stance b's pose exactly (the settle
    ends within END_POSE_BAND of it), so the next move starts where this one ends."""
    moving = moving_limb(a, b, index)
    q0 = stance_pose(wall, a, cfg, index)
    try:
        q_end = stance_pose(wall, b, cfg, index + 1)
    except OptimisationError as e:
        what = f"transition {index} ({moving} to {b.contacts[moving]})"
        raise OptimisationError(f"{what}: the robot cannot hold stance {index + 1} ({e})") from e
    move = solve_transition(wall, a, b, q0, cfg, index)
    settle = solve_settle(wall, b, move.q[-1], q_end, cfg, index + 1)
    q = np.vstack([move.q, settle.q[1:]])
    q[-1] = q_end
    return TransitionResult(
        q=q,
        dq=np.column_stack([rm.difference(q0, qk) for qk in q]),
        tau=np.vstack([move.tau, settle.tau[1:]]),
        forces={limb: np.vstack([move.forces[limb], settle.forces[limb][1:]]) for limb in LIMBS},
        moving=move.moving,
        solve_time=move.solve_time + settle.solve_time,
        iterations=move.iterations + settle.iterations,
    )


def layout(wall: WallModel, a: Stance, b: Stance, cfg: TransitionConfig) -> np.ndarray:
    """3x8: the contact points of a, then b, relative to the moving limb's start point. Two
    transitions with the same layout are the same move, shifted: one's dq can seed the other."""
    start = contact_point(wall, a.contacts[moving_limb(a, b, 0)], cfg)
    both = np.hstack([contact_points(wall, a, cfg), contact_points(wall, b, cfg)])
    return both - start[:, None]


def same_layout(one: np.ndarray, other: np.ndarray) -> bool:
    return bool(np.abs(one - other).max() <= SEED_MATCH)


SEED_MATCH = 0.02  # [m] layouts this close make a solved move a good seed [Ours]


def shifted_seed(q0: np.ndarray, previous_q: np.ndarray, shift: np.ndarray, cfg: TransitionConfig):
    """A seed (NV x samples) from a solved motion of the same move elsewhere: its configurations
    moved by `shift` [m] (the trunk; joints and trunk turn as they were), in the tangent space at
    q0. The gap between q0 and the moved start fades out over phase A, so the seed starts at q0
    and follows the solved motion from the release on. Reusing dq instead (relative to the other
    start) put the arm 15 mm into a hold: the starts differ by up to ~20 deg in the joints."""
    moved = np.array(previous_q, dtype=float)
    moved[:, 0:3] += shift
    dq = np.column_stack([rm.difference(q0, q) for q in moved])
    gap = dq[:, 0].copy()
    for k in range(dq.shape[1]):
        dq[:, k] -= max(0.0, 1.0 - k / cfg.release_sample) * gap
    return dq


def coarse(cfg: TransitionConfig) -> TransitionConfig:
    """The same move with half the samples and no midpoints: a few seconds, for a first path
    (the last start solve_transition tries) and for feasibility's path check."""
    return replace(
        cfg, n_samples=cfg.n_samples // 2, release_sample=cfg.release_sample // 2, midpoints=False
    )


def _halves(cfg: TransitionConfig) -> bool:
    """Can coarse(cfg) be solved and its path put back on cfg's samples?"""
    n, k = cfg.n_samples, cfg.release_sample
    return n % 2 == 0 and k % 2 == 0 and 0 < k // 2 < n // 2 - 2


def upsample(dq_coarse: np.ndarray) -> np.ndarray:
    """A coarse path on the full samples: coarse sample c is at full sample 2c (both phases
    halve), the ones between halfway, in the tangent space at the start."""
    n = 2 * (dq_coarse.shape[1] - 1)
    mid = (dq_coarse[:, :-1] + dq_coarse[:, 1:]) / 2  # between coarse samples c and c + 1
    cols = [dq_coarse[:, k // 2] if k % 2 == 0 else mid[:, k // 2] for k in range(n + 1)]
    return np.column_stack(cols)


def _rescue(wall, a, b, q0, cfg, index, p, check, what, error) -> _Solution:
    """Every start failed: find a coarse path (fewer, freer samples) and start from it. On the
    example's hard wall this solved a hand move (R3 to R5) that failed from every other start."""
    try:
        rough = solve_transition(wall, a, b, q0, coarse(cfg), index, rescue=False)
    except OptimisationError:
        raise error from None  # no coarse path either: report the full solve's failure
    seed = upsample(rough.dq)
    _warm_start(p, wall, b, q0, moving_limb(a, b, index), cfg, index, seed)
    return _solve_clear(p, what, cfg, check, seed=seed)


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


def _fresh(p: _Problem) -> None:
    """Forget the previous solve: every variable and multiplier starts at zero unless set after.
    The problems are reused, and without this a move solved or failed depending on what was
    solved before it."""
    p.opti.set_initial(p.opti.x, np.zeros(p.opti.nx))
    p.opti.set_initial(p.opti.lam_g, np.zeros(p.opti.ng))


def _warm_start(p: _Problem, wall, b, q0, moving, cfg, index, seed=None) -> None:
    """dq: `seed` if given (a solved motion reused, see solve_transition), else still in phase
    A, then a straight line (in the tangent space at q0) to a static pose of stance b, solved
    from a fresh guess so that it is in the reference posture rather than wherever q0 leads
    (balanced, but not necessarily clear: the transition's own clearances decide);
    forces: the weight shared over the limbs carrying it. With a seed, the rest of the problem's
    state (multipliers, slacks) is kept: the problem is per moving limb, so it comes from the
    solve the seed was taken from; without one, everything starts afresh (_fresh)."""
    if seed is None:
        _fresh(p)
    n, k_release = cfg.n_samples, cfg.release_sample
    if seed is not None:
        p.opti.set_initial(p.dq, seed)
    else:
        start = initial_guess(wall, b, cfg)
        q_end = static_pose(wall, b, start, cfg, index + 1, strict=False).q[0]
        dq_end = rm.difference(q0, q_end)
        for k in range(n + 1):
            p.opti.set_initial(p.dq[:, k], max(0.0, (k - k_release) / (n - k_release)) * dq_end)
    for k in range(n + 1):
        carrying = [limb for limb in LIMBS if limb != moving or k < k_release]
        for limb in LIMBS:
            share = weight_share(len(carrying)) if limb in carrying else np.zeros(3)
            p.opti.set_initial(p.forces[limb][:, k], share)
    if p.mid_forces is not None:
        for k in range(n):
            held = [limb for limb in LIMBS if limb != moving or k + 1 <= k_release]
            for limb in LIMBS:
                share = weight_share(len(held)) if limb in held else np.zeros(3)
                p.opti.set_initial(p.mid_forces[limb][:, k], share)


def weight_share(n_limbs: int) -> np.ndarray:
    """[N] warm start for one hold's force: an equal share of the robot's weight, upwards."""
    return np.array([0.0, 0.0, load_robot_model().mass * GRAVITY / n_limbs])


def initial_guess(wall: WallModel, stance: Stance, cfg: TransitionConfig | None = None):
    """A rough configuration to start solve_static_pose from: the trunk in front of the
    stance's holds, upright and facing the wall, the legs in the reference posture and the
    other motors at zero [Ours]."""
    cfg = cfg or TransitionConfig()
    points = contact_points(wall, stance, cfg)
    model = load_robot_model()
    q = np.zeros(7 + NJ)
    q[0:3] = [points[0].mean(), GUESS_STANDOFF, points[2].mean()]
    q[3:7] = FACING_WALL
    q[7:] = [cfg.nominal.get(name, 0.0) for name in MICROBAN_JOINTS]  # the legs as the reference
    q[7:] = np.clip(q[7:], model.lower, model.upper)
    return q
