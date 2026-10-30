"""The solver: quasi-static transitions, one small NLP (CasADi Opti + IPOPT) per change of hold.

Quasi-static: no inertia, every sample k is a statically balanced pose. Per sample:
    dq_k   the configuration, as a tangent vector around the transition's start q0:
           q_k = integrate(q0, dq_k), so the trunk quaternion stays a unit quaternion
    f_k^c  the force [N] that hold c exerts on the robot's ball, one per limb in contact
Torques are not variables. The 19 motor rows of the static balance give them,
    tau_k = g(q_k) - sum_c J_c(q_k)^T f_k^c                                   [N·m]
and the 6 trunk rows must balance:
    sum_c f_k^c + m g = 0   and   sum_c (x_c - com) x f_k^c = 0   (moments about the CoM).

Subject to, at every sample (balanced_pose and the problems below):
    the trunk balance above; joint limits; |tau| <= tau_limit; the trunk near its reference;
    each held ball on its hold, pushed by it within its friction pyramid; a ball in the air
    carries nothing.
Two phases, samples k = 0..N:
    A (k = 0..K): four contacts; the body shifts until the moving limb carries nothing at K.
    B (k = K..N): three contacts; the moving limb travels and reaches its new hold at N.
Midpoints: halfway between consecutive samples (the joints interpolated, as the controller
will), the same balance, friction and limits apply, with their own hold forces, and the held balls
stay within midpoint_play of their holds (midpoint_poses).
Cost: w_tau sum ||tau||^2 (samples and midpoints) + w_smooth sum ||dq_{k+1} - dq_k||^2 + posture.

A move of a plan (solve_move): stance a's own pose, the transition, then a settle (all four balls
held) into stance b's own pose.

Collisions are ignored by default for now: with TransitionConfig.avoid_collisions (or the
two-step approach_from_above) the problems take the extra constraints and the collision loop of
avoidance.py; the solver itself only calls it.

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

from rcr.optimiser import avoidance as av
from rcr.optimiser import robot_model as rm
from rcr.optimiser.collision import HOLD_MESH_OFFSET, SOCKET_RADIUS, Clearance
from rcr.optimiser.robot_model import GRAVITY, NJ, NQ, NV, RobotModel, load_robot_model
from rcr.robot import hooks
from rcr.types import LIMBS, MICROBAN_JOINTS, Limb, Stance, WallModel

# [N·m] the current limit times BAM's XL330 torque constant: 1.75 A x 0.346 N·m/A, where the
# organisers' sim clips each motor [Microban constants.py BAM_MAX_CURRENT, microban_sim]
SERVO_TORQUE_LIMIT = 0.605
GUESS_STANDOFF = -0.10  # [m] trunk y in the first static pose's guess, in front of the wall [Ours]
FACING_WALL = (np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4))  # (w,x,y,z): trunk front along +y

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
    # as much as a hand's [the sim's hold mesh]. (Feet pushed only out from the wall, -y, made the
    # robot lean out and hang from its arms.) A simplification of the ball in its spherical socket,
    # which can push in more directions [Ours].
    normals: dict[str, tuple[float, float, float]] = field(
        default_factory=lambda: dict.fromkeys(LIMBS, (0.0, 0.0, 1.0))
    )
    # Below the organisers' sim, where ball on hold is max(ball 1.0, hold 0.5) = 1.0 [Ours]
    mu: dict[str, float] = field(default_factory=lambda: dict.fromkeys(LIMBS, 0.8))
    # Ball centre relative to Hold.position (the bolt midpoint on the wall face): the bottom of
    # the hold's spherical socket, 26 mm out and 14 mm below the bolts [Rules Figure 3], lowered
    # by the socket/ball clearance (12.7 - 12.5 mm): where a ball rests in the organisers' sim.
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
    # Also hold the limits at the midpoint of every interval (the joints are interpolated there):
    # balls on their holds, balance, friction and torque limits, with their own hold forces.
    # Without it the plan met the limits only at the samples: +8 % torque and 4 mm of ball drift
    # between them on the test climb [Ours].
    midpoints: bool = True
    # How far a held ball may be from its hold at a midpoint: interpolated joints cannot keep a
    # closed chain exactly on four points, and the ball has 2.7 mm of play in its socket [Ours].
    midpoint_play: float = 0.001  # [m]
    # Every move starts at stance a's own pose and ends at stance b's (stance_pose: balanced,
    # solved from a fresh guess): the move itself ends freely, then a settle with all four
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
    # --- Collision avoidance (avoidance.py), off by default for now ------------------------------
    # Clear of the holds, the wall and itself throughout the motion (collision.py), by sequential
    # convex programming on the exact clearances (avoidance.solve_clear), with the moving ball out
    # from the wall in mid-air (clearance), the static pose out of the wall (wall_margin) and the
    # first solve steered. Off: the motion ignores every collision (runner: --collision) [Ours].
    avoid_collisions: bool = False
    # The moving limb keeps this far out from the wall face (y <= -clearance) in the middle of
    # phase B, i.e. on samples K+2..N-2, so it can leave and enter its cups [Ours].
    # 0.06 m: the hold sticks out 43 mm [the sim's hold mesh], the ball is 12.5 mm in radius,
    # + 4.5 mm, so the ball can pass in front of a hold (the task's first value, 0.04 m, could not).
    clearance: float = 0.06  # [m]
    # Every point of Microban's collision shapes stays this far out of the wall face (y <=
    # -wall_margin): in the static pose, with avoid_collisions. Without it the trunk can sink
    # into the wall [Ours].
    wall_margin: float = 0.002  # [m]
    # That constraint in a transition's NLP: off, as the exact check covers the wall too, and
    # its 7200 rows (288 points x 25 poses) were 70 % of the NLP [Ours].
    wall_points: bool = False
    collision_clearance: Clearance = field(default_factory=Clearance)
    collision_slots: int = 40  # linearised clearances per interval between samples [Ours]
    w_collision: float = 1e3  # [1/m] penalty on violated clearance (an exact penalty) [Ours]
    trust_radius: float = 0.05  # [rad, m] first trust region per component of dq [Ours]
    # If a transition cannot be made clear, SCP is run once more from this smaller first trust
    # region: it can stall a fraction of a millimetre short where a ball leaves its socket [Ours].
    retry_trust_radius: float = 0.02  # [rad, m]
    scp_iterations: int = 30  # most re-solves for the clearances [Ours]
    scp_polish: int = 0  # once clear, at most this many more re-solves to lower the cost [Ours]
    # The moving ball's way in, in every solve: out from the wall until N-3, LIFT above its new
    # socket inside NECK_CONE at N-2, SEAT_LIFT above it inside SEAT_CONE at N-1, so the last
    # interval is a short, straight drop (avoidance.py). The climbing model's ball fills its
    # socket (0.2 mm of play), and coming in along one long last interval it grazed the rim by
    # ~2.5 mm on N1's foot moves [Ours]. Off by default (runner: --approach). Needs n_samples >=
    # release_sample + 5.
    approach_from_above: bool = False


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


# --- Physics of one sample: the subject-to's ----------------------------------------------------


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


def balanced_pose(opti, model, dq, q_ref, forces, cfg):
    """Constrain one sample to a statically balanced, allowed pose. Returns (q, tau)."""
    trunk_bounds(opti, dq, cfg)
    q = model.integrate(q_ref, dq)
    wrench, tau = static_balance(model, q, forces)
    opti.subject_to(wrench == 0)
    joint_and_torque_limits(opti, model, q, tau, cfg)
    if cfg.wall_points:
        av.out_of_the_wall(opti, model, q, cfg)
    return q, tau


def posture(q, cfg: TransitionConfig):
    """Soft preference for the reference posture: the leg joints near cfg.nominal, and the spine
    leaning spine_target from vertical. spine_z, the vertical component of the trunk's own z axis,
    is 1 - 2 (x^2 + y^2) for its quaternion (w, x, y, z)."""
    joint = {name: q[7 + i] for i, name in enumerate(MICROBAN_JOINTS)}
    legs = sum((joint[name] - angle) ** 2 for name, angle in cfg.nominal.items())
    spine_z = 1 - 2 * (q[4] ** 2 + q[5] ** 2)
    return cfg.w_nominal * legs + cfg.w_spine * (spine_z - np.cos(cfg.spine_target)) ** 2


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
    q_ref: ca.MX  # parameter NQ: the configuration dq is measured from
    holds: ca.MX  # parameter 3x4: the contact points of the start stance, LIMBS order
    target: ca.MX | None = None  # parameter 3: the moving limb's new contact point
    mid_forces: dict | None = None  # limb -> 3 x N hold forces at the interval midpoints
    # static pose only: parameter 4, 1 if that limb's hold carries load, 0 if its force is zero
    loaded: ca.MX | None = None
    end_dq: ca.MX | None = None  # parameter NV: a settle's last sample's dq
    # collision avoidance only (avoidance.py): None / empty without it
    neck_min: ca.MX | None = None  # parameter 4: the least rise of each hook's neck (seed only)
    approach: ca.MX | None = None  # parameter, 1 or 0: the moving ball above its sockets
    slots: list = field(default_factory=list)  # per interval: (A, B, rhs, slack), linearised
    centre: ca.MX | None = None  # parameter NV x samples: the trust region's centre
    radius: ca.MX | None = None  # parameter: the trust region's half-width on each part of dq
    free_neck: tuple = ()  # limbs whose hook is not in a socket: no neck bound for them
    end_q: np.ndarray | None = None  # the pose end_dq leads to (NQ), for the neck bounds


_problems: dict[tuple, _Problem] = {}


def _built(key: tuple, build) -> _Problem:
    """The problem built for this key, built on first use (building costs more than solving)."""
    if key not in _problems:
        _problems[key] = build()
    return _problems[key]


def static_pose_problem(cfg: TransitionConfig) -> _Problem:
    """One balanced pose with all four balls on their holds; cost tau^2 + w_stay dq^2."""
    avoid = cfg.avoid_collisions

    def build() -> _Problem:
        model, opti = load_robot_model(), ca.Opti()
        q_ref, holds = opti.parameter(NQ), opti.parameter(3, len(LIMBS))
        dq = opti.variable(NV, 1)
        forces = {limb: opti.variable(3, 1) for limb in LIMBS}
        neck_min = opti.parameter(len(LIMBS)) if avoid else None
        loaded = opti.parameter(len(LIMBS))
        # with collision avoidance, out of the wall: one pose, the wall points are cheap here
        q, tau = balanced_pose(
            opti, model, dq[:, 0], q_ref, forces, replace(cfg, wall_points=avoid)
        )
        opti.subject_to(model.limb_positions(q) == holds)
        for i, limb in enumerate(LIMBS):
            friction_pyramid(opti, forces[limb], cfg.normals[limb], cfg.mu[limb])
            if avoid:
                av.neck_from_above(opti, model, q_ref, dq[:, 0], limb, neck_min)
            # unloaded (0): no push, so no force at all (friction pyramid); loaded (1): a bound
            # that never binds. One plain row: `(1 - loaded) f == 0` gave 0 == 0 rows that
            # changed IPOPT's path even with every limb loaded.
            push = ca.dot(forces[limb], np.array(cfg.normals[limb], dtype=float))
            opti.subject_to(push <= loaded[i] * MAX_HOLD_FORCE)
        objective = cfg.w_tau * ca.sumsqr(tau) + cfg.w_stay * ca.sumsqr(dq) + posture(q, cfg)
        p = _Problem(opti, dq, forces, [q], [tau], objective, q_ref, holds, loaded=loaded)
        _minimise(p, cfg, cold=True)
        p.neck_min = neck_min
        return p

    return _built(("static", repr(cfg)), build)


def transition_problem(moving: Limb, cfg: TransitionConfig) -> _Problem:
    """The two-phase transition of one moving limb; see the module docstring."""
    n, k_release = cfg.n_samples, cfg.release_sample
    if not 0 < k_release < n - 2:
        raise ValueError(f"need 0 < release_sample < n_samples - 2, got {k_release}, {n}")
    if cfg.approach_from_above and n < k_release + 5:
        raise ValueError(f"approach_from_above needs n_samples >= release_sample + 5, got {n}")
    avoid = cfg.avoid_collisions

    def build() -> _Problem:
        model, opti = load_robot_model(), ca.Opti()
        q0, holds, target = opti.parameter(NQ), opti.parameter(3, len(LIMBS)), opti.parameter(3)
        dq = opti.variable(NV, n + 1)
        forces = {limb: opti.variable(3, n + 1) for limb in LIMBS}
        opti.subject_to(dq[:, 0] == 0)
        neck_min = opti.parameter(len(LIMBS)) if avoid else None
        approach = opti.parameter() if avoid or cfg.approach_from_above else None
        m = LIMBS.index(moving)
        qs, taus = [], []
        for k in range(n + 1):
            sample = {limb: forces[limb][:, k] for limb in LIMBS}
            q, tau = balanced_pose(opti, model, dq[:, k], q0, sample, cfg)
            balls = model.limb_positions(q)
            for i, limb in enumerate(LIMBS):
                in_socket = limb != moving or k <= k_release or k == n
                if avoid and in_socket:
                    av.neck_from_above(opti, model, q0, dq[:, k], limb, neck_min)
                if limb != moving or k <= k_release:
                    opti.subject_to(balls[:, i] == holds[:, i])  # on its hold of the start stance
                if limb != moving or k < k_release:
                    friction_pyramid(opti, sample[limb], cfg.normals[limb], cfg.mu[limb])
                else:
                    opti.subject_to(sample[limb] == 0)  # f_K^moving = 0, and nothing after
            if avoid:
                av.mid_air(opti, balls, m, k, cfg)
            if k == n:
                opti.subject_to(balls[:, m] == target)  # the moving limb arrives
            if approach is not None:
                av.approach(opti, balls, m, holds, target, k, cfg, approach)
            qs.append(q)
            taus.append(tau)
        mid_forces, mid_taus, mid_qs = midpoint_poses(opti, model, dq, q0, holds, moving, cfg)
        smooth = sum(ca.sumsqr(dq[:, k + 1] - dq[:, k]) for k in range(n))
        torque = sum(ca.sumsqr(t) for t in taus + mid_taus)
        stance = sum(posture(q, cfg) for q in qs + mid_qs)
        objective = cfg.w_tau * torque + cfg.w_smooth * smooth + stance
        p = _Problem(opti, dq, forces, qs, taus, objective, q0, holds, target, mid_forces)
        _minimise(p, cfg)
        p.neck_min, p.approach = neck_min, approach
        return p

    return _built(("transition", moving, repr(cfg)), build)


def settle_problem(cfg: TransitionConfig) -> _Problem:
    """A settle: all four balls stay on their holds while the robot moves from its start to a
    given end pose (end_dq), balanced at every sample and midpoint [Ours, after FARO]. Used to
    bring a move's free end to stance b's own pose (stance_pose)."""
    n = cfg.n_samples
    avoid = cfg.avoid_collisions

    def build() -> _Problem:
        model, opti = load_robot_model(), ca.Opti()
        q0, holds, end_dq = opti.parameter(NQ), opti.parameter(3, len(LIMBS)), opti.parameter(NV)
        dq = opti.variable(NV, n + 1)
        forces = {limb: opti.variable(3, n + 1) for limb in LIMBS}
        opti.subject_to(dq[:, 0] == 0)
        opti.subject_to(opti.bounded(end_dq - END_POSE_BAND, dq[:, n], end_dq + END_POSE_BAND))
        neck_min = opti.parameter(len(LIMBS)) if avoid else None
        qs, taus = [], []
        for k in range(n + 1):
            sample = {limb: forces[limb][:, k] for limb in LIMBS}
            q, tau = balanced_pose(opti, model, dq[:, k], q0, sample, cfg)
            opti.subject_to(model.limb_positions(q) == holds)
            for limb in LIMBS:
                if avoid:
                    av.neck_from_above(opti, model, q0, dq[:, k], limb, neck_min)
                friction_pyramid(opti, sample[limb], cfg.normals[limb], cfg.mu[limb])
            qs.append(q)
            taus.append(tau)
        mid_forces, mid_taus, mid_qs = midpoint_poses(opti, model, dq, q0, holds, None, cfg)
        smooth = sum(ca.sumsqr(dq[:, k + 1] - dq[:, k]) for k in range(n))
        torque = sum(ca.sumsqr(t) for t in taus + mid_taus)
        stance = sum(posture(q, cfg) for q in qs + mid_qs)
        objective = cfg.w_tau * torque + cfg.w_smooth * smooth + stance
        p = _Problem(
            opti, dq, forces, qs, taus, objective, q0, holds, mid_forces=mid_forces, end_dq=end_dq
        )
        _minimise(p, cfg)
        p.neck_min = neck_min
        return p

    return _built(("settle", repr(cfg)), build)


# [rad, m] how far a settle's last sample may be from its end pose, per component of dq: a bound,
# not an equality, as the pose also meets the balls' equalities (to IPOPT's tolerance) [Ours]
END_POSE_BAND = 1e-4
MAX_HOLD_FORCE = 100.0  # [N] ~10x the robot's weight: a loaded hold's bound never binds


def _minimise(p: _Problem, cfg: TransitionConfig, cold: bool = False) -> None:
    """Set the problem's cost (plus, with collision avoidance, room for its linearised
    clearances: avoidance.clearance_slots) and IPOPT. cold: the problem starts from a rough guess
    (the static pose), not next to its answer, so IPOPT's own barrier start and iteration budget:
    with cfg's small mu_init it found a start pose with a joint 3.8 rad from the usual one, from
    which the example's first move cannot be made clear [Ours]."""
    penalty = 0
    if cfg.avoid_collisions:
        p.slots, penalty, p.centre, p.radius = av.clearance_slots(p.opti, p.dq, cfg)
    p.opti.minimize(p.objective + penalty)
    options = cfg.ipopt
    if cold:
        options = {k: v for k, v in options.items() if k != "mu_init"} | {"max_iter": 1000}
    p.opti.solver("ipopt", {"print_time": False, "expand": True}, options)


# --- Solving ------------------------------------------------------------------------------------


def solve_opti(opti: ca.Opti, what: str, patience: int = 0) -> tuple[ca.OptiSol, float]:
    """Run IPOPT: (solution, wall time [s]).

    If IPOPT runs out of iterations, it continues from where it stopped up to `patience` more
    times (for a solve that starts far from its answer, e.g. a transition's first). If it still
    runs out at a point that satisfies every constraint (to FEASIBLE_TOL), that point is
    returned: it is a valid motion, only not certified optimal (with collision avoidance, its
    loop's own acceptance test decides what is kept). Otherwise OptimisationError with the status
    and the largest constraint violation.
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
FEASIBLE_TOL = 1e-5  # [m, N, N·m, m^2]
# The first solve starts far from its answer: it may continue this many times past max_iter (150
# each, ~1000 in all); with collision avoidance, the rounds after it start next to theirs [Ours].
FIRST_SOLVE_PATIENCE = 6


@dataclass
class _Solution:
    """What one accepted solve gave, read out of CasADi."""

    dq: np.ndarray  # NV x samples
    q: np.ndarray  # samples x NQ
    tau: np.ndarray  # samples x 19
    forces: dict[str, np.ndarray]  # limb -> samples x 3
    objective: float
    slack: float  # [m] summed slack of the linearised clearances (0 without them)
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


def _solve(p: _Problem, what: str, cfg: TransitionConfig, check, strict=True, seed=None):
    """IPOPT on the problem as it is set; with cfg.avoid_collisions, the collision loop
    (avoidance.solve_clear: check(dq) measures the clearances, strict raises if they cannot be
    met, seed is a reused motion to start from)."""
    if cfg.avoid_collisions:
        return av.solve_clear(p, what, cfg, check, strict, seed)
    if p.approach is not None:  # approach_from_above: the ball's way in, in every solve
        p.opti.set_value(p.approach, 1.0)
    return _read(p, *solve_opti(p.opti, what, FIRST_SOLVE_PATIENCE))


def solve_static_pose(
    wall: WallModel,
    stance: Stance,
    q_guess: np.ndarray,
    cfg: TransitionConfig | None = None,
    index: int = 0,
) -> np.ndarray:
    """A statically balanced configuration (NQ) with every ball on its hold (with
    cfg.avoid_collisions, clear of the holds and the wall).

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
    check = av.checker(wall, q_ref, [contacts], own, cfg)
    return _solve(p, what or f"static pose for stance {index}", cfg, check, strict)


_stance_poses: dict[tuple, np.ndarray] = {}


def stance_pose(wall: WallModel, stance: Stance, cfg: TransitionConfig, index: int = 0):
    """The stance's own pose (NQ): balanced, solved from a fresh guess (initial_guess), so it
    depends on the stance and the wall only. The plan starts at stance 0's; with cfg.settle
    every move starts at its stance a's and ends at its stance b's. Kept per wall, stance and
    settings for the run."""
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
    With cfg.avoid_collisions the motion stays clear of the holds and the wall throughout.
    seed: a solved transition's dq (NV x samples) to start from, e.g. the same limb's last move
    on holds laid out alike (move_cache.same_layout); if that fails, the usual start is tried.
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
    contacts = av.transition_contacts(a, b, moving, cfg)
    check = av.checker(wall, q0, contacts, av.own_holds(a, b), cfg)
    s = None
    if seed is not None and seed.shape == (NV, cfg.n_samples + 1):
        _warm_start(p, wall, b, q0, moving, cfg, index, seed)
        try:
            s = _solve(p, what, cfg, check, seed=seed)
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
            s = _solve(p, what, cfg, check)
        except OptimisationError as e:
            try:
                if not cfg.avoid_collisions:
                    raise  # the retry below is the collision loop's, from a smaller trust region
                s = _solve(p, what, replace(cfg, trust_radius=cfg.retry_trust_radius), check)
            except OptimisationError:
                if not (rescue and _halves(cfg)):
                    raise
                s = _rescue(wall, a, b, q0, cfg, index, p, check, what, e)
    return TransitionResult(
        q=s.q, dq=s.dq, tau=s.tau, forces=s.forces, moving=moving,
        solve_time=s.solve_time, iterations=s.iterations,
    )  # fmt: skip


def solve_settle(
    wall: WallModel,
    stance: Stance,
    q0: np.ndarray,
    q_end: np.ndarray,
    cfg: TransitionConfig,
    index: int = 0,
) -> TransitionResult:
    """Move from q0 to q_end with all four balls on the holds of `stance` (settle_problem)
    (with cfg.avoid_collisions, clear of the holds, the wall and itself). OptimisationError if it
    cannot."""
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
    check = av.checker(wall, q0, [dict(stance.contacts)] * (n + 1), own, cfg)
    s = _solve(p, f"settle into stance {index}", cfg, check)
    return TransitionResult(
        q=s.q, dq=s.dq, tau=s.tau, forces=s.forces, moving="",
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


# --- When every start fails: a coarse path first ------------------------------------------------


def coarse(cfg: TransitionConfig) -> TransitionConfig:
    """The same move with half the samples and no midpoints: a few seconds, for a first path
    (the last start solve_transition tries) and for feasibility's path check."""
    n, k_release = cfg.n_samples // 2, cfg.release_sample // 2
    return replace(
        cfg,
        n_samples=n,
        release_sample=k_release,
        midpoints=False,
        # its two-step way in needs room: without it, the coarse path is only a first guess
        approach_from_above=cfg.approach_from_above and n >= k_release + 5,
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
    return _solve(p, what, cfg, check, seed=seed)


# --- Starting points ----------------------------------------------------------------------------


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
