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
Cost: w_tau sum ||tau||^2 (samples and midpoints) + w_smooth sum ||dq_{k+1} - dq_k||^2 + posture,
and, if w_swing is set, the moving ball's distance^2 to a reference curve in phase B (swing_point).

A move of a plan is one transition: it starts where the previous one ended (the plan's first at
stance 0's own pose, stance_pose) and ends with the moving ball on its new hold, the rest of the
body wherever is cheapest. Collisions (with the holds, the wall or itself) are not modelled; the
swing curve only steers the moving ball away from them.

Each problem is built once and re-solved with new numbers (its start, holds and target are
CasADi parameters): building the derivatives costs ~3x more than solving. A transition's
structure depends only on which limb moves, so there are at most four, plus the static pose.

Wall frame (INTERFACES.md): x right, z up, y into the wall, the robot at y < 0, gravity -z.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace

import casadi as ca
import numpy as np
from climbing_sim import scene as sim_scene
from climbing_sim import wall as sim_wall

from rcr.optimiser import robot_model as rm
from rcr.optimiser.robot_model import GRAVITY, NJ, NQ, NV, RobotModel, load_robot_model
from rcr.robot import hooks
from rcr.types import LIMBS, MICROBAN_JOINTS, Limb, Stance, WallModel

# [N·m] the current limit times BAM's XL330 torque constant: 1.75 A x 0.346 N·m/A, where the
# organisers' sim clips each motor [Microban constants.py BAM_MAX_CURRENT, microban_sim]
SERVO_TORQUE_LIMIT = 0.605
GUESS_STANDOFF = -0.10  # [m] trunk y in the first static pose's guess, in front of the wall [Ours]
FACING_WALL = (np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4))  # (w,x,y,z): trunk front along +y
# [m] the hold's socket radius: the inner surface of the organisers' sim's socket spheres
SOCKET_RADIUS = round(
    min(np.linalg.norm(s["pos"]) - s["r"] for s in sim_scene.HOLD_SPHERES if s["pos"][2] < 0), 6
)

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
    # the hold's spherical socket, 26 mm out and 14 mm below the bolts (the organisers' sim,
    # wall.SOCKET_OUT and SOCKET_BELOW), lowered by the socket/ball clearance (12.7 - 12.5 mm):
    # where a ball rests in their sim. (0, -0.026, -0.0142) today.
    contact_offset: tuple[float, float, float] = (
        0.0,
        -sim_wall.SOCKET_OUT,
        -sim_wall.SOCKET_BELOW - (SOCKET_RADIUS - hooks.BALL_RADIUS),
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
    # closed chain exactly on four points [Ours].
    midpoint_play: float = 0.001  # [m]
    # A reference path for the moving ball in phase B, preferred softly (swing_point): it leaves
    # its cup straight up, is swing_lift higher and swing_out further from the wall halfway, and
    # comes straight down into the new cup. Keeps the ball off the wall and the cups' rims without
    # collision constraints; not a guarantee. 0 = off (the runner's --swing turns it on) [Ours].
    w_swing: float = 5000.0  # weight on ||ball - curve||^2 [1/m^2]
    swing_lift: float = 0.03  # [m] above the line between the two holds, halfway
    swing_out: float = 0.03  # [m] out from the wall (-y), halfway
    phase_a_duration: float = 1.0  # [s] for phase A, chosen after solving (quasi-static) [Ours]
    phase_b_duration: float = 2.0  # [s] for phase B [Ours]
    ipopt: dict = field(
        # mu_init 1e-3 and max_iter 150: from re-solves that start next to their answer; a solve
        # that runs out continues from where it stopped (solve_opti's patience) [Ours].
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
    # N + 1 samples; times from rcr.optimiser.sample_time
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
    """Every joint within its URDF limits, and every motor torque within +-tau_limit."""
    opti.subject_to(opti.bounded(model.lower, q[7:], model.upper))
    opti.subject_to(opti.bounded(-cfg.tau_limit, tau, cfg.tau_limit))


def trunk_bounds(opti: ca.Opti, dq, cfg: TransitionConfig) -> None:
    """The trunk stays within max_trunk_shift / max_trunk_turn of the reference configuration."""
    opti.subject_to(opti.bounded(-cfg.max_trunk_shift, dq[0:3], cfg.max_trunk_shift))
    opti.subject_to(opti.bounded(-cfg.max_trunk_turn, dq[3:6], cfg.max_trunk_turn))


def balanced_pose(opti, model, dq, q_ref, forces, cfg):
    """The constraints every sample (and midpoint) shares. Returns (q, tau), the pose and its
    torques. Subject to:
        the trunk near the reference pose      trunk_bounds
        the trunk balanced                     static_balance(...) == 0
        joint limits, |torque| <= tau_limit    joint_and_torque_limits"""
    trunk_bounds(opti, dq, cfg)
    q = model.integrate(q_ref, dq)
    wrench, tau = static_balance(model, q, forces)
    opti.subject_to(wrench == 0)
    joint_and_torque_limits(opti, model, q, tau, cfg)
    return q, tau


def posture(q, cfg: TransitionConfig):
    """Soft preference for the reference posture: the leg joints near cfg.nominal, and the spine
    leaning spine_target from vertical. spine_z, the vertical component of the trunk's own z axis,
    is 1 - 2 (x^2 + y^2) for its quaternion (w, x, y, z)."""
    joint = {name: q[7 + i] for i, name in enumerate(MICROBAN_JOINTS)}
    legs = sum((joint[name] - angle) ** 2 for name, angle in cfg.nominal.items())
    spine_z = 1 - 2 * (q[4] ** 2 + q[5] ** 2)
    return cfg.w_nominal * legs + cfg.w_spine * (spine_z - np.cos(cfg.spine_target)) ** 2


def swing_point(start, target, s: float, cfg: TransitionConfig):
    """The moving ball's reference at s in [0, 1] (phase B's progress), from `start` (its old
    contact point) to `target` (its new one): a quartic Bezier curve whose end tangents are
    vertical (out of and into the cups, which open upwards), swing_lift up and swing_out out from
    the wall (-y) at s = 0.5. Works on numbers and on CasADi expressions alike."""
    # At s = 0.5 the Bernstein weights are (1, 4, 6, 4, 1)/16: the middle point counts 6/16 and
    # the two lifted end points 4/16 each, so these are scaled to put the peak where asked
    up = np.array([0.0, 0.0, cfg.swing_lift / (14 / 16)])
    out = np.array([0.0, -cfg.swing_out / (6 / 16), 0.0])
    points = [start, start + up, (start + target) / 2 + up + out, target + up, target]
    weights = [math.comb(4, i) * s**i * (1 - s) ** (4 - i) for i in range(5)]
    return sum(w * p for w, p in zip(weights, points, strict=True))


def midpoint_poses(opti: ca.Opti, model: RobotModel, dq, q0, holds, moving: Limb, cfg):
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
    objective: ca.MX
    q_ref: ca.MX  # parameter NQ: the configuration dq is measured from
    holds: ca.MX  # parameter 3x4: the contact points of the start stance, LIMBS order
    target: ca.MX | None = None  # parameter 3: the moving limb's new contact point
    mid_forces: dict | None = None  # limb -> 3 x N hold forces at the interval midpoints


_problems: dict[tuple, _Problem] = {}


def _built(key: tuple, build) -> _Problem:
    """The problem built for this key, built on first use (building costs more than solving)."""
    if key not in _problems:
        _problems[key] = build()
    return _problems[key]


def static_pose_problem(cfg: TransitionConfig) -> _Problem:
    """One pose the robot can hold still, all four balls on their holds.

    Variables: dq (the pose, from q_ref), and each hold's force on its ball.
    Subject to: balanced_pose (balance, limits, trunk near q_ref); every ball on its hold;
        every hold pushing within its friction pyramid.
    Cost: w_tau ||tau||^2 + w_stay ||dq||^2 + posture."""

    def build() -> _Problem:
        model, opti = load_robot_model(), ca.Opti()
        q_ref, holds = opti.parameter(NQ), opti.parameter(3, len(LIMBS))
        dq = opti.variable(NV, 1)
        forces = {limb: opti.variable(3, 1) for limb in LIMBS}
        # --- subject to
        q, tau = balanced_pose(opti, model, dq[:, 0], q_ref, forces, cfg)
        opti.subject_to(model.limb_positions(q) == holds)  # every ball on its hold
        for limb in LIMBS:
            friction_pyramid(opti, forces[limb], cfg.normals[limb], cfg.mu[limb])
        # --- cost
        objective = cfg.w_tau * ca.sumsqr(tau) + cfg.w_stay * ca.sumsqr(dq) + posture(q, cfg)
        p = _Problem(opti, dq, forces, [q], [tau], objective, q_ref, holds)
        _minimise(p, cfg, cold=True)
        return p

    return _built(("static", repr(cfg)), build)


def transition_problem(moving: Limb, cfg: TransitionConfig) -> _Problem:
    """One move: the `moving` limb goes from its hold to a new one, samples k = 0..N.

    Variables, per sample: dq_k (the pose, from the start q0), and each hold's force on its ball.
    Phases: A, k = 0..K, all four balls held; the moving limb's force reaches 0 at K.
            B, k = K..N, the other three held; the moving ball travels, on its new hold at N.
    Subject to:
        sample 0 is the start pose                                  dq_0 = 0
        every sample: balance, limits, trunk near q0                balanced_pose
        a held ball on its hold (the moving one up to K)            balls == holds
        a held ball pushed within its friction pyramid (up to K-1)  friction_pyramid
        the moving ball carries nothing from K on                   force == 0
        the moving ball on its new hold at N                        balls[N] == target
        the same at every midpoint between samples                  midpoint_poses
    Cost:
        w_tau    * sum ||tau||^2              over the samples and midpoints (low torque)
      + w_smooth * sum ||dq_{k+1} - dq_k||^2  between samples (smooth)
      + sum posture(q)                        over the samples and midpoints (legs, spine)
      + w_swing * sum ||ball(q) - swing_point||^2  phase B's samples and midpoints (if w_swing)"""
    n, k_release = cfg.n_samples, cfg.release_sample
    if not 0 < k_release < n - 2:
        raise ValueError(f"need 0 < release_sample < n_samples - 2, got {k_release}, {n}")

    def build() -> _Problem:
        model, opti = load_robot_model(), ca.Opti()
        q0, holds, target = opti.parameter(NQ), opti.parameter(3, len(LIMBS)), opti.parameter(3)
        dq = opti.variable(NV, n + 1)
        forces = {limb: opti.variable(3, n + 1) for limb in LIMBS}
        # --- subject to
        opti.subject_to(dq[:, 0] == 0)  # sample 0 is the start pose
        m = LIMBS.index(moving)
        qs, taus = [], []
        for k in range(n + 1):
            sample = {limb: forces[limb][:, k] for limb in LIMBS}
            q, tau = balanced_pose(opti, model, dq[:, k], q0, sample, cfg)
            balls = model.limb_positions(q)
            for i, limb in enumerate(LIMBS):
                if limb != moving or k <= k_release:  # held (the moving ball until it lets go)
                    opti.subject_to(balls[:, i] == holds[:, i])
                if limb != moving or k < k_release:  # carrying weight
                    friction_pyramid(opti, sample[limb], cfg.normals[limb], cfg.mu[limb])
                else:  # the moving ball, from K on: carries nothing
                    opti.subject_to(sample[limb] == 0)
            if k == n:  # the moving ball on its new hold
                opti.subject_to(balls[:, m] == target)
            qs.append(q)
            taus.append(tau)
        mid_forces, mid_taus, mid_qs = midpoint_poses(opti, model, dq, q0, holds, moving, cfg)
        # --- cost
        torque = sum(ca.sumsqr(t) for t in taus + mid_taus)
        smooth = sum(ca.sumsqr(dq[:, k + 1] - dq[:, k]) for k in range(n))
        legs_and_spine = sum(posture(q, cfg) for q in qs + mid_qs)
        objective = cfg.w_tau * torque + cfg.w_smooth * smooth + legs_and_spine
        if cfg.w_swing:  # the moving ball near its swing curve, phase B's samples and midpoints
            span = n - k_release
            on_curve = [(qs[k], (k - k_release) / span) for k in range(k_release, n + 1)]
            if mid_qs:
                on_curve += [(mid_qs[k], (k + 0.5 - k_release) / span) for k in range(k_release, n)]
            swing = sum(
                ca.sumsqr(model.limb_positions(q)[:, m] - swing_point(holds[:, m], target, s, cfg))
                for q, s in on_curve
            )
            objective = objective + cfg.w_swing * swing
        p = _Problem(opti, dq, forces, qs, taus, objective, q0, holds, target, mid_forces)
        _minimise(p, cfg)
        return p

    return _built(("transition", moving, repr(cfg)), build)


def _minimise(p: _Problem, cfg: TransitionConfig, cold: bool = False) -> None:
    """Set the problem's cost and IPOPT. cold: the problem starts from a rough guess (the static
    pose), not next to its answer, so IPOPT's own barrier start and iteration budget: with cfg's
    small mu_init it found a start pose with a joint 3.8 rad from the usual one [Ours]."""
    p.opti.minimize(p.objective)
    options = cfg.ipopt
    if cold:
        options = {k: v for k, v in options.items() if k != "mu_init"} | {"max_iter": 1000}
    p.opti.solver("ipopt", {"print_time": False, "expand": True}, options)


# --- Solving ------------------------------------------------------------------------------------


def solve_opti(opti: ca.Opti, what: str, patience: int = 0) -> tuple[ca.OptiSol, float]:
    """Run IPOPT: (solution, wall time [s]).

    If IPOPT runs out of iterations, it continues from where it stopped up to `patience` more
    times (for a solve that starts far from its answer). If it still runs out at a point that
    satisfies every constraint (to FEASIBLE_TOL), that point is returned: it is a valid motion,
    only not certified optimal. Otherwise OptimisationError with the status and the largest
    constraint violation.
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
# A solve starts far from its answer: it may continue this many times past max_iter (150 each,
# ~1000 in all) [Ours].
PATIENCE = 6


@dataclass
class _Solution:
    """What one solve gave, read out of CasADi."""

    dq: np.ndarray  # NV x samples
    q: np.ndarray  # samples x NQ
    tau: np.ndarray  # samples x 19
    forces: dict[str, np.ndarray]  # limb -> samples x 3
    solve_time: float
    iterations: int


def _solve(p: _Problem, what: str) -> _Solution:
    """IPOPT on the problem as it is set (its parameters and starting point)."""
    sol, solve_time = solve_opti(p.opti, what, PATIENCE)
    return _Solution(
        dq=np.array(sol.value(p.dq)).reshape(NV, -1),
        q=np.array([np.array(sol.value(q)).ravel() for q in p.q]),
        tau=np.array([np.array(sol.value(t)).ravel() for t in p.tau]),
        forces={limb: np.array(sol.value(p.forces[limb])).reshape(3, -1).T for limb in LIMBS},
        solve_time=solve_time,
        iterations=int(sol.stats().get("iter_count", 0)),
    )


def static_pose(wall, stance, q_ref, cfg: TransitionConfig, index: int = 0) -> _Solution:
    """A statically balanced configuration with every ball on its hold: the same constraints as
    one sample of a transition, all four limbs in contact; cost w_tau ||tau||^2 + w_stay ||dq||^2,
    dq measured from q_ref. Returns the whole solution (q, tau, forces)."""
    check_holds(wall, stance, index)
    p = static_pose_problem(cfg)
    p.opti.set_value(p.q_ref, q_ref)
    p.opti.set_value(p.holds, contact_points(wall, stance, cfg))
    _fresh(p)
    p.opti.set_initial(p.dq, np.zeros((NV, 1)))
    for limb in LIMBS:
        p.opti.set_initial(p.forces[limb], weight_share(len(LIMBS)))
    return _solve(p, f"static pose for stance {index}")


_stance_poses: dict[tuple, np.ndarray] = {}


def stance_pose(wall: WallModel, stance: Stance, cfg: TransitionConfig, index: int = 0):
    """The stance's own pose (NQ): balanced, solved from a fresh guess (initial_guess), so it
    depends on the stance and the wall only. The plan starts at stance 0's. Kept per wall, stance
    and settings for the run."""
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
    rescue: bool = True,
) -> TransitionResult:
    """Move the one limb that changes from its hold in `a` to its hold in `b`, starting at q0.

    q0 must have every ball on its hold of `a` (e.g. stance_pose, or the previous transition's
    end). Only the moving limb's final point is fixed, not the final configuration.
    rescue: if the solve fails, solve the move coarsely (coarse()) and start from that path.
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
    try:
        s = _solve(p, what)
    except OptimisationError as e:
        if not (rescue and _halves(cfg)):
            raise
        s = _rescue(wall, a, b, q0, cfg, index, p, what, e)
    return TransitionResult(
        q=s.q, dq=s.dq, tau=s.tau, forces=s.forces, moving=moving,
        solve_time=s.solve_time, iterations=s.iterations,
    )  # fmt: skip


# --- When the solve fails: a coarse path first --------------------------------------------------


def coarse(cfg: TransitionConfig) -> TransitionConfig:
    """The same move with half the samples and no midpoints: a few seconds, for a first path."""
    return replace(
        cfg,
        n_samples=cfg.n_samples // 2,
        release_sample=cfg.release_sample // 2,
        midpoints=False,
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


def _rescue(wall, a, b, q0, cfg, index, p, what, error) -> _Solution:
    """The solve failed: find a coarse path (fewer, freer samples) and start from it. On the
    example's hard wall this solved a hand move (R3 to R5) that failed from every other start."""
    try:
        rough = solve_transition(wall, a, b, q0, coarse(cfg), index, rescue=False)
    except OptimisationError:
        raise error from None  # no coarse path either: report the full solve's failure
    _warm_start(p, wall, b, q0, moving_limb(a, b, index), cfg, index, upsample(rough.dq))
    return _solve(p, what)


# --- Starting points ----------------------------------------------------------------------------


def _fresh(p: _Problem) -> None:
    """Forget the previous solve: every variable and multiplier starts at zero unless set after.
    The problems are reused, and without this a move solved or failed depending on what was
    solved before it."""
    p.opti.set_initial(p.opti.x, np.zeros(p.opti.nx))
    p.opti.set_initial(p.opti.lam_g, np.zeros(p.opti.ng))


def _warm_start(p: _Problem, wall, b, q0, moving, cfg, index, path=None) -> None:
    """dq: `path` if given (a coarse solution, see _rescue), else still in phase A, then a
    straight line (in the tangent space at q0) to a static pose of stance b, solved from a fresh
    guess so that it is in the reference posture rather than wherever q0 leads; forces: the
    weight shared over the limbs carrying it. With a path, the rest of the problem's state
    (multipliers) is kept; without one, everything starts afresh (_fresh)."""
    n, k_release = cfg.n_samples, cfg.release_sample
    if path is not None:
        p.opti.set_initial(p.dq, path)
    else:
        _fresh(p)
        start = initial_guess(wall, b, cfg)
        q_end = static_pose(wall, b, start, cfg, index + 1).q[0]
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
    """A rough configuration to start a static pose from: the trunk in front of the stance's
    holds, upright and facing the wall, the legs in the reference posture and the other motors
    at zero [Ours]."""
    cfg = cfg or TransitionConfig()
    points = contact_points(wall, stance, cfg)
    model = load_robot_model()
    q = np.zeros(7 + NJ)
    q[0:3] = [points[0].mean(), GUESS_STANDOFF, points[2].mean()]
    q[3:7] = FACING_WALL
    q[7:] = [cfg.nominal.get(name, 0.0) for name in MICROBAN_JOINTS]  # the legs as the reference
    q[7:] = np.clip(q[7:], model.lower, model.upper)
    return q
