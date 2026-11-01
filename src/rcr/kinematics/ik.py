
"""Kinematic trajectory generation for the climbing robot.

The moving limb follows a piecewise-linear task-space path.
The other three limbs stay in their holds.

Selectable inverse kinematics: hierarchical pseudoinverse or constrained
lexicographic quadprog, following the same Advanced Robotics tasks:

1. Keep supporting limbs fixed.
2. Track the moving limb.
3. Stay close to the nominal hanging posture.

No dynamics or collision avoidance is included in the IK solver.
"""

import microban_model
import numpy as np
import pinocchio as pin
import quadprog
from numpy.linalg import norm, pinv

from rcr.planner.astar import (
    Q_HANG,
    initial_q,
    targets as stance_targets,
)
from rcr.types import (
    LIMBS,
    MICROBAN_JOINTS,
    ContactPlan,
    Feasibility,
    Stance,
    Trajectory,
    WallModel,
)
from rcr.kinematics.bezier import Bezier, connecting


# ---------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------

DT = 0.02
MOVE_TIME = 2.0

ABOVE = 0.02
OUT = 0.04

EPSILON = 1e-3
MAX_ITERATIONS = 200
IK_STEP = 0.2
POSTURE_GAIN = 0.2
LIMIT_GAIN = 0.02  # soft push towards joint-limit midpoints
VELOCITY_GAIN = 0.02  # soft preference for the previous output sample

# Numerical safeguard on each integration step.
MAX_STEP_NORM = 0.1


# ---------------------------------------------------------------------
# Robot model
# ---------------------------------------------------------------------

model = pin.buildModelFromUrdf(
    str(microban_model.urdf(climb=True)),
    pin.JointModelFreeFlyer(),
)

data = model.createData()

BALL = {
    limb: model.getFrameId(f"{limb}_ball")
    for limb in LIMBS
}

# Configuration indices in the robot controller's joint order.
JOINTS = [
    model.idx_qs[model.getJointId(name)]
    for name in MICROBAN_JOINTS
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def stance_key(stance: Stance):
    return tuple(stance.contacts[limb] for limb in LIMBS)


def project_to_joint_limits(q):
    """Clamp actuated joints, leaving the free flyer unchanged."""
    q = q.copy()
    q[7:] = np.clip(
        q[7:],
        model.lowerPositionLimit[7:],
        model.upperPositionLimit[7:],
    )
    return q


def ball_positions(q):
    """World positions of all four balls, in LIMBS order."""
    pin.framesForwardKinematics(model, data, q)

    return np.array([
        data.oMf[BALL[limb]].translation.copy()
        for limb in LIMBS
    ])


# ---------------------------------------------------------------------
# Task-space trajectory
# ---------------------------------------------------------------------


def limb_traj(p_a, p_b):
    """C2 piecewise Bezier trajectory through the six ball waypoints.

    Each segment connects two states (position, velocity, acceleration)
    using C-CROC's connecting().

    Returns B(t), with B.velocity(t) and B.acceleration(t).
    """
    p_a = np.asarray(p_a, dtype=float)
    p_b = np.asarray(p_b, dtype=float)

    up = np.array([0.0, 0.0, ABOVE])
    out = np.array([0.0, -OUT, 0.0])

    points = np.array([
        p_a,
        p_a + up,
        p_a + up + out,
        p_b + up + out,
        p_b + up,
        p_b,
    ])

    # Allocate time proportionally to segment length.
    segments = np.diff(points, axis=0)
    lengths = norm(segments, axis=1)

    if np.any(lengths <= 1e-10):
        raise ValueError("Degenerate limb trajectory")

    times = MOVE_TIME * np.r_[
        0.0, np.cumsum(lengths) / lengths.sum()
    ]
    durations = np.diff(times)

    # Desired velocities at the waypoints.
    velocities = np.zeros_like(points)

    # Estimate tangent directions using neighboring segments.
    for i in range(1, len(points) - 1):
        before = segments[i - 1] / lengths[i - 1]
        after = segments[i] / lengths[i]

        tangent = before + after

        if norm(tangent) > 1e-10:
            tangent /= norm(tangent)

            speed = min(
                lengths[i - 1] / durations[i - 1],
                lengths[i] / durations[i],
            )

            velocities[i] = 0.5 * speed * tangent

    # Stop at the initial and final sockets.
    velocities[0] = 0.0
    velocities[-1] = 0.0

    # Equal acceleration at each shared waypoint.
    # Zero is a simple first choice.
    accelerations = np.zeros_like(points)

    # Build five C-CROC Bezier segments.
    curves = []

    for i in range(len(points) - 1):
        start = (points[i], velocities[i], accelerations[i])
        end = (points[i + 1], velocities[i + 1], accelerations[i + 1])

        curves.append(
            Bezier(
                connecting(start, end, durations[i]),
                t_min=float(times[i]),
                t_max=float(times[i + 1]),
            )
        )

    def evaluate(t, derivative=0):
        t = float(np.clip(t, 0.0, MOVE_TIME))

        i = min(
            np.searchsorted(times[1:], t, side="left"),
            len(curves) - 1,
        )

        return curves[i].derivative(derivative)(t)

    def B(t):
        return evaluate(t)

    B.velocity = lambda t: evaluate(t, 1)
    B.acceleration = lambda t: evaluate(t, 2)

    B.waypoints = points
    B.times = times

    return B



# ---------------------------------------------------------------------
# Inverse kinematics
# ---------------------------------------------------------------------

def ik_pinv(q0, targets, moving=None, dt=None):
    """Solve IK from configuration q0.

    targets: dict mapping limb names to desired XYZ positions.

    moving:
        None  -> all four limbs have equal priority (stance IK).
        limb  -> other three limbs first, moving limb second.

    Nominal hanging joint posture is the lowest-priority task.

    If dt is supplied, all inner IK iterations share joint bounds
    defined relative to q0 (the previous OUTPUT sample).
    """
    q = q0.copy()

    held = [
        limb for limb in LIMBS
        if limb != moving
    ]

    lower = model.lowerPositionLimit[7:].copy()
    upper = model.upperPositionLimit[7:].copy()
    if dt is not None:
        vmax = model.velocityLimit[6:]
        if np.any(vmax <= 0):
            raise ValueError("Nonpositive URDF joint velocity limit")
        lower = np.maximum(lower, q0[7:] - vmax * dt)
        upper = np.minimum(upper, q0[7:] + vmax * dt)
    if np.any(lower > upper + 1e-10):
        raise ValueError("Empty joint-position/velocity interval")

    q_mid = (model.lowerPositionLimit[7:] + model.upperPositionLimit[7:]) / 2
    finite_mid = np.isfinite(q_mid)


    for _ in range(MAX_ITERATIONS):

        pin.computeJointJacobians(model, data, q)
        pin.updateFramePlacements(model, data)

        errors = {}
        J = {}

        for limb in LIMBS:
            frame = BALL[limb]

            p = data.oMf[frame].translation
            errors[limb] = targets[limb] - p

            J[limb] = pin.getFrameJacobian(
                model,
                data,
                frame,
                pin.LOCAL_WORLD_ALIGNED,
            )[:3]

        # Convergence of all end-effectors
        if max(norm(e) for e in errors.values()) < EPSILON:
            return q

        # Priority 1: supporting limbs
        Jc = np.vstack([J[limb] for limb in held])
        ec = np.concatenate([errors[limb] for limb in held])

        Jc_inv = pinv(Jc)

        dq = Jc_inv @ ec
        P = np.eye(model.nv) - Jc_inv @ Jc

        # Priority 2: moving limb
        if moving is not None:
            Jm = J[moving]
            em = errors[moving]

            dq += pinv(Jm @ P) @ (em - Jm @ dq)

        # Priority 3: nominal hanging joint posture
        # Project into the null space of ALL ball constraints.
        J_all = np.vstack([J[limb] for limb in LIMBS])
        P_all = np.eye(model.nv) - pinv(J_all) @ J_all

        soft = np.zeros(model.nv)
        soft[6:] = POSTURE_GAIN * (Q_HANG[7:] - q[7:])
        soft[6:][finite_mid] += LIMIT_GAIN * (q_mid[finite_mid] - q[7:][finite_mid])
        if dt is not None:
            soft[6:] += VELOCITY_GAIN * (q0[7:] - q[7:])

        dq += P_all @ soft

        # Bounded integration step
        step = IK_STEP * dq
        step_norm = norm(step)

        if step_norm > MAX_STEP_NORM:
            step *= MAX_STEP_NORM / step_norm

        q = pin.integrate(model, q, step)
        # Clamp against the SAME q0-based velocity interval on every
        # numerical iteration, not a new interval around the latest iterate.
        q[7:] = np.clip(q[7:], lower, upper)

        if not np.all(np.isfinite(q)):
            raise ValueError("IK produced non-finite configuration")

    raise ValueError("IK did not converge")


# ---------------------------------------------------------------------
# QP solver (same priorities, explicit Cartesian slack and joint bounds)
# ---------------------------------------------------------------------

# Tiny tie-breaker gives strictly positive definite QP Hessians.
RIDGE = 1e-6
SLACK_WEIGHT = 1e3
# Per-iteration numerical trust region (not a physical joint velocity limit).
MAX_INNER_JOINT_STEP = 0.1
MAX_INNER_BASE_TRANSLATION = 0.05
MAX_INNER_BASE_ROTATION = 0.15


def _qp(G, a, E, f, lo, hi):
    """min .5*x'G*x - a'x subject to E*x=f and lo<=x<=hi."""
    n = len(a)
    cols = [row.copy() for row in E]
    rhs = list(f)
    for j in range(n):
        if np.isfinite(lo[j]):
            c = np.zeros(n)
            c[j] = 1.0
            cols.append(c)
            rhs.append(lo[j])
        if np.isfinite(hi[j]):
            c = np.zeros(n)
            c[j] = -1.0
            cols.append(c)
            rhs.append(-hi[j])
    C = np.column_stack(cols) if cols else np.empty((n, 0))
    try:
        return quadprog.solve_qp(G, a, C, np.asarray(rhs), meq=len(f))[0]
    except ValueError as exc:
        raise ValueError(f"QP solver failed: {exc}") from exc


def _step_bounds(q, sample_q, dt):
    """Bounds on tangent-space correction relative to *current* q.

    Output joint bounds are always relative to the SAME previous output
    configuration (sample_q), not the previous inner IK iterate.
    """
    nv = model.nv
    lo = np.full(nv, -np.inf)
    hi = np.full(nv, np.inf)
    lo[:3] = -MAX_INNER_BASE_TRANSLATION
    hi[:3] = MAX_INNER_BASE_TRANSLATION
    lo[3:6] = -MAX_INNER_BASE_ROTATION
    hi[3:6] = MAX_INNER_BASE_ROTATION

    qlo = model.lowerPositionLimit[7:].copy()
    qhi = model.upperPositionLimit[7:].copy()
    if dt is not None:
        vmax = model.velocityLimit[6:]
        if np.any(vmax <= 0):
            raise ValueError("Nonpositive actuated-joint velocity limit")
        qlo = np.maximum(qlo, sample_q[7:] - vmax * dt)
        qhi = np.minimum(qhi, sample_q[7:] + vmax * dt)
    if np.any(qlo > qhi + 1e-9):
        raise ValueError("Empty position/velocity feasible interval")
    lo[6:] = np.maximum(qlo - q[7:], -MAX_INNER_JOINT_STEP)
    hi[6:] = np.minimum(qhi - q[7:], MAX_INNER_JOINT_STEP)
    if np.any(lo > hi + 1e-9):
        raise ValueError("No permitted IK correction within joint limits")
    return lo, hi


def _task_qp(J, e, lo, hi, frozen_J=None, frozen_value=None):
    """Track Cartesian task with explicit slack under joint bounds.

    Optionally preserve the already-achieved previous-priority displacement.
    """
    nv = model.nv
    ne = len(e)
    n = nv + ne
    G = np.diag(np.r_[np.full(nv, RIDGE), np.full(ne, SLACK_WEIGHT)])
    a = np.zeros(n)
    # Equality: J*d + slack = e
    E = np.hstack((J, np.eye(ne)))
    f = np.asarray(e)
    if frozen_J is not None:
        frozen = np.hstack((frozen_J, np.zeros((len(frozen_value), ne))))
        E = np.vstack((E, frozen))
        f = np.r_[f, frozen_value]
    lower = np.r_[lo, np.full(ne, -np.inf)]
    upper = np.r_[hi, np.full(ne, np.inf)]
    return _qp(G, a, E, f, lower, upper)[:nv]


def ik_qp(q0, targets, moving=None, dt=None):
    """QP alternative with the same targets and priorities as ik_pinv.

    q0 is the previous 50 Hz output configuration. It remains the velocity-
    constraint reference across *all* inner numerical IK iterations.
    """
    q = q0.copy()
    held = [limb for limb in LIMBS if limb != moving]
    midpoint = (model.lowerPositionLimit[7:] + model.upperPositionLimit[7:]) / 2
    finite_mid = np.isfinite(midpoint)

    for _ in range(MAX_ITERATIONS):
        pin.computeJointJacobians(model, data, q)
        pin.updateFramePlacements(model, data)

        J, error = {}, {}
        for limb in LIMBS:
            frame = BALL[limb]
            error[limb] = targets[limb] - data.oMf[frame].translation
            J[limb] = pin.getFrameJacobian(
                model, data, frame, pin.LOCAL_WORLD_ALIGNED
            )[:3]

        if max(norm(e) for e in error.values()) < EPSILON:
            return q

        lo, hi = _step_bounds(q, q0, dt)
        Jc = np.vstack([J[limb] for limb in held])
        ec = np.concatenate([error[limb] for limb in held])

        # Priority 1: minimize held-ball slack under hard joint bounds.
        d = _task_qp(Jc, IK_STEP * ec, lo, hi)
        held_achieved = Jc @ d

        if moving is not None:
            # Priority 2: improve moving ball WITHOUT undoing held correction.
            Jm = J[moving]
            d = _task_qp(
                Jm, IK_STEP * error[moving], lo, hi,
                frozen_J=Jc, frozen_value=held_achieved,
            )
            E = np.vstack((Jc, Jm))
        else:
            E = Jc

        # Priority 3: same nominal posture, joint-centering and motion
        # preferences as the bounded pseudoinverse solver. Exactly preserve
        # higher-priority LINEARIZED Cartesian changes.
        soft = np.zeros(model.nv)
        soft[6:] = POSTURE_GAIN * (Q_HANG[7:] - q[7:])
        soft[6:][finite_mid] += LIMIT_GAIN * (midpoint[finite_mid] - q[7:][finite_mid])
        if dt is not None:
            soft[6:] += VELOCITY_GAIN * (q0[7:] - q[7:])

        target_d = d + IK_STEP * soft
        d = _qp(
            np.eye(model.nv), target_d,
            E, E @ d, lo, hi,
        )

        q = pin.integrate(model, q, d)
        if not np.all(np.isfinite(q)):
            raise ValueError("QP IK produced non-finite configuration")

    raise ValueError("QP IK did not converge to Cartesian tolerance")




def solve_ik(q0, targets, moving=None, dt=None, method="qp"):
    """Choose between the two solvers, without changing the IK task.

    method="pinv" : hierarchical pseudoinverse + bounded joint clipping.
    method="qp"   : hierarchical constrained quadratic programs.
    """
    if method == "pinv":
        return ik_pinv(q0, targets, moving=moving, dt=dt)
    if method == "qp":
        return ik_qp(q0, targets, moving=moving, dt=dt)
    raise ValueError(f"Unknown IK method: {method!r}")

# ---------------------------------------------------------------------
# Initial stance
# ---------------------------------------------------------------------


def start_pose(wall: WallModel, stance: Stance, method="qp"):
    """Find a configuration with all four balls in their sockets.

    Start from the nominal hanging configuration rigidly fitted to
    the stance, then solve positional IK.
    """
    p = stance_targets(wall, stance_key(stance))

    targets = {
        limb: p[i]
        for i, limb in enumerate(LIMBS)
    }

    q0 = initial_q(p)

    return solve_ik(q0, targets, moving=None, method=method)


# ---------------------------------------------------------------------
# Single movement
# ---------------------------------------------------------------------

def ik_traj(wall: WallModel, a: Stance, b: Stance, q0, method="qp"):
    """Generate a joint trajectory from stance a to stance b.

    The moving ball follows limb_traj; the others remain fixed.
    Each IK problem is warm-started from the previous configuration.

    Returns an array of shape (N, model.nq), including q0.
    """
    moving = [
        limb for limb in LIMBS
        if a.contacts[limb] != b.contacts[limb]
    ]

    if len(moving) != 1:
        raise ValueError("Exactly one limb must move")

    moving = moving[0]
    i = LIMBS.index(moving)

    p_a = stance_targets(wall, stance_key(a))
    p_b = stance_targets(wall, stance_key(b))

    B = limb_traj(p_a[i], p_b[i])

    q = q0.copy()
    qs = [q.copy()]

    times = np.linspace(
        0.0,
        MOVE_TIME,
        round(MOVE_TIME / DT) + 1,
    )

    for t in times[1:]:

        targets = {
            limb: p_a[j].copy()
            for j, limb in enumerate(LIMBS)
        }

        targets[moving] = B(t)

        q = solve_ik(q, targets, moving, dt=DT, method=method)

        qs.append(q.copy())

    return np.asarray(qs)


# ---------------------------------------------------------------------
# Feasibility
# ---------------------------------------------------------------------

def feasible_ik_traj(
    wall: WallModel,
    a: Stance,
    b: Stance,
    method="qp",
) -> Feasibility:
    """Preliminary feasibility check from a nominal starting stance.

    Actual chained trajectories must be validated separately.
    """
    try:
        q0 = start_pose(wall, a, method=method)
        qs = ik_traj(wall, a, b, q0, method=method)

        if not np.all(np.isfinite(qs)):
            return Feasibility(
                False,
                reason="Non-finite trajectory",
            )

        if np.any(qs[:, 7:] < model.lowerPositionLimit[7:] - 1e-8):
            return Feasibility(
                False,
                reason="Lower joint limit violated",
            )

        if np.any(qs[:, 7:] > model.upperPositionLimit[7:] + 1e-8):
            return Feasibility(
                False,
                reason="Upper joint limit violated",
            )

        if not check_velocity_limits(qs):
            return Feasibility(False, reason="Joint velocity limits exceeded")

        return Feasibility(True)

    except (ValueError, np.linalg.LinAlgError) as e:
        return Feasibility(False, reason=str(e))


# ---------------------------------------------------------------------
# Complete climbing trajectory
# ---------------------------------------------------------------------

def generate_trajectory(wall: WallModel, plan: ContactPlan, method="qp", q0=None):
    """Chain all movements without resetting configurations."""
    stances = plan.stances

    if not stances:
        raise ValueError("Contact plan is empty")

    q = start_pose(wall, stances[0], method=method) if q0 is None else q0.copy()
    qs = [q.copy()]

    for a, b in zip(stances[:-1], stances[1:]):
        segment = ik_traj(wall, a, b, q, method=method)

        # Avoid duplicating the transition configuration.
        qs.extend(segment[1:])

        q = segment[-1].copy()

    return np.asarray(qs)


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------

def to_trajectory(qs: np.ndarray) -> Trajectory:
    """Convert Pinocchio configurations to the robot trajectory type."""
    times = [
        round(k * DT, 9)
        for k in range(len(qs))
    ]

    positions = [
        [float(q[j]) for j in JOINTS]
        for q in qs
    ]

    # Pinocchio quaternion: (x, y, z, w)
    # Trajectory quaternion: (w, x, y, z)
    trunk_poses = [
        [float(v) for v in q[[0, 1, 2, 6, 3, 4, 5]]]
        for q in qs
    ]

    return Trajectory(
        joint_names=list(MICROBAN_JOINTS),
        times=times,
        positions=positions,
        planned_positions=[row.copy() for row in positions],
        trunk_poses=trunk_poses,
    )


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------

def joint_motion(qs):
    """Output-sample displacements, velocities and accelerations of 19 joints."""
    qs = np.asarray(qs)
    if len(qs) < 3:
        raise ValueError("Need at least 3 configurations")
    dq = np.array([
        pin.difference(model, qa, qb)[6:]
        for qa, qb in zip(qs[:-1], qs[1:])
    ])
    v = dq / DT
    acceleration = np.diff(v, axis=0) / DT
    return dq, v, acceleration


def check_velocity_limits(qs):
    """True when every 50 Hz actuated-joint displacement is within its limit."""
    qs = np.asarray(qs)
    if len(qs) < 2:
        return True
    v = np.array([
        pin.difference(model, qa, qb)[6:] / DT
        for qa, qb in zip(qs[:-1], qs[1:])
    ])
    return bool(np.all(np.abs(v) <= model.velocityLimit[6:] + 1e-9))
