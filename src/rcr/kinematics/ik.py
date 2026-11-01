
"""Kinematic trajectory generation for the climbing robot.

The moving ball follows a C2 piecewise Bezier trajectory through six
Cartesian waypoints, while the other three balls remain fixed.

Two interchangeable IK solvers:
    - ik_pinv: hierarchical pseudoinverse IK
    - ik_qp:   hierarchical constrained QP IK

Priorities:
    1. Keep supporting balls fixed.
    2. Track the moving ball.
    3. Prefer the hanging posture, joint-limit clearance,
       low joint motion, and smooth joint velocities.

Joint position and sampled velocity limits are imposed for both methods.
No dynamics or collision avoidance is included.
"""

import microban_model
import numpy as np
import pinocchio as pin
import quadprog
from numpy.linalg import norm, pinv

from rcr.kinematics.bezier import Bezier, connecting
from rcr.planner.astar import Q_HANG, initial_q, targets as stance_targets
from rcr.types import (
    LIMBS,
    MICROBAN_JOINTS,
    ContactPlan,
    Feasibility,
    Stance,
    Trajectory,
    WallModel,
)


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
LIMIT_GAIN = 0.02
VELOCITY_GAIN = 0.02
ACCEL_GAIN = 0.5

MAX_STEP_NORM = 0.1

RIDGE = 1e-6
SLACK_WEIGHT = 1e3

MAX_INNER_JOINT_STEP = 0.1
MAX_INNER_BASE_TRANSLATION = 0.05
MAX_INNER_BASE_ROTATION = 0.15


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

JOINTS = [
    model.idx_qs[model.getJointId(name)]
    for name in MICROBAN_JOINTS
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def stance_key(stance: Stance):
    return tuple(stance.contacts[limb] for limb in LIMBS)


def ball_positions(q):
    """World positions of all four balls in LIMBS order."""
    pin.framesForwardKinematics(model, data, q)

    return np.array([
        data.oMf[BALL[limb]].translation.copy()
        for limb in LIMBS
    ])


def joint_bounds(q0, dt=None):
    """Joint position bounds, optionally limited by one output timestep."""
    lower = model.lowerPositionLimit[7:].copy()
    upper = model.upperPositionLimit[7:].copy()

    if dt is not None:
        vmax = model.velocityLimit[6:]

        if np.any(vmax <= 0):
            raise ValueError("Nonpositive joint velocity limit")

        lower = np.maximum(lower, q0[7:] - vmax * dt)
        upper = np.minimum(upper, q0[7:] + vmax * dt)

    if np.any(lower > upper + 1e-10):
        raise ValueError("Empty joint position/velocity interval")

    return lower, upper


# ---------------------------------------------------------------------
# Cartesian ball trajectory
# ---------------------------------------------------------------------

def limb_traj(p_a, p_b):
    """C2 piecewise quintic Bezier trajectory through six waypoints.

    Returns B(t), with:
        B.velocity(t)
        B.acceleration(t)
        B.waypoints
        B.times
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

    segments = np.diff(points, axis=0)
    lengths = norm(segments, axis=1)

    if (
        not np.all(np.isfinite(points))
        or np.any(lengths <= 1e-10)
    ):
        raise ValueError("Degenerate limb trajectory")

    times = MOVE_TIME * np.r_[
        0.0,
        np.cumsum(lengths) / lengths.sum(),
    ]

    durations = np.diff(times)

    # Desired velocity at each waypoint.
    velocities = np.zeros_like(points)

    for i in range(1, len(points) - 1):
        tangent = (
            segments[i - 1] / lengths[i - 1]
            + segments[i] / lengths[i]
        )

        if norm(tangent) > 1e-10:
            speed = min(
                lengths[i - 1] / durations[i - 1],
                lengths[i] / durations[i],
            )

            velocities[i] = (
                0.5 * speed * tangent / norm(tangent)
            )

    # Shared accelerations at waypoints.
    accelerations = np.zeros_like(points)

    curves = []

    for i in range(len(durations)):
        start = (
            points[i],
            velocities[i],
            accelerations[i],
        )

        end = (
            points[i + 1],
            velocities[i + 1],
            accelerations[i + 1],
        )

        curves.append(
            Bezier(
                connecting(start, end, durations[i]),
                t_min=float(times[i]),
                t_max=float(times[i + 1]),
            )
        )

    # Precompute analytic derivative curves.
    velocity_curves = [c.derivative(1) for c in curves]
    acceleration_curves = [c.derivative(2) for c in curves]

    def evaluate(t, curves_to_evaluate):
        t = float(np.clip(t, 0.0, MOVE_TIME))

        i = min(
            np.searchsorted(times[1:], t, side="left"),
            len(curves) - 1,
        )

        return curves_to_evaluate[i](t)

    def B(t):
        return evaluate(t, curves)

    B.velocity = lambda t: evaluate(t, velocity_curves)
    B.acceleration = lambda t: evaluate(t, acceleration_curves)

    B.waypoints = points
    B.times = times

    return B


# ---------------------------------------------------------------------
# Shared IK tasks
# ---------------------------------------------------------------------

def errors_and_jacobians(q, targets):
    """Cartesian ball-position errors and Jacobians."""
    pin.computeJointJacobians(model, data, q)
    pin.updateFramePlacements(model, data)

    errors = {}
    jacobians = {}

    for limb in LIMBS:
        frame = BALL[limb]

        errors[limb] = (
            targets[limb] - data.oMf[frame].translation
        )

        jacobians[limb] = pin.getFrameJacobian(
            model,
            data,
            frame,
            pin.LOCAL_WORLD_ALIGNED,
        )[:3]

    return errors, jacobians


def soft_target(q, sample_q, previous_q=None, dt=None):
    """Lowest-priority joint preference shared by both solvers.

    sample_q: previous output configuration q[k-1].
    previous_q: configuration q[k-2].

    Acceleration regularisation encourages:

        q[k] - q[k-1] ~= q[k-1] - q[k-2]

    Only the actuated joints are regularised.
    """
    soft = np.zeros(model.nv)

    # Nominal hanging posture.
    soft[6:] = POSTURE_GAIN * (
        Q_HANG[7:] - q[7:]
    )

    # Prefer joint-limit midpoints.
    lower = model.lowerPositionLimit[7:]
    upper = model.upperPositionLimit[7:]

    finite = np.isfinite(lower) & np.isfinite(upper)

    midpoint = np.zeros_like(lower)
    midpoint[finite] = (
        lower[finite] + upper[finite]
    ) / 2.0

    soft[6:][finite] += LIMIT_GAIN * (
        midpoint[finite] - q[7:][finite]
    )

    if dt is not None:
        # Prefer smaller movement from previous output.
        soft[6:] += VELOCITY_GAIN * (
            sample_q[7:] - q[7:]
        )

        if previous_q is not None:
            # Previous commanded joint displacement.
            previous_step = pin.difference(
                model, previous_q, sample_q
            )[6:]

            # Constant-velocity prediction.
            predicted = sample_q[7:] + previous_step

            soft[6:] += ACCEL_GAIN * (
                predicted - q[7:]
            )

    return soft


# ---------------------------------------------------------------------
# Pseudoinverse IK
# ---------------------------------------------------------------------

def ik_pinv(q0, targets, moving=None, dt=None, previous_q=None):
    """Hierarchical pseudoinverse IK.

    Priority 1: supporting contacts.
    Priority 2: moving ball.
    Priority 3: joint regularisation.

    Position and sampled velocity limits are enforced by clipping.
    """
    q = q0.copy()
    held = [limb for limb in LIMBS if limb != moving]

    lower, upper = joint_bounds(q0, dt)

    for _ in range(MAX_ITERATIONS):
        errors, J = errors_and_jacobians(q, targets)

        if max(norm(e) for e in errors.values()) < EPSILON:
            return q

        # Priority 1: supporting contacts.
        Jc = np.vstack([J[limb] for limb in held])
        ec = np.concatenate([errors[limb] for limb in held])

        Jc_inv = pinv(Jc)

        dq = Jc_inv @ ec
        P = np.eye(model.nv) - Jc_inv @ Jc

        # Priority 2: moving ball.
        if moving is not None:
            Jm = J[moving]
            em = errors[moving]

            dq += pinv(Jm @ P) @ (
                em - Jm @ dq
            )

        # Priority 3: posture, joint limits and smooth velocity.
        J_all = np.vstack([
            J[limb] for limb in LIMBS
        ])

        P_all = (
            np.eye(model.nv)
            - pinv(J_all) @ J_all
        )

        dq += P_all @ soft_target(
            q, q0, previous_q, dt
        )

        # Numerical step limit.
        step = IK_STEP * dq

        if norm(step) > MAX_STEP_NORM:
            step *= MAX_STEP_NORM / norm(step)

        q = pin.integrate(model, q, step)

        # Clamp against the SAME output-sample bounds.
        q[7:] = np.clip(q[7:], lower, upper)

        if not np.all(np.isfinite(q)):
            raise ValueError(
                "Pseudoinverse IK produced non-finite configuration"
            )

    raise ValueError("Pseudoinverse IK did not converge")


# ---------------------------------------------------------------------
# QP helpers
# ---------------------------------------------------------------------

def solve_qp(G, a, E, f, lower, upper):
    """Solve:

        min 0.5*x.T@G@x - a.T@x

        subject to:
            E@x = f
            lower <= x <= upper
    """
    n = len(a)

    columns = [row.copy() for row in E]
    rhs = list(f)

    for j in range(n):
        if np.isfinite(lower[j]):
            col = np.zeros(n)
            col[j] = 1.0

            columns.append(col)
            rhs.append(lower[j])

        if np.isfinite(upper[j]):
            col = np.zeros(n)
            col[j] = -1.0

            columns.append(col)
            rhs.append(-upper[j])

    C = (
        np.column_stack(columns)
        if columns
        else np.empty((n, 0))
    )

    try:
        return quadprog.solve_qp(
            G,
            a,
            C,
            np.asarray(rhs),
            meq=len(f),
        )[0]

    except ValueError as exc:
        raise ValueError(
            f"QP solver failed: {exc}"
        ) from exc


def qp_step_bounds(q, sample_q, dt):
    """Limits on one numerical QP correction.

    Physical joint velocity bounds are relative to sample_q,
    not to the current numerical IK iterate.
    """
    lower = np.full(model.nv, -np.inf)
    upper = np.full(model.nv, np.inf)

    # Floating-base numerical trust region.
    lower[:3] = -MAX_INNER_BASE_TRANSLATION
    upper[:3] = MAX_INNER_BASE_TRANSLATION

    lower[3:6] = -MAX_INNER_BASE_ROTATION
    upper[3:6] = MAX_INNER_BASE_ROTATION

    # Actuated joint bounds.
    q_lower, q_upper = joint_bounds(sample_q, dt)

    lower[6:] = np.maximum(
        q_lower - q[7:],
        -MAX_INNER_JOINT_STEP,
    )

    upper[6:] = np.minimum(
        q_upper - q[7:],
        MAX_INNER_JOINT_STEP,
    )

    if np.any(lower > upper + 1e-9):
        raise ValueError("No feasible QP correction")

    return lower, upper


def task_qp(
    J,
    error,
    lower,
    upper,
    frozen_J=None,
    frozen_value=None,
):
    """Cartesian task QP with explicit slack.

    Higher-priority corrections can be frozen as equalities.
    """
    ne = len(error)
    n = model.nv + ne

    G = np.diag(
        np.r_[
            np.full(model.nv, RIDGE),
            np.full(ne, SLACK_WEIGHT),
        ]
    )

    a = np.zeros(n)

    # J*d + slack = error.
    E = np.hstack((J, np.eye(ne)))
    f = np.asarray(error)

    if frozen_J is not None:
        frozen = np.hstack((
            frozen_J,
            np.zeros((len(frozen_value), ne)),
        ))

        E = np.vstack((E, frozen))
        f = np.r_[f, frozen_value]

    lower_full = np.r_[
        lower,
        np.full(ne, -np.inf),
    ]

    upper_full = np.r_[
        upper,
        np.full(ne, np.inf),
    ]

    return solve_qp(
        G,
        a,
        E,
        f,
        lower_full,
        upper_full,
    )[:model.nv]


# ---------------------------------------------------------------------
# QP IK
# ---------------------------------------------------------------------

def ik_qp(q0, targets, moving=None, dt=None, previous_q=None):
    """Hierarchical constrained QP IK.

    Same tasks as ik_pinv, but with joint constraints built
    directly into the optimization.
    """
    q = q0.copy()
    held = [limb for limb in LIMBS if limb != moving]

    for _ in range(MAX_ITERATIONS):
        errors, J = errors_and_jacobians(q, targets)

        if max(norm(e) for e in errors.values()) < EPSILON:
            return q

        lower, upper = qp_step_bounds(q, q0, dt)

        Jc = np.vstack([
            J[limb] for limb in held
        ])

        ec = np.concatenate([
            errors[limb] for limb in held
        ])

        # Priority 1: supporting balls.
        dq = task_qp(
            Jc,
            IK_STEP * ec,
            lower,
            upper,
        )

        achieved = Jc @ dq

        # Priority 2: moving ball.
        if moving is not None:
            Jm = J[moving]

            dq = task_qp(
                Jm,
                IK_STEP * errors[moving],
                lower,
                upper,
                frozen_J=Jc,
                frozen_value=achieved,
            )

            E = np.vstack((Jc, Jm))

        else:
            E = Jc

        # Priority 3: posture and velocity smoothness.
        target_dq = dq + IK_STEP * soft_target(
            q,
            q0,
            previous_q,
            dt,
        )

        # Keep higher-priority Cartesian corrections unchanged.
        dq = solve_qp(
            np.eye(model.nv),
            target_dq,
            E,
            E @ dq,
            lower,
            upper,
        )

        q = pin.integrate(model, q, dq)

        if not np.all(np.isfinite(q)):
            raise ValueError(
                "QP IK produced non-finite configuration"
            )

    raise ValueError("QP IK did not converge")


# ---------------------------------------------------------------------
# Solver selection
# ---------------------------------------------------------------------

def solve_ik(
    q0,
    targets,
    moving=None,
    dt=None,
    method="qp",
    previous_q=None,
):
    """Solve IK using either pseudoinverse or QP."""
    if method == "pinv":
        return ik_pinv(
            q0,
            targets,
            moving=moving,
            dt=dt,
            previous_q=previous_q,
        )

    if method == "qp":
        return ik_qp(
            q0,
            targets,
            moving=moving,
            dt=dt,
            previous_q=previous_q,
        )

    raise ValueError(f"Unknown IK method: {method}")


# ---------------------------------------------------------------------
# Initial stance
# ---------------------------------------------------------------------

def start_pose(
    wall: WallModel,
    stance: Stance,
    method="qp",
):
    """Compute an initial configuration with all four balls fixed."""
    p = stance_targets(wall, stance_key(stance))

    targets = {
        limb: p[i].copy()
        for i, limb in enumerate(LIMBS)
    }

    q0 = initial_q(p)

    return solve_ik(
        q0,
        targets,
        moving=None,
        method=method,
    )


# ---------------------------------------------------------------------
# Single climbing movement
# ---------------------------------------------------------------------

def ik_traj(
    wall: WallModel,
    a: Stance,
    b: Stance,
    q0,
    method="qp",
    previous_q=None,
):
    """Generate IK configurations at 50 Hz for one climbing movement."""
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

    # Assume rest at the first move unless a previous sample is supplied.
    prev = (
        q.copy()
        if previous_q is None
        else previous_q.copy()
    )

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

        q_next = solve_ik(
            q,
            targets,
            moving=moving,
            dt=DT,
            method=method,
            previous_q=prev,
        )

        qs.append(q_next.copy())

        prev = q.copy()
        q = q_next

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
    """Preliminary trajectory feasibility from nominal starting stance."""
    try:
        q0 = start_pose(wall, a, method=method)
        qs = ik_traj(
            wall,
            a,
            b,
            q0,
            method=method,
        )

        if not np.all(np.isfinite(qs)):
            return Feasibility(
                False, reason="Non-finite trajectory"
            )

        if np.any(
            qs[:, 7:] < model.lowerPositionLimit[7:] - 1e-8
        ):
            return Feasibility(
                False, reason="Lower joint limit violated"
            )

        if np.any(
            qs[:, 7:] > model.upperPositionLimit[7:] + 1e-8
        ):
            return Feasibility(
                False, reason="Upper joint limit violated"
            )

        if not check_velocity_limits(qs):
            return Feasibility(
                False, reason="Joint velocity limits exceeded"
            )

        return Feasibility(True)

    except (ValueError, np.linalg.LinAlgError) as exc:
        return Feasibility(False, reason=str(exc))


# ---------------------------------------------------------------------
# Complete trajectory
# ---------------------------------------------------------------------

def generate_trajectory(
    wall: WallModel,
    plan: ContactPlan,
    method="qp",
    q0=None,
):
    """Chain climbing moves, preserving previous joint velocity."""
    stances = plan.stances

    if not stances:
        raise ValueError("Empty contact plan")

    q = (
        start_pose(wall, stances[0], method=method)
        if q0 is None
        else q0.copy()
    )

    prev = q.copy()
    qs = [q.copy()]

    for a, b in zip(stances[:-1], stances[1:]):
        segment = ik_traj(
            wall,
            a,
            b,
            q,
            method=method,
            previous_q=prev,
        )

        qs.extend(segment[1:])

        prev = segment[-2].copy()
        q = segment[-1].copy()

    return np.asarray(qs)


# ---------------------------------------------------------------------
# Robot trajectory format
# ---------------------------------------------------------------------

def to_trajectory(qs: np.ndarray) -> Trajectory:
    """Convert Pinocchio configurations into robot commands."""
    times = [
        round(k * DT, 9)
        for k in range(len(qs))
    ]

    positions = [
        [float(q[j]) for j in JOINTS]
        for q in qs
    ]

    # Pinocchio quaternion: (x, y, z, w).
    # Trajectory quaternion: (w, x, y, z).
    trunk_poses = [
        [
            float(v)
            for v in q[[0, 1, 2, 6, 3, 4, 5]]
        ]
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
    """Return joint displacements, velocities and accelerations."""
    qs = np.asarray(qs)

    if len(qs) < 3:
        raise ValueError("Need at least three configurations")

    displacements = np.array([
        pin.difference(model, qa, qb)[6:]
        for qa, qb in zip(qs[:-1], qs[1:])
    ])

    velocities = displacements / DT
    accelerations = np.diff(velocities, axis=0) / DT

    return displacements, velocities, accelerations


def check_velocity_limits(qs):
    """Check actuated joint velocity limits at 50 Hz."""
    qs = np.asarray(qs)

    if len(qs) < 2:
        return True

    velocities = np.array([
        pin.difference(model, qa, qb)[6:] / DT
        for qa, qb in zip(qs[:-1], qs[1:])
    ])

    return bool(
        np.all(
            np.abs(velocities)
            <= model.velocityLimit[6:] + 1e-9
        )
    )
