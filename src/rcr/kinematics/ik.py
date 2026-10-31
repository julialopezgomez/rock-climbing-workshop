
"""Kinematic trajectory generation for the climbing robot.

The moving limb follows a piecewise-linear task-space path.
The other three limbs stay in their holds.

Inverse kinematics uses hierarchical Jacobian pseudoinverses,
following the Advanced Robotics approach:

1. Keep supporting limbs fixed.
2. Track the moving limb.
3. Stay close to the nominal hanging posture.

No dynamics or collision avoidance is included in the IK solver.
"""

import microban_model
import numpy as np
import pinocchio as pin
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


# ---------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------

DT = 0.02
MOVE_TIME = 2.0

ABOVE = 0.025
OUT = 0.04

EPSILON = 1e-3
MAX_ITERATIONS = 200
IK_STEP = 0.2
POSTURE_GAIN = 0.1

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
    """Piecewise-linear ball trajectory from socket a to socket b.

    The ball moves up, away from the wall, across, inward, and down.
    Returns B(t), defined for t in [0, MOVE_TIME].
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

    lengths = norm(np.diff(points, axis=0), axis=1)

    if lengths.sum() == 0 or np.any(lengths == 0):
        raise ValueError("Degenerate limb trajectory")

    times = MOVE_TIME * np.r_[
        0.0,
        np.cumsum(lengths) / lengths.sum(),
    ]

    def B(t):
        t = np.clip(t, 0.0, MOVE_TIME)

        i = min(
            np.searchsorted(times[1:], t),
            len(points) - 2,
        )

        s = (t - times[i]) / (times[i + 1] - times[i])

        return (1 - s) * points[i] + s * points[i + 1]

    return B


# ---------------------------------------------------------------------
# Inverse kinematics
# ---------------------------------------------------------------------

def solve_ik(q0, targets, moving=None):
    """Solve IK from configuration q0.

    targets: dict mapping limb names to desired XYZ positions.

    moving:
        None  -> all four limbs have equal priority (stance IK).
        limb  -> other three limbs first, moving limb second.

    Nominal hanging joint posture is the lowest-priority task.
    """
    q = q0.copy()

    held = [
        limb for limb in LIMBS
        if limb != moving
    ]

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

        posture_error = np.zeros(model.nv)
        posture_error[6:] = Q_HANG[7:] - q[7:]

        dq += POSTURE_GAIN * P_all @ posture_error

        # Bounded integration step
        step = IK_STEP * dq
        step_norm = norm(step)

        if step_norm > MAX_STEP_NORM:
            step *= MAX_STEP_NORM / step_norm

        q = pin.integrate(model, q, step)
        q = project_to_joint_limits(q)

        if not np.all(np.isfinite(q)):
            raise ValueError("IK produced non-finite configuration")

    raise ValueError("IK did not converge")


# ---------------------------------------------------------------------
# Initial stance
# ---------------------------------------------------------------------

def start_pose(wall: WallModel, stance: Stance):
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

    return solve_ik(q0, targets, moving=None)


# ---------------------------------------------------------------------
# Single movement
# ---------------------------------------------------------------------

def ik_traj(wall: WallModel, a: Stance, b: Stance, q0):
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

        q = solve_ik(q, targets, moving)

        qs.append(q.copy())

    return np.asarray(qs)


# ---------------------------------------------------------------------
# Feasibility
# ---------------------------------------------------------------------

def feasible_ik_traj(
    wall: WallModel,
    a: Stance,
    b: Stance,
) -> Feasibility:
    """Preliminary feasibility check from a nominal starting stance.

    Actual chained trajectories must be validated separately.
    """
    try:
        q0 = start_pose(wall, a)
        qs = ik_traj(wall, a, b, q0)

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

        return Feasibility(True)

    except (ValueError, np.linalg.LinAlgError) as e:
        return Feasibility(False, reason=str(e))


# ---------------------------------------------------------------------
# Complete climbing trajectory
# ---------------------------------------------------------------------

def generate_trajectory(wall: WallModel, plan: ContactPlan):
    """Chain all movements without resetting configurations."""
    stances = plan.stances

    if not stances:
        raise ValueError("Contact plan is empty")

    q = start_pose(wall, stances[0])
    qs = [q.copy()]

    for a, b in zip(stances[:-1], stances[1:]):
        segment = ik_traj(wall, a, b, q)

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
