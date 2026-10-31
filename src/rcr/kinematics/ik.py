"""Kinematic motion of a move: the moving limb's ball follows limb_traj from its socket on hold a to
its socket on hold b, the other three stay in theirs, and the whole body follows by IK [Ours].
Starting over from ik_old.py.
"""

import microban_model
import numpy as np
import pinocchio as pin
from numpy.linalg import norm, pinv

from rcr.planner.astar import Q_HANG
from rcr.planner.astar import ik as ik_astar
from rcr.planner.astar import targets as stance_targets
from rcr.types import (
    LIMBS,
    MICROBAN_JOINTS,
    ContactPlan,
    Feasibility,
    Stance,
    Trajectory,
    WallModel,
)

DT = 0.02  # [s] between samples: the robot's 50 Hz loop
MOVE_TIME = 2.0
ABOVE = 0.025  # metres
OUT = 0.04  # metres
EPSILON = 1e-4
MAX_ITERATIONS = 100
IK_STEP = 0.2
# The soft task (solve_ik's priority 3) [Ours]: weights of the trunk's lean and of the joints'
# gap to the hang pose, and the damping of its inverse
TRUNK_WEIGHT = 1.0
POSTURE_WEIGHT = 0.1
SOFT_GAIN = 0.1  # fraction of its gap the soft task closes per IK iteration
DAMPING = 0.03

# trunk free: q = (x, y, z, qx, qy, qz, qw, 19 joints)
model = pin.buildModelFromUrdf(str(microban_model.urdf(climb=True)), pin.JointModelFreeFlyer())
data = model.createData()
BALL = {limb: model.getFrameId(f"{limb}_ball") for limb in LIMBS}
# q's index of each joint, in MICROBAN_JOINTS order (the model's own order differs)
JOINTS = [model.idx_qs[model.getJointId(name)] for name in MICROBAN_JOINTS]


def limb_traj(p_a, p_b):
    """Piecewise-linear end-effector path B(t)."""
    up = np.array([0.0, 0.0, ABOVE])
    out = np.array([0.0, -OUT, 0.0])

    points = np.array(
        [
            p_a,
            p_a + up,
            p_a + up + out,
            p_b + up + out,
            p_b + up,
            p_b,
        ]
    )

    lengths = norm(np.diff(points, axis=0), axis=1)
    times = MOVE_TIME * np.r_[0.0, np.cumsum(lengths) / lengths.sum()]

    def B(t):
        t = np.clip(t, 0.0, MOVE_TIME)
        i = min(np.searchsorted(times[1:], t), len(points) - 2)
        s = (t - times[i]) / (times[i + 1] - times[i])
        return (1 - s) * points[i] + s * points[i + 1]

    return B


def project_to_joint_limits(q):
    """Clamp actuated joints without modifying the free flyer."""
    q_proj = q.copy()
    q_proj[7:] = np.clip(q_proj[7:], model.lowerPositionLimit[7:], model.upperPositionLimit[7:])
    return q_proj


def damped(A):
    """Damped least-squares inverse: like pinv, but bounded where A is near singular."""
    return A.T @ np.linalg.inv(A @ A.T + DAMPING**2 * np.eye(A.shape[0]))


def solve_ik(q0, targets, moving):
    """IK from q0 to four ball position targets.

    Supporting limbs have priority over the moving limb.
    """
    q = q0.copy()
    held = [limb for limb in LIMBS if limb != moving]

    for _ in range(MAX_ITERATIONS):
        pin.computeJointJacobians(model, data, q)
        pin.updateFramePlacements(model, data)

        errors = {}
        jacobians = {}

        for limb in LIMBS:
            frame = BALL[limb]
            position = data.oMf[frame].translation
            errors[limb] = targets[limb] - position

            jacobians[limb] = pin.getFrameJacobian(model, data, frame, pin.LOCAL_WORLD_ALIGNED)[:3]

        if max(norm(e) for e in errors.values()) < EPSILON:
            return q

        # Priority 1: supporting contacts
        Jc = np.vstack([jacobians[limb] for limb in held])
        ec = np.concatenate([errors[limb] for limb in held])

        Jc_inv = pinv(Jc)
        dq_c = Jc_inv @ ec
        P = np.eye(model.nv) - Jc_inv @ Jc

        # Priority 2: moving end-effector
        Jm = jacobians[moving]
        em = errors[moving]

        dq = dq_c + pinv(Jm @ P) @ (em - Jm @ dq_c)
        P = P - pinv(Jm @ P) @ (Jm @ P)

        # Priority 3 (soft): trunk not leaning (its x into the wall, +y; only turning about y
        # allowed) and the joints toward the hang pose, weighted, in the null space of the balls
        R = data.oMi[1].rotation
        lean = np.cross(R[:, 0], [0.0, 1.0, 0.0])[[0, 2]]  # its rotation about x and z
        Jt = pin.getJointJacobian(model, data, 1, pin.LOCAL_WORLD_ALIGNED)[[3, 5]]
        posture = pin.difference(model, q, Q_HANG)[6:]
        A = np.vstack([TRUNK_WEIGHT * Jt, POSTURE_WEIGHT * np.eye(model.nv)[6:]])
        r = SOFT_GAIN * np.concatenate([TRUNK_WEIGHT * lean, POSTURE_WEIGHT * posture])
        dq = dq + damped(A @ P) @ (r - A @ dq)

        q = pin.integrate(model, q, IK_STEP * dq)
        q = project_to_joint_limits(q)

    raise ValueError("IK did not converge")


def start_pose(wall, stance):
    k = tuple(stance.contacts[limb] for limb in LIMBS)
    verdict, q, _ = ik_astar(wall, k)

    if not verdict.feasible:
        raise ValueError(verdict.reason)
    return q


def ik_traj(wall, a, b, q0):
    """Generate q(t) by solving IK at every sample."""

    moving = [limb for limb in LIMBS if a.contacts[limb] != b.contacts[limb]]

    if len(moving) != 1:
        raise ValueError("Exactly one limb must move")

    moving = moving[0]

    k_a = tuple(a.contacts[limb] for limb in LIMBS)
    k_b = tuple(b.contacts[limb] for limb in LIMBS)

    p_a = stance_targets(wall, k_a)
    p_b = stance_targets(wall, k_b)

    B = limb_traj(p_a[LIMBS.index(moving)], p_b[LIMBS.index(moving)])

    q = q0.copy()
    qs = [q.copy()]

    times = np.linspace(0, MOVE_TIME, round(MOVE_TIME / DT) + 1)

    for t in times[1:]:
        targets = {limb: p_a[i].copy() for i, limb in enumerate(LIMBS)}

        targets[moving] = B(t)

        # Warm-start IK from the previous configuration
        q = solve_ik(q, targets, moving)

        qs.append(q.copy())

    return np.array(qs)


def feasible_ik_traj(wall: WallModel, a: Stance, b: Stance) -> Feasibility:
    """Check whether IK can generate a trajectory from stance a to b."""
    try:
        q0 = start_pose(wall, a)
        qs = ik_traj(wall, a, b, q0)

        # Check joint limits
        for q in qs:
            if np.any(q[7:] < model.lowerPositionLimit[7:]) or np.any(
                q[7:] > model.upperPositionLimit[7:]
            ):
                return Feasibility(False, reason="Joint limits violated")

        return Feasibility(True)

    except (ValueError, np.linalg.LinAlgError) as e:
        return Feasibility(False, reason=str(e))


def generate_trajectory(wall: WallModel, plan: ContactPlan):
    """Generate a continuous configuration trajectory for a contact plan."""

    stances = plan.stances
    q = start_pose(wall, stances[0])

    qs = [q.copy()]

    for a, b in zip(stances[:-1], stances[1:], strict=True):
        segment = ik_traj(wall, a, b, q)

        # The first configuration is already in qs.
        qs.extend(segment[1:])

        # Warm-start the next movement from this endpoint.
        q = segment[-1].copy()

    return np.asarray(qs)


def to_trajectory(qs: np.ndarray) -> Trajectory:
    """Convert a sequence of configurations to a Trajectory object."""
    times = [round(k * DT, 9) for k in range(len(qs))]
    positions = [[float(q[j]) for j in JOINTS] for q in qs]  # joints, in MICROBAN_JOINTS order
    # Pinocchio's quaternion is (x, y, z, w); ours (w, x, y, z)
    trunk_poses = [[float(v) for v in q[[0, 1, 2, 6, 3, 4, 5]]] for q in qs]

    return Trajectory(
        joint_names=list(MICROBAN_JOINTS),
        times=times,
        positions=positions,
        planned_positions=[row.copy() for row in positions],
        trunk_poses=trunk_poses,
    )
