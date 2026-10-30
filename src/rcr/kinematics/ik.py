"""Kinematic motion of a move: the moving ball follows a polynomial path from its socket on hold a
to its socket on hold b, the other three stay in theirs, and the whole body follows by hierarchical
IK with an error correction (CLIK) [Ours, after the planner's astar.ik]. No torques, no collisions.
"""

import json
from math import comb
from pathlib import Path

import microban_model
import numpy as np
import pinocchio as pin
from numpy.linalg import norm, pinv

from rcr.types import LIMBS, Stance, WallModel

SOCKET = np.array([0.0, -0.026, -0.014])  # ball centre from Hold.position [simulator wall.py]
HANG = Path(__file__).parents[3] / "third_party/simulator/climbing_sim/assets/poses/hang.json"
TASK_ORDER = ("right_hand", "left_hand", "right_foot", "left_foot")
# trunk z up, x into the wall (+y) [simulator README, Frames]
TRUNK_UPRIGHT = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
EPSILON = 1e-3  # [m] a ball this close to its target has reached it
MAX_ITERATIONS = 200  # for the start pose
STEP = 0.5  # fraction of the error the start pose's IK closes per iteration
# The ball's path [Ours, to tune]: out of socket a straight off the wall, up and away from it to a
# bit above socket b, then down into it.
OUT = 0.04  # [m] away from the wall (-y), at both ends
ABOVE = 0.05  # [m] above socket b, before going in
MOVE_TIME = 2.0  # [s] per move
DT = 0.02  # [s] between samples: the robot's 50 Hz loop
GAIN = 0.5  # fraction of a ball's tracking error corrected per sample [Ours]
LOCK_PASSES = 4  # ik_step: solves at most, each locking the joints it would push past a limit
MAX_ERROR = 0.01  # [m] a held ball further than this from its socket: the move cannot be done
DAMPING = 0.03  # damped(): the moving ball's and the trunk's tasks [Ours]
ITERATIONS = 3  # IK steps per sample, toward that sample's targets [Ours]

# trunk free: q = (x, y, z, qx, qy, qz, qw, 19 joints)
model = pin.buildModelFromUrdf(str(microban_model.urdf(climb=True)), pin.JointModelFreeFlyer())
data = model.createData()
BALL = {limb: model.getFrameId(f"{limb}_ball") for limb in LIMBS}

Q_HANG = pin.neutral(model)
for name, angle in json.loads(HANG.read_text())["joints"].items():
    Q_HANG[model.idx_qs[model.getJointId(name)]] = angle
pin.framesForwardKinematics(model, data, Q_HANG)
BALLS_HANG = np.array([data.oMf[BALL[limb]].translation for limb in LIMBS])


def sockets(wall: WallModel, stance: Stance) -> np.ndarray:
    """Ball centre in each limb's hold, in LIMBS order."""
    position = {h.id: h.position for h in wall.holds}
    return np.array([position[stance.contacts[limb]] for limb in LIMBS]) + SOCKET


def balls(q) -> np.ndarray:
    """Ball centres at q, in LIMBS order."""
    pin.framesForwardKinematics(model, data, q)
    return np.array([data.oMf[BALL[limb]].translation for limb in LIMBS])


def damped(A):
    """Damped least-squares inverse: like pinv, but the step stays bounded where A is near
    singular (a limb stretched out), so a target out of reach is only approached."""
    return A.T @ np.linalg.inv(A @ A.T + DAMPING**2 * np.eye(A.shape[0]))


def ik_step(q, displacement, moving=None):
    """One hierarchical IK step: each limb's ball moved by its displacement (in LIMBS order), the
    held ones first in TASK_ORDER, then the moving one, each in the null space of the ones before;
    last, the trunk turned upright and facing the wall as far as the balls allow.
    - The held balls are solved exactly (their sockets are always in reach); the moving ball and
      the trunk with `damped`, so the moving ball gets as close as it can to a target out of reach
      while the held balls stay in their sockets [Ours]. With pinv for all and a fixed order, a
      long hand move with OUT 0.05 / ABOVE 0.10 diverged (NaN), or pulled a held hand 6 mm out.
    - A joint the step would push past a limit is locked there and the step solved again without
      it [Ours]: with clipping alone (the planner's), a held ball drifted up to 10 mm off its
      socket on a level-2 climb (seed 5226)."""
    pin.computeJointJacobians(model, data, q)
    pin.updateFramePlacements(model, data)
    order = [limb for limb in TASK_ORDER if limb != moving] + ([moving] if moving else [])
    tasks = [
        (
            pin.getFrameJacobian(model, data, BALL[limb], pin.LOCAL_WORLD_ALIGNED)[:3],
            LIMBS.index(limb),
            limb == moving,
        )
        for limb in order
    ]
    trunk = pin.getJointJacobian(model, data, 1, pin.LOCAL_WORLD_ALIGNED)[3:]
    turn = GAIN * pin.log3(TRUNK_UPRIGHT @ data.oMi[1].rotation.T)
    free = np.ones(model.nv)  # 0: a locked joint (its column out of every Jacobian)
    for _ in range(LOCK_PASSES):
        vq, P = np.zeros(model.nv), np.eye(model.nv)
        for J, i, is_moving in tasks:
            JP = J * free @ P
            vq = vq + (damped(JP) if is_moving else pinv(JP)) @ (displacement[i] - J * free @ vq)
            P = P - pinv(JP) @ JP  # the exact null space, also under a damped task
        vq = vq + damped(trunk * free @ P) @ (turn - trunk * free @ vq)
        new = pin.integrate(model, q, vq)
        past = (new > model.upperPositionLimit) | (new < model.lowerPositionLimit)
        past = past[7:]  # the joints' (the free flyer has no limits)
        if not past.any():
            break
        free[6:][past] = 0
    return projecttojointlimits(new)


def projecttojointlimits(q):
    return np.minimum(np.maximum(model.lowerPositionLimit, q), model.upperPositionLimit)


def start_pose(wall: WallModel, stance: Stance):
    """The hang posture with its trunk fitted rigidly to the sockets [simulator sim.py
    reset_hanging], then IK until every ball is within EPSILON of its socket. ValueError if not."""
    t = sockets(wall, stance)
    cp, cs = BALLS_HANG.mean(0), t.mean(0)
    U, _, Vt = np.linalg.svd((BALLS_HANG - cp).T @ (t - cs))
    R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    q = Q_HANG.copy()
    q[0:3] = cs - R @ cp
    q[3:7] = pin.Quaternion(R).coeffs()
    for _ in range(MAX_ITERATIONS):
        error = t - balls(q)
        if norm(error, axis=1).max() < EPSILON:
            return q
        q = ik_step(q, STEP * error)
    raise ValueError(f"start pose: balls {norm(error, axis=1).max() * 1000:.0f} mm from sockets")


def ball_path(p_a, p_b):
    """B(s), s in [0, 1]: a Bezier from p_a to p_b, of degree len(P) - 1 (Bernstein weights, so
    any number of control points). The control points are p_a, p_b + away, p_b + away + up,
    p_b + up, p_b, with away = (0, -OUT, 0) and up = (0, 0, ABOVE): it leaves p_a towards the
    point in front of socket b, and arrives at p_b straight down from above it.
    """
    away, up = np.array([0.0, -OUT, 0.0]), np.array([0.0, 0.0, ABOVE])
    P = np.array([p_a, p_b + away, p_b + away + up, p_b + up, p_b])
    n = len(P) - 1
    weights = [comb(n, k) for k in range(n + 1)]

    def B(s):
        return sum(weights[k] * s**k * (1 - s) ** (n - k) * P[k] for k in range(n + 1))

    return B


def timing(t):
    """s(t): min-jerk, from 0 to 1 over MOVE_TIME, zero velocity and acceleration at both ends."""
    tau = min(max(t / MOVE_TIME, 0.0), 1.0)
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def moving_limb(a: Stance, b: Stance) -> str:
    (limb,) = [limb for limb in LIMBS if a.contacts[limb] != b.contacts[limb]]
    return limb


def track_move(wall: WallModel, a: Stance, b: Stance, q0):
    """The configurations of the move a -> b, one every DT from q0 (excluded) to the end: at each
    sample, ITERATIONS IK steps closing GAIN of each ball's error to its target (the moving ball's
    on ball_path, the others' their sockets). Where the path is out of the limb's reach, the
    moving ball only gets as close as it can (ik_step). Also the moving ball's worst distance from
    its path [m]. ValueError, naming the move, if a held ball gets MAX_ERROR from its socket."""
    limb = moving_limb(a, b)
    i = LIMBS.index(limb)
    held = sockets(wall, a)
    B = ball_path(held[i], sockets(wall, b)[i])
    q, worst, out = q0, 0.0, []
    for k in range(1, round(MOVE_TIME / DT) + 1):
        target = held.copy()
        target[i] = B(timing(k * DT))
        for _ in range(ITERATIONS):
            q = ik_step(q, GAIN * (target - balls(q)), moving=limb)
        error = norm(target - balls(q), axis=1)
        if not np.delete(error, i).max() <= MAX_ERROR:  # also NaN: the IK diverged
            move = f"{limb} {a.contacts[limb]} -> {b.contacts[limb]}"
            e = np.delete(error, i).max() * 1000
            raise ValueError(f"{move}: a held ball {e:.0f} mm from its socket at {k * DT:.2f} s")
        worst = max(worst, error[i])
        out.append(q)
    return out, worst
