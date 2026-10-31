"""Kinematic motion of a move: the moving ball follows a polynomial path from its socket on hold a
to its socket on hold b, the other three stay in theirs, and the whole body follows by hierarchical
IK with an error correction (CLIK) [Ours, after the planner's astar.ik]. No torques, no collisions.
"""

import json
from pathlib import Path

import microban_model
import numpy as np
import pinocchio as pin
import quadprog
from numpy.linalg import norm, pinv

from rcr.kinematics.bezier import Bezier, connecting
from rcr.types import LIMBS, Feasibility, Stance, WallModel

SOCKET = np.array([0.0, -0.026, -0.014])  # ball centre from Hold.position [simulator wall.py]
HANG = Path(__file__).parents[3] / "third_party/simulator/climbing_sim/assets/poses/hang.json"
TASK_ORDER = ("right_hand", "left_hand", "right_foot", "left_foot")
# trunk z up, x into the wall (+y) [simulator README, Frames]
TRUNK_UPRIGHT = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
EPSILON = 1e-3  # [m] a ball this close to its target has reached it (1e-4: 6 % of start poses fail)
MAX_ITERATIONS = 200  # for the start pose
STEP = 0.5  # fraction of the error the start pose's IK closes per iteration
# The ball's path [Ours]: a corridor clear of every hold (ball_path). Measured on the simulator's
# collision shapes (ball radius 12.5 mm, holds 43 mm out of the wall): the 440 hold -> hold paths
# of 1113 moves on 6 walls clear every hold by 6.3 mm, and so does the ball as the IK tracks it.
# A single curve over cut the corner into the next hold (down to -14 mm), whatever its parameters.
# [m] straight up out of socket a before moving out (a ball leaves a socket only within ~10 deg of
# vertical [simulator wall.py, EXIT_BLOCK]); also above socket b, before coming down into it.
# Moving out at 25 mm, the ball clears its own hold by 7.9 mm (20 mm: 2.9), and a hold 60 mm
# above and 15 mm beside (the routes allow one there) by 8.5 mm (30 mm: 3.8).
ABOVE = 0.025
# [m] in front of the sockets while going across (66 mm from the wall): 10.5 mm in front of every
# hold (35 mm: 5.5)
OUT = 0.04
CORNER_TIME = 0.3  # [s] each of up, out, in and down; across takes the rest of MOVE_TIME
MOVE_TIME = 2.0  # [s] per move
DT = 0.02  # [s] between samples: the robot's 50 Hz loop
GAIN = 0.5  # fraction of a ball's tracking error corrected per sample [Ours]
LOCK_PASSES = 4  # ik_step: solves at most, each locking the joints it would push past a limit
MAX_ERROR = 0.01  # [m] a held ball further than this from its socket: the move cannot be done
ITERATIONS = 10  # IK steps per sample, toward that sample's targets [Ours]
# [m] the trunk's origin above the mean height of the four balls' targets [Ours]: the hang pose
# fitted to the start stance has it 7 mm below; raised 2 cm, the balls still reach their sockets
# with no joint at a limit (3 cm puts two at one)
TRUNK_HEIGHT = 0.015
# The stance pose [Ours]: every move starts and ends in its stances' (start_pose), so a move checked
# alone (feasible_ik_traj) is the move optimise makes. The joints are pulled toward the hang pose's,
# last and lowest, also during the move: 0.1 ended 195 of 243 level-2/3 moves already within
# JOINT_TOL of b's (0.3: 186; without, none).
POSTURE_GAIN = 0.1  # fraction of each joint's gap to the posture closed per IK step
SETTLED = 1e-5  # an IK step this small (norm of the change of q): q has stopped moving
JOINT_TOL = 0.01  # [rad] a move ending this close to b's stance pose jumps to it; further, blends
BLEND_TIME = 0.3  # [s] min-jerk from where a move ended to b's stance pose, balls held, at least
# The QP IK (ik_step_qp) [Ours]: its tasks' weights, and how far inside its limits it tries to stay
HELD_WEIGHT = 10  # a held ball's task, against the moving ball's 1: the held stay in their sockets
JOINT_MARGIN = 0.1  # [rad] the joints try to stay this far inside their limits (slack)
SPEED_MARGIN = 0.8  # the joints try to stay under this fraction of their speed limits (slack)
SLACK_WEIGHT = 10  # on the squared slack [rad^2]: inside the margins only if the balls need it
MANIPULABILITY_GAIN = 1e-5  # on the gradient of the moving limb's log-manipulability

# trunk free: q = (x, y, z, qx, qy, qz, qw, 19 joints)
model = pin.buildModelFromUrdf(str(microban_model.urdf(climb=True)), pin.JointModelFreeFlyer())
data = model.createData()
BALL = {limb: model.getFrameId(f"{limb}_ball") for limb in LIMBS}
SPEED_LIMIT = model.velocityLimit[6:]  # [rad/s] the joints' (6.6) [microban model URDF]


def chain(limb):
    """The velocity indices of the limb's joints, trunk to ball."""
    j, out = model.frames[BALL[limb]].parentJoint, []
    while j > 1:
        out.insert(0, model.idx_vs[j])
        j = model.parents[j]
    return out


CHAIN = {limb: chain(limb) for limb in LIMBS}

Q_HANG = pin.neutral(model)
for name, angle in json.loads(HANG.read_text())["joints"].items():
    Q_HANG[model.idx_qs[model.getJointId(name)]] = angle
pin.framesForwardKinematics(model, data, Q_HANG)
BALLS_HANG = np.array([data.oMf[BALL[limb]].translation for limb in LIMBS])


def sockets(wall: WallModel, stance: Stance) -> np.ndarray:
    """Ball centre in each limb's hold, in LIMBS order."""
    position = {h.id: h.position for h in wall.holds}
    return np.array([position[stance.contacts[limb]] for limb in LIMBS]) + SOCKET


def key(stance: Stance) -> tuple:
    return tuple(stance.contacts[limb] for limb in LIMBS)


def balls(q) -> np.ndarray:
    """Ball centres at q, in LIMBS order."""
    pin.framesForwardKinematics(model, data, q)
    return np.array([data.oMf[BALL[limb]].translation for limb in LIMBS])


# def damped(A):
#     """Damped least-squares inverse: like pinv, but the step stays bounded where A is near
#     singular (a limb stretched out), so a target out of reach is only approached."""
#     return A.T @ np.linalg.inv(A @ A.T + DAMPING**2 * np.eye(A.shape[0]))


def quadprog_solve_qp(H, q, G=None, h=None, C=None, d=None, verbose=False):
    """
    min (1/2)x' P x + q' x
    subject to  G x <= h
    subject to  C x  = d
    """
    qp_G = 0.5 * (H + H.T)  # make sure P is symmetric
    qp_a = -q
    qp_C = None
    qp_b = None
    meq = 0
    if C is not None:
        if G is not None:
            qp_C = -np.vstack([C, G]).T
            qp_b = -np.hstack([d, h])
        else:
            qp_C = -C.transpose()
            qp_b = -d
        meq = C.shape[0]
    elif G is not None:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
    res = quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)
    if verbose:
        return res
    return res[0]


def ik_step(q, displacement, moving=None, height=None, posture=None):
    ik_step_qp(q, displacement, moving, height, posture)
    # try:
    #     return ik_step_qp(q, displacement, moving)
    # except NotImplementedError:
    #     return ik_step_pinv(q, displacement, moving, height, posture)


def manipulability(q, limb):
    """Yoshikawa's w = sqrt(det(J J^T)) of the limb's ball over the limb's own joints.
    -- high w: the limb can move its ball in any direction, so it can push downwards on a hold.
    -- low w: the limb is stretched out, so it can only move its ball in"""
    pin.computeJointJacobians(model, data, q)
    pin.updateFramePlacements(model, data)
    J = pin.getFrameJacobian(model, data, BALL[limb], pin.LOCAL_WORLD_ALIGNED)[:3, CHAIN[limb]]
    return np.sqrt(max(np.linalg.det(J @ J.T), 1e-30))


def manipulability_gradient(q, limb, h=1e-6):
    """d log(w) / dq over the limb's joints (forward differences), 0 elsewhere."""
    gradient, w = np.zeros(model.nv), np.log(manipulability(q, limb))
    for v in CHAIN[limb]:
        dq = np.zeros(model.nv)
        dq[v] = h
        gradient[v] = (np.log(manipulability(pin.integrate(model, q, dq), limb)) - w) / h
    return gradient


def ik_step_qp(q, displacement, moving=None, height=None, posture=None):
    """
    moving: the limb whose ball is moved by its displacement; the others stay in their sockets."""
    nv, nj = model.nv, model.nv - 6 #nj is the number of joints (the free flyer has no limits)
    gradient = manipulability_gradient(q, moving) if moving else np.zeros(nv)
    pin.computeJointJacobians(model, data, q)
    pin.updateFramePlacements(model, data)
    H, c = 1e-6 * np.eye(nv), np.zeros(nv)  # cost: v' H v - 2 c' v
    for limb in TASK_ORDER:
        J = pin.getFrameJacobian(model, data, BALL[limb], pin.LOCAL_WORLD_ALIGNED)[:3]
        weight = 1.0 if limb == moving else HELD_WEIGHT
        H += weight * J.T @ J
        c += weight * J.T @ displacement[LIMBS.index(limb)]
    # x = (v, s_q, s_v): the quadprog helper's (1/2) x' P x + p' x
    zero = np.zeros((nv, 2 * nj))
    SLACK_WEIGHT = 0  
    P = 2 * np.block([[H, zero], [zero.T, SLACK_WEIGHT * np.eye(2 * nj)]])
    p = np.concatenate([-2 * c, np.zeros(2 * nj)])
    S = np.hstack([np.zeros((nj, 6)), np.eye(nj)])  # the joints' part of v
    Z, E = np.zeros((nj, nj)), np.eye(nj)
    upper, lower = model.upperPositionLimit[7:] - q[7:], q[7:] - model.lowerPositionLimit[7:]
    speed = SPEED_LIMIT * DT / ITERATIONS
    rows = [  # (G, h) of G x <= h
        (np.hstack([S, Z, Z]), upper),  # joint limits
        (np.hstack([-S, Z, Z]), lower),
        (np.hstack([S, Z, Z]), speed),  # speed limits
        (np.hstack([-S, Z, Z]), speed),
        (np.hstack([S, -E, Z]), upper - JOINT_MARGIN),  # JOINT_MARGIN inside the joint limits
        (np.hstack([-S, -E, Z]), lower - JOINT_MARGIN),
        (np.hstack([S, Z, -E]), SPEED_MARGIN * speed),  # under SPEED_MARGIN of the speed limits
        (np.hstack([-S, Z, -E]), SPEED_MARGIN * speed),
        (np.hstack([np.zeros((2 * nj, nv)), -np.eye(2 * nj)]), np.zeros(2 * nj)),  # slacks >= 0
    ]
    G = np.vstack([g for g, _ in rows])
    h = np.concatenate([b for _, b in rows])
    v = quadprog_solve_qp(P, p, G, h)[:nv]
    return projecttojointlimits(pin.integrate(model, q, v))

# def ik_step_pinv(q, displacement, moving=None, height=None, posture=None):
#     """One hierarchical IK step: each limb's ball moved by its displacement (in LIMBS order), the
#     held ones first in TASK_ORDER, then the moving one, each in the null space of the ones before;
#     then, if given, the trunk's origin brought GAIN of the way to that height [m]; then the trunk
#     turned upright and facing the wall, both as far as the balls allow; last, if given, the joints
#     brought POSTURE_GAIN of the way to posture's, as far as all the above allows.
#     - The held balls are solved exactly (their sockets are always in reach); the moving ball and
#       the trunk with `damped`, so the moving ball gets as close as it can to a target out of reach
#       while the held balls stay in their sockets [Ours]. With pinv for all and a fixed order, a
#       long hand move with OUT 0.05 / ABOVE 0.10 diverged (NaN), or pulled a held hand 6 mm out.
#     - A joint the step would push past a limit is locked there and the step solved again without
#       it [Ours]: with clipping alone (the planner's), a held ball drifted up to 10 mm off its
#       socket on a level-2 climb (seed 5226)."""
#     pin.computeJointJacobians(model, data, q)
#     pin.updateFramePlacements(model, data)
#     order = [limb for limb in TASK_ORDER if limb != moving] + ([moving] if moving else [])
#     tasks = [
#         (
#             pin.getFrameJacobian(model, data, BALL[limb], pin.LOCAL_WORLD_ALIGNED)[:3],
#             LIMBS.index(limb),
#             limb == moving,
#         )
#         for limb in order
#     ]
#     trunk_J = pin.getJointJacobian(model, data, 1, pin.LOCAL_WORLD_ALIGNED)
#     trunk, rise = trunk_J[3:], trunk_J[2:3]  # its rotation; its origin's height
#     turn = GAIN * pin.log3(TRUNK_UPRIGHT @ data.oMi[1].rotation.T)
#     lift = GAIN * np.array([(height or 0.0) - data.oMi[1].translation[2]])
#     free = np.ones(model.nv)  # 0: a locked joint (its column out of every Jacobian)
#     for _ in range(LOCK_PASSES):
#         vq, P = np.zeros(model.nv), np.eye(model.nv)
#         for J, i, is_moving in tasks:
#             JP = J * free @ P
#             vq = vq + (damped(JP) if is_moving else pinv(JP)) @ (displacement[i] - J * free @ vq)
#             P = P - pinv(JP) @ JP  # the exact null space, also under a damped task
#         if height is not None:
#             Jz = rise * free
#             vq = vq + damped(Jz @ P) @ (lift - Jz @ vq)
#             P = P - pinv(Jz @ P) @ Jz @ P
#         vq = vq + damped(trunk * free @ P) @ (turn - trunk * free @ vq)
#         if posture is not None:
#             TP = trunk * free @ P
#             P = P - pinv(TP) @ TP
#             joints = np.eye(model.nv)[6:] * free
#             vq = vq + damped(joints @ P) @ (POSTURE_GAIN * (posture[7:] - q[7:]) - joints @ vq)
#         new = pin.integrate(model, q, vq)
#         past = (new > model.upperPositionLimit) | (new < model.lowerPositionLimit)
#         past = past[7:]  # the joints' (the free flyer has no limits)
#         if not past.any():
#             break
#         free[6:][past] = 0
#     return projecttojointlimits(new)
#TODO: answer question: how much do this ik step differ from the planners ik. Do we need all these gains on lift etc? do we need the damped value? We should be using hte same helpers if necessary, but the collision chceching is not super necessary.
# also, this ik_step should just be getting hte joint positions that follow the moving limb's ball trajectory. So we don't need to check for all joints.
#TODO: Also, we should make sure that at contacts, in later step, we check com constraint and torque manipulability on the forces.


def projecttojointlimits(q):
    return np.minimum(np.maximum(model.lowerPositionLimit, q), model.upperPositionLimit)


_poses = {}  # (wall id, stance) -> its stance pose


def start_pose(wall: WallModel, stance: Stance):
    """The stance pose: the hang posture with its trunk fitted rigidly to the sockets [simulator
    sim.py reset_hanging], then IK, the joints pulled toward the hang pose's (posture), until every
    ball is within EPSILON of its socket and q has SETTLED. If q still moves after MAX_ITERATIONS
    (joints locking and unlocking at a limit), the last q, the balls within EPSILON. Deterministic,
    so cached. ValueError if the balls are not reached."""
    k = (wall.id, key(stance))
    if k not in _poses:
        t = sockets(wall, stance)
        height = t[:, 2].mean() + TRUNK_HEIGHT
        cp, cs = BALLS_HANG.mean(0), t.mean(0)
        U, _, Vt = np.linalg.svd((BALLS_HANG - cp).T @ (t - cs))
        R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
        q = Q_HANG.copy()
        q[0:3] = cs - R @ cp
        q[3:7] = pin.Quaternion(R).coeffs()
        for _ in range(MAX_ITERATIONS):
            error = t - balls(q)
            new = ik_step(q, STEP * error, height=height, posture=Q_HANG)
            settled = norm(pin.difference(model, q, new)) < SETTLED
            q = new
            if settled and norm(error, axis=1).max() < EPSILON:
                break
        error = norm(t - balls(q), axis=1).max()
        if not error < EPSILON:
            raise ValueError(f"start pose: balls {error * 1000:.0f} mm from sockets")
        _poses[k] = q
    return _poses[k]
#TODO: we should check that the start pose maximises for torque manipulability so that the end effectors can push downwards, and then last step check if COM is within the support polygon. For every transition, we should in the later steps check the configuration that the robot arrived to the stance at, and readjust the body configuration to maximise torque manipulability and COM stability before starting hte next transition.


def ball_path(p_a, p_b):
    """B(t), t in [0, MOVE_TIME]: five Bezier curves through the corridor's corners, each at rest
    at both ends, so continuous in position, velocity and acceleration where they meet [C-CROC
    eq. 4, bezier.connecting] and each a straight line (it cannot cut a corner into a hold, nor
    rise above it): up ABOVE out of socket a, out OUT from the wall, across to in front of socket
    b, in, and down into socket b.
    
    - p_a, p_b: the moving ball's sockets on holds a and b
    
    
    """
    rest, up, out = np.zeros(3), np.array([0.0, 0.0, ABOVE]), np.array([0.0, -OUT, 0.0])
    corners = [p_a, p_a + up, p_a + up + out, p_b + up + out, p_b + up, p_b]
    times = np.cumsum([0.0, CORNER_TIME, CORNER_TIME, MOVE_TIME - 4 * CORNER_TIME, CORNER_TIME])
    times = [*times, MOVE_TIME]
    curves = [
        Bezier(connecting((p, rest, rest), (q, rest, rest), t1 - t0), t0, t1)
        for p, q, t0, t1 in zip(corners, corners[1:], times, times[1:], strict=False)
    ]

    def B(t):
        t = min(max(t, 0.0), MOVE_TIME)
        return next(curve for curve in curves if t <= curve.T_max_)(t)

    return B


def timing(t, duration=MOVE_TIME):
    """s(t): min-jerk, from 0 to 1 over duration, zero velocity and acceleration at both ends."""
    tau = min(max(t / duration, 0.0), 1.0)
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def moving_limb(a: Stance, b: Stance) -> str:
    (limb,) = [limb for limb in LIMBS if a.contacts[limb] != b.contacts[limb]]
    return limb


def track_move(wall: WallModel, a: Stance, b: Stance, q0):
    """The configurations of the move a -> b, one every DT from q0 (excluded) to the end: at each
    sample, ITERATIONS IK steps closing GAIN of each ball's error to its target (the moving ball's
    on ball_path, the others' their sockets), the joints pulled toward the hang pose's. Where the
    path is out of the limb's reach, the moving ball only gets as close as it can (ik_step). It
    ends exactly in b's stance pose, where b -> c starts: from within JOINT_TOL it jumps there,
    from further it blends there over BLEND_TIME, balls held. Also the moving ball's worst distance
    from its path [m]. ValueError, naming the move, if a ball gets MAX_ERROR from its socket."""
    limb = moving_limb(a, b) # the limb that moves
    move = f"{limb} {a.contacts[limb]} -> {b.contacts[limb]}"
    i = LIMBS.index(limb)
    held = sockets(wall, a)
    B = ball_path(held[i], sockets(wall, b)[i])
    q, worst, out = q0, 0.0, [] # out will contain the configurations of the move, one every DT from q0 (excluded) to the end
    for k in range(1, round(MOVE_TIME / DT) + 1):
        target = held.copy()
        target[i] = B(k * DT)
        height = target[:, 2].mean() + TRUNK_HEIGHT
        for _ in range(ITERATIONS):
            q = ik_step(q, (target - balls(q)), moving=limb, height=height, posture=Q_HANG)
        error = norm(target - balls(q), axis=1)
        if not np.delete(error, i).max() <= MAX_ERROR:  # also NaN: the IK diverged
            e = np.delete(error, i).max() * 1000
            raise ValueError(f"{move}: a held ball {e:.0f} mm from its socket at {k * DT:.2f} s")
        worst = max(worst, error[i])
        out.append(q)
    qb, sockets_b = start_pose(wall, b), sockets(wall, b)
    gap = np.abs(q[7:] - qb[7:])
    if gap.max() > JOINT_TOL:
        # min-jerk peaks at 1.875 x the mean speed: no joint over SPEED_MARGIN of its speed limit
        blend = max(BLEND_TIME, 1.875 * (gap / SPEED_LIMIT).max() / SPEED_MARGIN)
        q_end = q
        for k in range(1, int(np.ceil(blend / DT)) + 1):
            q = pin.interpolate(model, q_end, qb, timing(k * DT, blend))
            e = norm(sockets_b - balls(q), axis=1).max()
            if not e <= MAX_ERROR:
                raise ValueError(f"{move}: a ball {e * 1000:.0f} mm from its socket in the blend")
            out.append(q)
    out[-1] = qb
    return out, worst


_cache = {}  # (wall id, a, b) -> Feasibility


def feasible_ik_traj(wall: WallModel, a: Stance, b: Stance) -> Feasibility:
    """Can the robot go from stance a to stance b? track_move from a's stance pose: every move ends
    in its stance pose, so this is the move optimise makes, and cached. Infeasible if a stance pose
    is not reached, a ball gets MAX_ERROR from its socket, or the moving ball MAX_ERROR from its
    path (whose end is socket b). An IK that diverges (NaN) is one of these: no overflow warning."""
    k = (wall.id, key(a), key(b))
    if k not in _cache:
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                _, worst = track_move(wall, a, b, start_pose(wall, a))
            if worst <= MAX_ERROR:
                _cache[k] = Feasibility(True, cost=worst)
            else:
                reason = f"{key(a)} -> {key(b)}: the moving ball {worst * 1000:.0f} mm off its path"
                _cache[k] = Feasibility(False, reason=reason)
        except ValueError as e:
            _cache[k] = Feasibility(False, reason=str(e))
    return _cache[k]


#TODO: probably in another file that's planner/kinematics/etc agnostic. But we need to come up with a good cache implementation. This should include: 
# - The level of feasibility it's checking (ik_contact, ik_traj, and in the future possibly RL from constance)
# - Stance a and b (or just b, since a is not used in the feasibility check). This should be wall agnostic, as different walls can have the same stance transition.
# - A boolean for whether the feasibility check was successful or not, optionally (but ideally) a reason for failure, and possibly a cost that the astar can have into account. 
# - The cache is mainly for the planner, so that it can check which nodes are good to expand or not. 