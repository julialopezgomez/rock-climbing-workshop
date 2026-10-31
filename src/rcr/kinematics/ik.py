"""Contact-prioritized differential IK for Microban's climbing trajectories.

The moving ball follows a piecewise-smooth hold-clearing corridor while held
balls remain fixed. Uses Pinocchio null-space IK, without dynamics or collisions.
"""

import json
from pathlib import Path

import microban_model
import numpy as np
import pinocchio as pin
from numpy.linalg import norm, pinv

from rcr.kinematics.bezier import Bezier
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
MOVE_TIME = 2.0  # [s] per move
DT = 0.02  # [s] between samples: the robot's 50 Hz loop
GAIN = 0.5  # fraction of a ball's tracking error corrected per sample [Ours]
MAX_ERROR = 0.01  # [m] a held ball further than this from its socket: the move cannot be done
DAMPING = 0.03  # damped(): the moving ball's and the trunk's tasks [Ours]
ITERATIONS = 3  # IK steps per sample, toward that sample's targets [Ours]
# [m] the trunk's origin above the mean height of the four balls' targets [Ours]: the hang pose
# fitted to the start stance has it 7 mm below; raised 2 cm, the balls still reach their sockets
# with no joint at a limit (3 cm puts two at one)
TRUNK_HEIGHT = 0.015
# The stance pose [Ours]: every move starts and ends in its stances' (start_pose), so a move checked
# alone (feasible_ik_traj) is the move optimise makes. The joints are pulled toward the hang pose's,
# last and lowest, also during the move: 0.1 ended 195 of 243 level-2/3 moves already within
# JOINT_TOL of b's (0.3: 186; without, none).
SPEED_MARGIN = 0.8  # fraction of URDF speed limit used in final stance blending
POSTURE_GAIN = 0.1  # fraction of each joint's gap to the posture closed per IK step
SETTLED = 1e-5  # an IK step this small (norm of the change of q): q has stopped moving
JOINT_TOL = 0.01  # [rad] a move ending this close to b's stance pose jumps to it; further, blends
BLEND_TIME = 0.3  # [s] min-jerk from where a move ended to b's stance pose, balls held, at least
WAIT_TIME = 0.5  # [s] each move ends holding still in b's stance pose, where b -> c starts [Ours]
# Robust differential IK: contacts are prioritized, with damped lower-priority tasks.
# Joint steps are capped to respect actuator speeds and prevent singularity jumps.

# trunk free: q = (x, y, z, qx, qy, qz, qw, 19 joints)
model = pin.buildModelFromUrdf(str(microban_model.urdf(climb=True)), pin.JointModelFreeFlyer())
data = model.createData()
BALL = {limb: model.getFrameId(f"{limb}_ball") for limb in LIMBS}
SPEED_LIMIT = model.velocityLimit[6:]  # [rad/s] the joints' (6.6) [microban model URDF]



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


def damped(A):
    """Damped least-squares inverse: like pinv, but the step stays bounded where A is near
    singular (a limb stretched out), so a target out of reach is only approached."""
    return A.T @ np.linalg.inv(A @ A.T + DAMPING**2 * np.eye(A.shape[0]))


def ik_step(q, displacement, moving=None, height=None, posture=None):
    """One hierarchical differential IK step, following the ARO null-space method.

    Held balls have first priority (simultaneously), the moving ball second,
    then trunk height/orientation and a neutral posture. Integrate on Pinocchio's
    floating-base configuration manifold; never integrate quaternion components
    as if they were independent Euclidean coordinates.

    Returns one bounded configuration update. Repeated small corrections are
    performed by start_pose() and track_move(), not by this routine.
    """
    pin.computeJointJacobians(model, data, q)
    pin.updateFramePlacements(model, data)

    J = {
        limb: pin.getFrameJacobian(
            model, data, BALL[limb], pin.LOCAL_WORLD_ALIGNED
        )[:3]
        for limb in LIMBS
    }
    dq = np.zeros(model.nv)
    P = np.eye(model.nv)

    def task(A, target, damping=0.0):
        nonlocal dq, P
        AP = A @ P
        inv = pinv(AP) if damping == 0 else damped(AP)
        dq = dq + inv @ (target - A @ dq)
        P = P - pinv(AP) @ AP

    # Solve all anchored contacts together: no arbitrary hand/foot preference.
    held = [limb for limb in LIMBS if limb != moving]
    A = np.vstack([J[limb] for limb in held])
    b = np.concatenate([displacement[LIMBS.index(limb)] for limb in held])
    task(A, b)

    if moving is not None:
        task(J[moving], displacement[LIMBS.index(moving)], DAMPING)

    base_J = pin.getJointJacobian(model, data, 1, pin.LOCAL_WORLD_ALIGNED)
    if height is not None:
        target_z = GAIN * (height - data.oMi[1].translation[2])
        task(base_J[2:3], np.array([target_z]), DAMPING)
    orientation_error = GAIN * pin.log3(TRUNK_UPRIGHT @ data.oMi[1].rotation.T)
    task(base_J[3:], orientation_error, DAMPING)

    if posture is not None:
        A = np.eye(model.nv)[6:]
        target = POSTURE_GAIN * pin.difference(model, q, posture)[6:]
        task(A, target, DAMPING)

    # Scale the *whole* step, preserving the null-space task priorities.
    # The speed bound applies to the total 20 ms motion, shared by ITERATIONS
    # substeps. Joint position bounds are enforced without clipping away
    # the contact solution after integration.
    scale = 1.0
    speed = SPEED_LIMIT * DT / ITERATIONS
    joint_dq = dq[6:]
    active = np.abs(joint_dq) > 1e-12
    if np.any(active):
        scale = min(scale, float(np.min(speed[active] / np.abs(joint_dq[active]))))

    lower = model.lowerPositionLimit[7:]
    upper = model.upperPositionLimit[7:]
    joint_q = q[7:]
    for j, delta in enumerate(joint_dq):
        if delta > 1e-12:
            scale = min(scale, max(0.0, (upper[j] - joint_q[j]) / delta))
        elif delta < -1e-12:
            scale = min(scale, max(0.0, (lower[j] - joint_q[j]) / delta))

    q_next = pin.integrate(model, q, scale * dq)
    if not np.all(np.isfinite(q_next)):
        raise ValueError("Differential IK produced non-finite joint positions")
    return q_next


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
    """B(t), t in [0, MOVE_TIME]: a five-segment hold-clearing corridor.
    Segment durations are proportional to length, with quintic time scaling on
    each segment to give zero velocity and acceleration at the corners.
    The geometric path stays on each straight segment: up, out, across, in, down.

    - p_a, p_b: the moving ball's sockets on holds a and b


    """
    up, out = np.array([0.0, 0.0, ABOVE]), np.array([0.0, -OUT, 0.0])
    corners = [p_a, p_a + up, p_a + up + out, p_b + up + out, p_b + up, p_b]
    lengths = norm(np.diff(corners, axis=0), axis=1)
    times = [*(MOVE_TIME * np.cumsum([0.0, *lengths]) / lengths.sum())] # the time at each corner, from 0 to MOVE_TIME
    curves = [
        Bezier([p, q], t0, t1)
        for p, q, t0, t1 in zip(corners, corners[1:], times, times[1:], strict=False)
    ]

    def B(t):
        t = min(max(t, 0.0), MOVE_TIME)
        curve = next(curve for curve in curves if t <= curve.T_max_)
        # Stop smoothly at each corridor corner instead of jumping in velocity.
        # Geometry remains on the same straight pieces, preserving hold clearance.
        duration = curve.T_max_ - curve.T_min_
        smooth_t = curve.T_min_ + duration * timing(t - curve.T_min_, duration)
        return curve(smooth_t)

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
    from further it blends there over BLEND_TIME, balls held; then holds still there for
    WAIT_TIME. Also the moving ball's worst distance from its path [m]. ValueError, naming the
    move, if a ball gets MAX_ERROR from its socket."""
    limb = moving_limb(a, b)
    move = f"{limb} {a.contacts[limb]} -> {b.contacts[limb]}"
    i = LIMBS.index(limb)
    held = sockets(wall, a)
    B = ball_path(held[i], sockets(wall, b)[i])
    q, worst, out = q0, 0.0, []
    for k in range(1, round(MOVE_TIME / DT) + 1):
        target = held.copy()
        target[i] = B(k * DT)
        height = target[:, 2].mean() + TRUNK_HEIGHT
        for _ in range(ITERATIONS):
            q = ik_step(q, GAIN * (target - balls(q)), moving=limb, height=height, posture=Q_HANG)
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
    out += [qb] * round(WAIT_TIME / DT)
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