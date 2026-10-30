"""A* over stances. A move puts one limb on a free hold if feasible(wall, a, b) says so; cost 1."""

import heapq
import json
from itertools import combinations, count
from pathlib import Path

import microban_model
import mujoco
import numpy as np
import pinocchio as pin
from climbing_sim import Hold as SimHold
from climbing_sim import Wall as SimWall
from climbing_sim import build_model
from climbing_sim.wall import GRID_X0, GRID_Z0, HOLE_PITCH
from numpy.linalg import norm, pinv

from rcr.types import LIMBS, ContactPlan, Feasibility, FeasibilityFn, Stance, WallModel

SOCKET = np.array([0.0, -0.026, -0.014])  # ball centre from Hold.position [simulator wall.py]
HANG = Path(__file__).parents[3] / "third_party/simulator/climbing_sim/assets/poses/hang.json"
EPSILON = 1e-3  # [m]
DT = 0.5
MAX_ITERATIONS = 200
WEIGHT = 1.0  # A* heuristic weight: 1.0 = admissible, 0.0 = Dijkstra
# TODO: order by move: moving arm, diagonal leg, other leg, other arm
TASK_ORDER = ("right_hand", "left_hand", "right_foot", "left_foot")
# trunk z up, x into the wall (+y) [simulator README, Frames]
TRUNK_UPRIGHT = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

# trunk free: q = (x, y, z, qx, qy, qz, qw, 19 joints)
model = pin.buildModelFromUrdf(str(microban_model.urdf(climb=True)), pin.JointModelFreeFlyer())
data = model.createData()
BALL = {limb: model.getFrameId(f"{limb}_ball") for limb in LIMBS}


def jointlimitscost(model, q):
    up = max(q - model.upperPositionLimit)
    down = max(model.lowerPositionLimit - q)
    return max(0, max(up, down))


def jointlimitsviolated(model, q):
    """Return true if config not in joint limits"""
    return jointlimitscost(model, q) > 0.0


def projecttojointlimits(model, q):
    return np.minimum(np.maximum(model.lowerPositionLimit, q), model.upperPositionLimit)


def qpos(m, q):
    """Pinocchio q -> MuJoCo qpos: quaternion (x, y, z, w) -> (w, x, y, z), joints by name."""
    out = np.zeros(m.nq)
    out[0:3], out[3:7] = q[0:3], q[[6, 3, 4, 5]]
    for j in range(2, model.njoints):
        out[m.joint(model.names[j]).qposadr] = q[model.idx_qs[j]]
    return out


_scenes = {}


def scene(wall):
    """The simulator's MuJoCo scene of this wall (built once), and what collision() needs."""
    if wall.id not in _scenes:
        # Hold.position is the simulator's bolt_axis: back to (col, row)
        cells = [
            (
                round((h.position[0] - GRID_X0) / HOLE_PITCH) - 1,
                round((h.position[2] - GRID_Z0) / HOLE_PITCH),
            )
            for h in wall.holds
        ]
        m = build_model(wall=SimWall(holds=[SimHold(*c) for c in cells]), cameras=[])
        d = mujoco.MjData(m)
        robot = {g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == m.body("trunk").id}
        ball = {limb: m.geom(f"{limb}_ball_coll").id for limb in LIMBS}
        hold = {
            h.id: m.body(f"hold{n}_c{c}_r{r}").id
            for n, (h, (c, r)) in enumerate(zip(wall.holds, cells, strict=True))
        }
        # self pairs already touching in the hanging posture (like an SRDF's disabled pairs)
        d.qpos[:] = qpos(m, Q_HANG)
        mujoco.mj_forward(m, d)
        hanging = {frozenset((c.geom1, c.geom2)) for c in d.contact[: d.ncon]}
        hanging = {pair for pair in hanging if pair <= robot}
        _scenes[wall.id] = m, d, robot, ball, hold, hanging
    return _scenes[wall.id]


def collision(wall, k, q):
    """The first robot contact, but a ball in its own hold or self pairs touching when hanging,
    as 'body into body (depth)'; "" if none."""
    m, d, robot, ball, hold, hanging = scene(wall)
    own = {(ball[limb], hold[h]) for limb, h in zip(LIMBS, k, strict=True)}
    d.qpos[:] = qpos(m, q)
    mujoco.mj_forward(m, d)
    for c in d.contact[: d.ncon]:
        a, b = c.geom1, c.geom2
        if (
            frozenset((a, b)) in hanging
            or (a, m.geom_bodyid[b]) in own
            or (b, m.geom_bodyid[a]) in own
        ):
            continue
        if a in robot or b in robot:
            body_a, body_b = (m.body(m.geom_bodyid[g]).name for g in (a, b))
            return f"{body_a} into {body_b} ({mm(-c.dist)} deep)"
    return ""


def chain(limb):
    """Joints from the trunk to the ball."""
    joints, j = [], model.frames[BALL[limb]].parentJoint
    while j > 1:
        joints.insert(0, j)
        j = model.parents[j]
    return joints


# first joint (shoulder / hip) in the trunk frame, and reach = limb stretched straight
ROOT = {limb: model.jointPlacements[chain(limb)[0]].translation for limb in LIMBS}
REACH = {
    limb: sum(norm(model.jointPlacements[j].translation) for j in chain(limb)[1:])
    + norm(model.frames[BALL[limb]].placement.translation)
    for limb in LIMBS
}
# the longest step the simulator's routes leave a limb [simulator routes.py]
STEP = 0.105
MIN_APART = 1.1  # two balls at least this fraction of their roots' distance apart [Ours]
MAX_APART = 0.58  # and at most this fraction of their reaches + roots' distance [Ours]
# [m] most one hand is above the other, and one foot above the other [Ours]: the example
# plan the optimiser solved has up to 0.195 m (hands) and 0.135 m (feet)
MAX_LEVEL = 0.15

Q_HANG = pin.neutral(model)
for name, angle in json.loads(HANG.read_text())["joints"].items():
    Q_HANG[model.idx_qs[model.getJointId(name)]] = angle
pin.framesForwardKinematics(model, data, Q_HANG)
BALLS_HANG = np.array([data.oMf[BALL[limb]].translation for limb in LIMBS])


def reach(targets):
    """Two balls are at most MAX_APART of their reaches + the distance between their roots
    apart, and at least MIN_APART of that root distance."""
    for i, j in combinations(range(4), 2):
        a, b = LIMBS[i], LIMBS[j]
        roots, apart = norm(ROOT[a] - ROOT[b]), norm(targets[i] - targets[j])
        most, least = MAX_APART * (REACH[a] + REACH[b] + roots), MIN_APART * roots
        if apart > most:
            return fail(f"reach: {a}, {b} too far apart ({mm(apart)} > {mm(most)})")
        if apart < least:
            return fail(f"reach: {a}, {b} too close ({mm(apart)} < {mm(least)})")
    right_hand, left_hand, right_foot, left_foot = targets  # LIMBS order; x right, z up
    if right_hand[0] <= left_hand[0]:
        gap = right_hand[0] - left_hand[0]
        return fail(f"reach: right hand not right of left hand ({mm(gap)})")
    if right_foot[0] <= left_foot[0]:
        gap = right_foot[0] - left_foot[0]
        return fail(f"reach: right foot not right of left foot ({mm(gap)})")
    above = min(right_hand[2], left_hand[2]) - max(right_foot[2], left_foot[2])
    if above <= 0:
        return fail(f"reach: a hand not above the feet ({mm(above)} above the higher foot)")
    if abs(right_hand[2] - left_hand[2]) > MAX_LEVEL:
        level = abs(right_hand[2] - left_hand[2])
        return fail(f"reach: one hand too far above the other ({mm(level)} > {mm(MAX_LEVEL)})")
    if abs(right_foot[2] - left_foot[2]) > MAX_LEVEL:
        level = abs(right_foot[2] - left_foot[2])
        return fail(f"reach: one foot too far above the other ({mm(level)} > {mm(MAX_LEVEL)})")
    return None


def mm(metres):
    return f"{metres * 1000:.0f} mm"


def fail(reason):
    return Feasibility(False, reason=reason)


def initial_q(targets):
    """Hang posture, trunk fitted rigidly to the targets [simulator sim.py reset_hanging]."""
    cp, cs = BALLS_HANG.mean(0), targets.mean(0)
    U, _, Vt = np.linalg.svd((BALLS_HANG - cp).T @ (targets - cs))
    R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    q = Q_HANG.copy()
    q[0:3] = cs - R @ cp
    q[3:7] = pin.Quaternion(R).coeffs()
    return q


def ik(wall, k):
    """Hierarchical IK on the four ball positions, joints clipped; then joint limits, collision."""
    t = targets(wall, k)
    q = initial_q(t)
    for _ in range(MAX_ITERATIONS):
        pin.computeJointJacobians(model, data, q)
        pin.updateFramePlacements(model, data)
        balls = [data.oMf[BALL[limb]].translation for limb in LIMBS]
        error = dict(zip(LIMBS, t - balls, strict=True))

        if max(norm(e) for e in error.values()) < EPSILON:
            if jointlimitsviolated(model, q):
                return Feasibility(False, reason="joint limits"), q
            for limb, ball in zip(LIMBS, balls, strict=True):
                side = 1 if limb.startswith("right") else -1  # robot's right is +x
                past = side * (data.oMi[chain(limb)[0]].translation[0] - ball[0])
                if past > 0:
                    return fail(f"{limb} crosses its root (by {mm(past)})"), q
            if hit := collision(wall, k, q):
                return fail(f"collision: {hit}"), q
            return Feasibility(True), q

        # each task in the null space of the ones before
        vq, P = np.zeros(model.nv), np.eye(model.nv)
        
        for limb in TASK_ORDER:
            J = pin.getFrameJacobian(model, data, BALL[limb], pin.LOCAL_WORLD_ALIGNED)[:3]
            JP_pinv = pinv(J @ P)
            vq = vq + JP_pinv @ (error[limb] - J @ vq)
            P = P - JP_pinv @ J @ P

        # last: trunk upright and facing the wall, as far as the balls allow
        J = pin.getJointJacobian(model, data, 1, pin.LOCAL_WORLD_ALIGNED)[3:]
        e = pin.log3(TRUNK_UPRIGHT @ data.oMi[1].rotation.T)
        vq = vq + pinv(J @ P) @ (e - J @ vq)

        q = pin.integrate(model, q, vq * DT)
        q = projecttojointlimits(model, q)

    left = max(norm(e) for e in error.values())
    return fail(f"ik: balls not reached ({mm(left)} left after {MAX_ITERATIONS} iterations)"), q


def targets(wall, k):
    """Socket of each limb's hold, in LIMBS order."""
    position = {h.id: h.position for h in wall.holds}
    return np.array([position[hold] for hold in k]) + SOCKET


_cache = {}


def feasible(wall: WallModel, a: Stance, b: Stance) -> Feasibility:
    """Reach, then IK, on stance b (does not depend on a, so cached)."""
    k = (wall.id, *key(b))
    if k not in _cache:
        _cache[k] = reach(targets(wall, key(b))) or ik(wall, key(b))[0]
    return _cache[k]


def key(stance):
    return tuple(stance.contacts[limb] for limb in LIMBS)


def stance(k):
    return Stance(contacts=dict(zip(LIMBS, k, strict=True)))



def solve(wall: WallModel, feasible: FeasibilityFn, HEURISTIC_WEIGHT: float = 1.0) -> ContactPlan:
    """Moves to a hand on a top hold, or else to the highest hand reachable.
    - z is the height of each hold.
    - k is a node: the tuple of holds for each limb, in LIMBS order.
    """
    z = {h.id: h.position[2] for h in wall.holds}
    top = max(z.values())

    def hand_z(k):
        return max(z[k[0]], z[k[1]])

    def h(k):  # steps left for all four limbs: not admissible, so not always the fewest moves
        return sum(top - z[hold] for hold in k) / STEP

    start = key(wall.start)
    g, parent, closed, tie = {start: 0}, {}, set(), count()
    heap = [(h(start), 0, next(tie), start)]  # (f, -g, tie, stance): deeper first on equal f
    best = start
    while heap:
        *_, k = heapq.heappop(heap)
        if k in closed:
            continue
        closed.add(k)
        if (hand_z(k), -g[k]) > (hand_z(best), -g[best]):
            best = k
        if hand_z(k) == top:
            break
        for i in range(4):
            # holds limb i can reach with the other three where they are
            children = [k[:i] + (hold,) + k[i + 1 :] for hold in z if hold not in k]
            children = [c for c in children if reach(targets(wall, c)) is None]
            for child in children:
                if g[k] + 1 >= g.get(child, np.inf):
                    continue
                if feasible(wall, stance(k), stance(child)).feasible:
                    g[child], parent[child] = g[k] + 1, k
                    heapq.heappush(heap, (g[child] + HEURISTIC_WEIGHT*h(child), -g[child], next(tie), child))
    path = [best]
    while path[-1] in parent:
        path.append(parent[path[-1]])
    return ContactPlan(wall_id=wall.id, stances=[stance(k) for k in reversed(path)])
