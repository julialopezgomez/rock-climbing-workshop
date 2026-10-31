"""The kinematic climbing motion with Rhoban's placo [Ours]: the same moves as ik.py, solved by
placo's KinematicsSolver instead of ik.ik_step.

Everything but the IK solver is ik.py's, so the two can be compared move by move: the moving
ball's path (ik.ball_path), the sockets, the hang posture, the trunk upright and its height
(ik.TRUNK_HEIGHT), the timing (ik.DT, ik.MOVE_TIME, ik.WAIT_TIME), every move ending in its
stance pose, and the output (optimise -> Trajectory, as rcr.kinematics.optimise).

placo's solver is a QP each step [placo docs, kinematics]: the held balls are hard tasks (they
stay in their sockets exactly or the step fails), the moving ball, the trunk and the posture are
soft tasks with weights, and the joint and speed limits are constraints. placo pins its own
pinocchio (3.8), the project another (4.x): this module is imported only by
scripts/placo_trajectory.py, which runs in its own environment, never by rcr.kinematics.
"""

import microban_model
import numpy as np
import pinocchio as pin
import placo

from rcr.kinematics import ik, validate_plan
from rcr.types import LIMBS, MICROBAN_JOINTS, ContactPlan, Stance, Trajectory, WallModel

ITERATIONS = 3  # solver steps per sample, as ik.ITERATIONS
# the moving ball's position task [Ours]: on the N3 plan (l3_s41545, 8 moves) its worst distance
# from the path is 0.1 mm at 300, 0.9 mm at 30, 19 mm at 1 (the trunk and posture pull it away)
MOVING_WEIGHT = 300.0
TRUNK_WEIGHT = 1e-2  # the trunk upright and at its height [Ours]
POSTURE_WEIGHT = 1e-3  # the joints toward the hang pose's [Ours]
STANCE_STEPS = 300  # at most, for a stance pose
SETTLE_TIME = 1.0  # [s] at most, after the path, for the body to settle on the new sockets
# placo shaping the motion (generate=True) [Ours]: before the moving ball lets go, the body shifts
# its CoM into the static-equilibrium polygon of the three balls that stay [Stéphane Caron's
# static_balance, the workshop tutorial: 80 deg cones, 5 N a ball], COM_MARGIN inside its edge,
# and keeps it there while the limb is in the air. Soft: where the margin cannot be had, the CoM
# gets as close as it can and the held balls stay hard.
COM_MARGIN = 0.015  # [m]
COM_WEIGHT = 1e3
SHIFT_TIME = 1.0  # [s] the shift, all four balls held, before the moving ball's path starts
FRICTION_DEG, MAX_FORCE = 80.0, 5.0  # the polygon's contacts [the tutorial]
MASS = pin.computeTotalMass(ik.model)  # [kg]

robot = placo.RobotWrapper(str(microban_model.urdf(climb=True)), placo.Flags.ignore_collisions)
solver = placo.KinematicsSolver(robot)
solver.mask_fbase(False)  # the trunk is free: the balls hold the robot
solver.enable_joint_limits(True)
ball_tasks = {limb: solver.add_position_task(f"{limb}_ball", np.zeros(3)) for limb in LIMBS}
trunk_task = solver.add_orientation_task("trunk", ik.TRUNK_UPRIGHT)
trunk_task.configure("trunk", "soft", TRUNK_WEIGHT)
height_task = solver.add_position_task("trunk", np.zeros(3))
height_task.mask.set_axises("z")
height_task.configure("height", "soft", TRUNK_WEIGHT)
posture_task = solver.add_joints_task()
posture_task.set_joints(
    {name: float(ik.Q_HANG[ik.model.idx_qs[ik.model.getJointId(name)]]) for name in MICROBAN_JOINTS}
)
posture_task.configure("posture", "soft", POSTURE_WEIGHT)


def set_targets(targets: np.ndarray, moving: str | None = None) -> None:
    """The balls' targets (LIMBS order); every ball hard but the moving one; the trunk's height
    ik.TRUNK_HEIGHT above their mean, as in ik.py."""
    for limb, target in zip(LIMBS, targets, strict=True):
        ball_tasks[limb].target_world = target
        hard = limb != moving
        ball_tasks[limb].configure(limb, "hard" if hard else "soft", 1.0 if hard else MOVING_WEIGHT)
    height_task.target_world = np.array([0.0, 0.0, targets[:, 2].mean() + ik.TRUNK_HEIGHT])


def step(steps: int) -> np.ndarray:
    """Solve `steps` times, each from where the last ended; the configuration (pinocchio's
    layout, as ik.model's)."""
    for _ in range(steps):
        robot.update_kinematics()
        solver.solve(True)
    robot.update_kinematics()
    return robot.state.q.copy()


_poses = {}  # (wall id, stance) -> its stance pose


def stance_pose(wall: WallModel, stance: Stance) -> np.ndarray:
    """The stance pose: the hang posture fitted rigidly to the sockets (as ik.start_pose), then
    placo with all four balls hard, the trunk and posture soft, until q stops moving. No speed
    limit: it is a pose, not a motion. ValueError if a ball stays ik.EPSILON from its socket."""
    k = (wall.id, ik.key(stance))
    if k not in _poses:
        t = ik.sockets(wall, stance)
        cp, cs = ik.BALLS_HANG.mean(0), t.mean(0)
        U, _, Vt = np.linalg.svd((ik.BALLS_HANG - cp).T @ (t - cs))
        R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
        q = ik.Q_HANG.copy()
        q[0:3] = cs - R @ cp
        q[3:7] = pin.Quaternion(R).coeffs()
        robot.state.q, robot.state.qd = q, np.zeros(ik.model.nv)
        solver.enable_velocity_limits(False)
        set_targets(t)
        for _ in range(STANCE_STEPS):
            new = step(1)
            settled = np.linalg.norm(pin.difference(ik.model, q, new)) < ik.SETTLED
            q = new
            if settled:
                break
        error = np.linalg.norm(t - ik.balls(q), axis=1).max()
        if not error < ik.EPSILON:
            raise ValueError(f"stance pose: a ball {error * 1000:.1f} mm from its socket")
        _poses[k] = q
    return _poses[k]


def support_polygon(balls: np.ndarray):
    """The static-equilibrium polygon (x, y vertices, counterclockwise) of balls (B x 3) in
    their cups, or None if they cannot hold the robot [static_balance]."""
    from scipy.spatial import QhullError
    from static_balance import ContactSet, PointContact, compute_static_equilibrium_polygon

    friction = float(np.tan(np.radians(FRICTION_DEG)))
    contacts = ContactSet(PointContact(p, [0.0, 0.0, 1.0], friction, MAX_FORCE) for p in balls)
    try:
        return compute_static_equilibrium_polygon(contacts, MASS)
    except (QhullError, ValueError, RuntimeError):
        return None


def track_move(wall: WallModel, a: Stance, b: Stance, q0: np.ndarray, generate: bool = False):
    """The configurations of the move a -> b, one every ik.DT from q0 (excluded), as
    ik.track_move: the moving ball along ik.ball_path, the others held (hard), ITERATIONS
    solver steps per sample within the joints' speed limits; then settling on b's sockets (at
    most SETTLE_TIME) and ik.WAIT_TIME holding still there. generate: placo also shapes the body's
    motion, the CoM into the support polygon of the three held balls first (SHIFT_TIME, all four
    held) and kept there until the moving ball lands (COM_MARGIN, COM_WEIGHT). Also the moving
    ball's worst distance from its path [m]. ValueError, naming the move, if the solver fails or a
    held ball gets ik.MAX_ERROR from its socket."""
    limb = ik.moving_limb(a, b)
    move = f"{limb} {a.contacts[limb]} -> {b.contacts[limb]}"
    i = LIMBS.index(limb)
    held = ik.sockets(wall, a)
    path = ik.ball_path(held[i], ik.sockets(wall, b)[i])
    robot.state.q, robot.state.qd = q0.copy(), np.zeros(ik.model.nv)
    solver.enable_velocity_limits(True)
    solver.dt = ik.DT / ITERATIONS
    q, worst, out = q0, 0.0, []
    polygon = support_polygon(np.delete(held, i, axis=0)) if generate else None
    com = None
    if polygon is not None:  # placo wants it clockwise
        com = solver.add_com_polygon_constraint([np.array(v) for v in polygon[::-1]], COM_MARGIN)
        com.configure("com", "soft", COM_WEIGHT)
        set_targets(held)
        for _ in range(round(SHIFT_TIME / ik.DT)):
            q = step(ITERATIONS)
            out.append(q)
    try:
        q, worst = follow(path, held, i, limb, move, q, out)
    finally:
        if com is not None:
            solver.remove_constraint(com)
    # settle on b's sockets: all four balls hard, the trunk and posture pulling, until q stops
    # moving (a blend in joint space to b's stance pose pulled a held ball 14 mm out on N3)
    set_targets(ik.sockets(wall, b))
    for _ in range(round(SETTLE_TIME / ik.DT)):
        new = step(ITERATIONS)
        settled = np.linalg.norm(pin.difference(ik.model, q, new)) < ik.SETTLED
        q = new
        out.append(q)
        if settled:
            break
    out += [q] * round(ik.WAIT_TIME / ik.DT)
    return out, worst


def follow(path, held: np.ndarray, i: int, limb: str, move: str, q: np.ndarray, out: list):
    """The moving ball (index i) along path for ik.MOVE_TIME, the others held: appends each
    sample's configuration to out; (the last one, the ball's worst distance from the path)."""
    worst = 0.0
    for k in range(1, round(ik.MOVE_TIME / ik.DT) + 1):
        target = held.copy()
        target[i] = path(k * ik.DT)
        set_targets(target, moving=limb)
        try:
            q = step(ITERATIONS)
        except RuntimeError as e:  # the QP: hard tasks out of reach within the limits
            raise ValueError(f"{move}: placo failed at {k * ik.DT:.2f} s ({e})") from e
        error = np.linalg.norm(target - ik.balls(q), axis=1)
        if not np.delete(error, i).max() <= ik.MAX_ERROR:
            e = np.delete(error, i).max() * 1000
            raise ValueError(f"{move}: a held ball {e:.0f} mm from its socket at {k * ik.DT:.2f} s")
        worst = max(worst, error[i])
        out.append(q)
    return q, worst


def optimise(wall: WallModel, plan: ContactPlan, generate: bool = False) -> Trajectory:
    """The full joint trajectory for this plan, as rcr.kinematics.optimise, by placo: the start
    pose of its first stance, then each move's samples every ik.DT (generate: see track_move)."""
    validate_plan(wall, plan)
    qs = [stance_pose(wall, plan.stances[0])]
    for a, b in zip(plan.stances, plan.stances[1:], strict=False):
        qs += track_move(wall, a, b, qs[-1], generate)[0]
    joints = [ik.model.idx_qs[ik.model.getJointId(name)] for name in MICROBAN_JOINTS]
    positions = [[float(q[j]) for j in joints] for q in qs]
    return Trajectory(
        joint_names=list(MICROBAN_JOINTS),
        times=[round(k * ik.DT, 9) for k in range(len(qs))],
        positions=positions,
        planned_positions=[row.copy() for row in positions],
        # pinocchio's quaternion is (x, y, z, w); ours (w, x, y, z)
        trunk_poses=[[float(v) for v in q[[0, 1, 2, 6, 3, 4, 5]]] for q in qs],
    )
