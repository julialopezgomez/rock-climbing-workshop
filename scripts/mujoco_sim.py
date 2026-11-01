"""Whole-body inverse-dynamics climb of a whole npz trajectory in MuJoCo [Ours].

Uses the organisers' ClimbingSim physics directly, with torque control and the sim's true state:
no servo model, no sensor delays. It tests the trajectory, not what the robot can do.

Each control step solves the accelerations for the tasks together (trunk pose, moving ball, held
balls still, moving foot's orientation, joints softly), then the contact forces and joint torques
that produce them: M a + h = S^T tau + J_c^T f.

Prints how many moves it made in a row, and stops at the first one it fails.

Run:
    uv run python scripts/mujoco_sim.py

On macOS with the MuJoCo viewer, use mjpython.
"""

from pathlib import Path

import mujoco
import numpy as np
from climbing_sim import JOINTS, ClimbingSim
from climbing_sim.routes import generate_wall
from climbing_sim.scene import BALL_SITES

from rcr.types import LIMBS, MICROBAN_JOINTS

# ------------------------------------------------------------
# Experiment
# ------------------------------------------------------------

# Mohammadreza's IK of level 3, seed 41545 (optimiser/swing-path-n2, data/results/to_n3)
NPZ = Path(__file__).parents[1] / "npz/wall-3-ik_l3_s41545.npz"
LEVEL, SEED = 3, 41545

DT = 0.02  # s between the rows of the npz
SPEED = 0.5  # of the trajectory's speed
SETTLE_TIME = 0.5  # s holding the first pose
GRACE_TIME = 0.5  # s for a moved ball to seat, holding the end of its move
HOOK_TOL = 0.006  # m, a ball this close to a socket's centre sits in it [simulator hooked()]

# PD gains on each task's acceleration, and its weight
KP_JOINT, W_JOINT = 1000.0, 0.001  # soft, so that a leg can give way when it hits a hold
KP_TRUNK, W_TRUNK = 400.0, 10.0
KP_BALL, W_BALL = 1000.0, 10.0
KP_FOOT, W_FOOT = 1000.0, 100.0
KD_HELD, W_HELD = 40.0, 100.0

TRUNK_OFFSET = np.array(
    [0.0, -0.01, 0.0]
)  # m, further from the wall than planned: fewer collisions
FORCE_REG = 1e-3  # contact forces against joint torques

# The XL330's torque at its 1.75 A current limit [Microban]
MAX_MOTOR_TORQUE = 0.605  # Nm

# The search for each foot move's tilt about the wall's x, relative to the plan: the plan's foot
# hits its next hold with its shaft, just before the ball seats
TILTS = np.arange(-35.0, 36.0, 5.0)  # deg
TILT_STRIDE = 2  # samples between searched tilts
TILT_CHANGE_COST = 0.15  # mm of penetration per degree of change between samples
TILT_COST = 0.01  # mm of penetration per degree of tilt
OUT_OF_REACH_COST = 1e3

RENDER = True
REALTIME = True


# ------------------------------------------------------------
# The trajectory
# ------------------------------------------------------------


def load_route():
    """The npz as (trunk positions, trunk quaternions w x y z, joints in the sim's order)."""
    q = np.load(NPZ)["q"]
    order = [MICROBAN_JOINTS.index(name) for name in JOINTS]
    return q[:, :3], q[:, 3:7], q[:, 7:][:, order]


def find_moves(balls, sockets):
    """Per move: its first and last sample, the moving limb's index, and the socket of each ball
    before and after. A stance is a sample with all four balls in sockets."""
    stances = []
    for i, at in enumerate(balls):
        d = np.linalg.norm(sockets[None] - at[:, None], axis=2)
        if (d.min(axis=1) < HOOK_TOL).all():
            stance = tuple(int(j) for j in d.argmin(axis=1))
            if not stances or stance != stances[-1][1]:
                stances.append((i, stance))
    moves = []
    for (i0, before), (i1, after) in zip(stances, stances[1:], strict=False):
        (limb,) = [k for k in range(len(LIMBS)) if before[k] != after[k]]
        moves.append({"i0": i0, "i1": i1, "limb": limb, "before": before, "after": after})
    return moves


def smooth(x, n=5):
    """Moving average over n samples, per column."""
    pad = np.pad(x, ((n // 2, n - 1 - n // 2), (0, 0)), mode="edge")
    return np.array([np.convolve(c, np.ones(n) / n, mode="valid") for c in pad.T]).T


def tilted(quat, degrees):
    """The quaternion turned by this many degrees about the wall's x."""
    rotation, out = np.zeros(4), np.zeros(4)
    mujoco.mju_axisAngle2Quat(rotation, np.array([1.0, 0.0, 0.0]), np.radians(degrees))
    mujoco.mju_mulQuat(out, rotation, quat)
    return out


def site_quat(data, site):
    out = np.zeros(4)
    mujoco.mju_mat2Quat(out, data.site_xmat[site])
    return out


def foot_tilts(sim, route, balls, move):
    """(samples, tilts): the foot's tilt along this move, by dynamic programming. The cheapest
    in penetration of the foot's body, changing slowly, from no tilt to no tilt. On `sim`, which
    this leaves anywhere: pass one that is not climbing."""
    m, d = sim.model, sim.data
    pos, quat, joints = route
    name = LIMBS[move["limb"]]
    site = m.site(BALL_SITES[name]).id
    body = m.site_bodyid[site]
    ball_geom = m.geom(f"{name}_ball_coll").id
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))

    def place(i):
        d.qpos[:3] = pos[i] + TRUNK_OFFSET
        d.qpos[3:7] = quat[i]
        d.qpos[sim._qadr] = joints[i]
        mujoco.mj_kinematics(m, d)

    place(move["i0"])
    mujoco.mj_comPos(m, d)
    mujoco.mj_jacSite(m, d, jp, jr, site)
    leg = np.flatnonzero(np.abs(jp[:, sim._vadr]).sum(axis=0) > 0)
    q_leg, v_leg = sim._qadr[leg], sim._vadr[leg]
    low, high = sim.joint_range[leg].T

    samples = list(range(move["i0"], move["i1"] + 1, TILT_STRIDE))
    if samples[-1] != move["i1"]:
        samples.append(move["i1"])
    cost = np.zeros((len(samples), len(TILTS)))
    for a, i in enumerate(samples):
        place(i)
        planned, q0 = site_quat(d, site), d.qpos[q_leg].copy()
        for b, tilt in enumerate(TILTS):
            target = tilted(planned, tilt)
            d.qpos[q_leg] = q0
            for _ in range(40):  # the leg's IK on the ball and the tilted orientation
                mujoco.mj_kinematics(m, d)
                mujoco.mj_comPos(m, d)
                mujoco.mj_jacSite(m, d, jp, jr, site)
                e_ball = balls[i, move["limb"]] - d.site_xpos[site]
                e_rot = np.zeros(3)
                mujoco.mju_subQuat(e_rot, target, site_quat(d, site))
                e_rot = d.site_xmat[site].reshape(3, 3) @ e_rot
                if np.linalg.norm(e_ball) < 5e-4 and np.linalg.norm(e_rot) < 5e-3:
                    break
                j = np.vstack([jp[:, v_leg], 0.1 * jr[:, v_leg]])
                step = j.T @ np.linalg.solve(j @ j.T + 1e-4 * np.eye(6), np.r_[e_ball, 0.1 * e_rot])
                d.qpos[q_leg] = np.clip(d.qpos[q_leg] + step, low, high)
            out_of_reach = np.linalg.norm(e_ball) > 2e-3 or np.linalg.norm(e_rot) > 0.05
            mujoco.mj_forward(m, d)
            deepest = 0.0
            for k in range(d.ncon):
                c = d.contact[k]
                on_foot = body in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2])
                if on_foot and ball_geom not in (c.geom1, c.geom2):
                    deepest = min(deepest, c.dist)
            cost[a, b] = OUT_OF_REACH_COST * out_of_reach - 1000 * deepest + TILT_COST * abs(tilt)

    change = np.abs(TILTS[None] - TILTS[:, None]) * TILT_CHANGE_COST
    none = int(np.argmin(np.abs(TILTS)))
    total = np.full(cost.shape, np.inf)
    total[0, none] = cost[0, none]
    back = np.zeros(cost.shape, dtype=int)
    for a in range(1, len(samples)):
        options = total[a - 1][None] + change
        back[a] = options.argmin(axis=1)
        total[a] = options.min(axis=1) + cost[a]
    path = [none]
    for a in range(len(samples) - 1, 0, -1):
        path.append(back[a, path[-1]])
    return samples, TILTS[path[::-1]]


# ------------------------------------------------------------
# Whole-body controller
# ------------------------------------------------------------


class Controller:
    def __init__(self, sim, planner, route, balls):
        self.sim, self.planner, self.balls = sim, planner, balls
        self.pos, self.quat, self.joints = route
        self.v_joints = np.gradient(self.joints, DT, axis=0)
        self.a_joints = np.gradient(smooth(self.v_joints), DT, axis=0)
        self.v_pos = np.gradient(self.pos, DT, axis=0)
        self.v_balls = np.gradient(balls, DT, axis=0)
        self.v_rot = np.zeros((len(self.pos), 3))
        for i in range(len(self.pos) - 1):
            mujoco.mju_subQuat(self.v_rot[i], self.quat[i + 1], self.quat[i])
        self.v_rot /= DT
        m = sim.model
        self.sites = [m.site(BALL_SITES[limb]).id for limb in LIMBS]
        self.actuators = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in JOINTS]
        self.mass = np.empty((m.nv, m.nv))
        self.jacobians = np.zeros((len(LIMBS), 3, m.nv))
        self.saturated = 0

    def planned_foot_quat(self, i, limb):
        """The foot's orientation in the plan at sample i."""
        m, d = self.planner.model, self.planner.data
        d.qpos[:3], d.qpos[3:7] = self.pos[i], self.quat[i]
        d.qpos[self.planner._qadr] = self.joints[i]
        mujoco.mj_kinematics(m, d)
        return site_quat(d, self.sites[limb])

    def step(self, i, alpha, held, moving=None, tilts=None):
        """Set the motors' torques for sample i + alpha, then take a physics step."""
        sim, m, d = self.sim, self.sim.model, self.sim.data
        va, nv = sim._vadr, m.nv
        j1 = min(i + 1, len(self.pos) - 1)
        mujoco.mj_forward(m, d)
        mujoco.mj_fullM(m, d, self.mass)
        for k, site in enumerate(self.sites):
            mujoco.mj_jacSite(m, d, self.jacobians[k], None, site)
        rows, targets = [], []

        def task(jacobian, acceleration, weight):
            rows.append(np.sqrt(weight) * jacobian)
            targets.append(np.sqrt(weight) * acceleration)

        def kd(kp):
            return 2.0 * np.sqrt(kp)

        trunk = np.zeros((6, nv))
        trunk[:6, :6] = np.eye(6)
        at = self.pos[i] + alpha * (self.pos[j1] - self.pos[i]) + TRUNK_OFFSET
        e_rot = np.zeros(3)
        mujoco.mju_subQuat(e_rot, self.quat[i], d.qpos[3:7])
        a_trunk = np.r_[
            KP_TRUNK * (at - d.qpos[:3]) + kd(KP_TRUNK) * (SPEED * self.v_pos[i] - d.qvel[:3]),
            KP_TRUNK * e_rot + kd(KP_TRUNK) * (SPEED * self.v_rot[i] - d.qvel[3:6]),
        ]
        task(trunk, a_trunk, W_TRUNK)

        for k in held:
            task(self.jacobians[k], -KD_HELD * (self.jacobians[k] @ d.qvel), W_HELD)

        if moving is not None:
            j, site = self.jacobians[moving], self.sites[moving]
            at = self.balls[i, moving] + alpha * (self.balls[j1, moving] - self.balls[i, moving])
            a_ball = KP_BALL * (at - d.site_xpos[site]) + kd(KP_BALL) * (
                SPEED * self.v_balls[i, moving] - j @ d.qvel
            )
            task(j, a_ball, W_BALL)

        if tilts is not None:
            site = self.sites[moving]
            j = np.zeros((3, nv))
            mujoco.mj_jacSite(m, d, None, j, site)
            tilt = np.interp(i + alpha, tilts[0], tilts[1])
            target = tilted(self.planned_foot_quat(i, moving), tilt)
            e_rot = np.zeros(3)
            mujoco.mju_subQuat(e_rot, target, site_quat(d, site))
            e_rot = d.site_xmat[site].reshape(3, 3) @ e_rot
            task(j, KP_FOOT * e_rot - kd(KP_FOOT) * (j @ d.qvel), W_FOOT)

        joints = np.zeros((len(JOINTS), nv))
        joints[np.arange(len(JOINTS)), va] = 1.0
        at = self.joints[i] + alpha * (self.joints[j1] - self.joints[i])
        a_joints = (
            SPEED**2 * self.a_joints[i]
            + KP_JOINT * (at - d.qpos[sim._qadr])
            + kd(KP_JOINT) * (SPEED * self.v_joints[i] - d.qvel[va])
        )
        task(joints, a_joints, W_JOINT)

        acceleration = np.linalg.lstsq(np.vstack(rows), np.concatenate(targets), rcond=None)[0]
        need = self.mass @ acceleration + d.qfrc_bias

        # The base has no motor: the held balls' forces make its rows, with the least joint torque
        contact = np.vstack([self.jacobians[k] for k in held])
        base, actuated = contact[:, :6].T, contact[:, va].T
        n = base.shape[1]
        kkt = np.block(
            [
                [actuated.T @ actuated + FORCE_REG * np.eye(n), base.T],
                [base, np.zeros((6, 6))],
            ]
        )
        rhs = np.r_[actuated.T @ need[va], need[:6]]
        forces = np.linalg.lstsq(kkt, rhs, rcond=None)[0][:n]
        tau = need[va] - actuated @ forces

        self.saturated += np.abs(tau).max() >= MAX_MOTOR_TORQUE
        d.ctrl[self.actuators] = np.clip(tau, -MAX_MOTOR_TORQUE, MAX_MOTOR_TORQUE)
        mujoco.mj_step(m, d)


# ------------------------------------------------------------
# Run in MuJoCo
# ------------------------------------------------------------


def climb(sim, planner, viewer=None):
    """The npz's moves in a row, stopping at the first one that fails. (moves made, how many, why
    it stopped)."""
    d = sim.data
    route = load_route()
    pos, quat, joints = route
    balls = np.array(
        [
            [planner.ball_positions_at(joints[i], pos[i], quat[i])[limb] for limb in LIMBS]
            for i in range(len(pos))
        ]
    )
    sockets = sim.wall.sockets()
    moves = find_moves(balls, sockets)
    controller = Controller(sim, planner, route, balls)

    physics_dt = sim.model.opt.timestep
    substeps = round(DT / physics_dt / SPEED)
    sync_every = round(DT / physics_dt)
    steps = 0

    def sync():
        nonlocal steps
        steps += 1
        if viewer is not None and steps % sync_every == 0:
            sim._sync_viewer()

    sim.reset(pos[0], quat[0], dict(zip(JOINTS, joints[0], strict=True)))
    for _ in range(round(SETTLE_TIME / physics_dt)):
        controller.step(0, 0.0, range(len(LIMBS)))
        sync()

    def hooked(limb):
        return sim.hooked()[LIMBS[limb]]

    for n, move in enumerate(moves):
        limb = move["limb"]
        name = LIMBS[limb]
        held = [k for k in range(len(LIMBS)) if k != limb]
        tilts = foot_tilts(planner, route, balls, move) if "foot" in name else None
        for i in range(move["i0"], move["i1"]):
            for sub in range(substeps):
                controller.step(i, sub / substeps, held, limb, tilts)
                sync()
            for k in held:
                if hooked(k) != move["before"][k]:
                    return n, len(moves), f"move {n} ({name}): {LIMBS[k]} left its hold"
        for _ in range(round(GRACE_TIME / DT)):
            if hooked(limb) == move["after"][limb]:
                break
            for _ in range(substeps):
                controller.step(move["i1"], 0.0, held, limb, tilts)
                sync()
        if hooked(limb) != move["after"][limb]:
            socket = sockets[move["after"][limb]]
            miss = 1000 * np.linalg.norm(d.site_xpos[controller.sites[limb]] - socket)
            return n, len(moves), f"move {n} ({name}): ball stopped {miss:.1f} mm from socket"
        print(f"move {n:2d} {name:<10} hooked, t = {d.time:.1f} s", flush=True)
    return len(moves), len(moves), "reached the end"


def main():
    wall = generate_wall(LEVEL, SEED)
    sim = ClimbingSim(wall, cameras=[])
    planner = ClimbingSim(wall, cameras=[])  # kinematics only, so as not to disturb the climb
    viewer = sim.view(realtime=REALTIME) if RENDER else None
    try:
        made, total, why = climb(sim, planner, viewer)
    finally:
        sim.close()
    print(f"{made} of {total} moves in a row: {why}")
    print("Final hooked:", sim.hooked())


if __name__ == "__main__":
    main()
