"""Sim (laptop): a backend with the same role as the robot. See README.md.

Receives joint targets, returns robot state. First our emulation of the official simulator,
later the official one: only the inside of Sim changes.

Inside: a MuJoCo scene built from the wall model (scene.py), Microban's torque
actuators driven by a servo-like PD, and an auto-grip rule that makes a ball
sit in a cup.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from rcr.sim.attach import WeldAttachment
from rcr.sim.geometry import CupHold
from rcr.sim.scene import ROBOT_PREFIX, SERVO_TORQUE_LIMIT, SceneConfig, build_scene
from rcr.types import CONTROL_HZ, JointTargets, Pose, RobotState, Stance, WallModel

CONTROL_DT = 1.0 / CONTROL_HZ


@dataclass
class ServoGains:
    """Servo-like PD with the real torque ceiling.

    Microban's own stiffness (0.886 N.m/rad) is too soft to track a climbing
    pose: holding an arm costs ~0.5 N.m, i.e. ~0.5 rad of droop. kp=3 tracks to
    about 15 mm. That gap is a hardware finding, not a simulation artefact.
    """

    kp: float = 3.0
    kd: float = 0.05
    torque_limit: float = SERVO_TORQUE_LIMIT


@dataclass
class GripRule:
    """When a ball counts as sitting in a cup [Ours].

    The controller only sends joint targets, so grip has to emerge. A ball
    within `capture` of a cup's grasp point, moving slowly, drops in. It leaves
    when the *commanded* pose wants it more than `release` away: the controller
    is lifting that limb. Force cannot be the signal here — four welds generate
    tens of newtons of internal force, while a servo can pull with about six.
    """

    capture: float = 0.020  # [m] how close a ball must be to drop into a cup
    settle_speed: float = 0.25  # [m/s] below which a ball can be captured


class Sim:
    def __init__(
        self,
        wall: WallModel,
        cfg: SceneConfig | None = None,
        gains: ServoGains | None = None,
        grip: GripRule | None = None,
        out: Path | None = Path("build/scene.xml"),
    ) -> None:
        self.wall = wall
        self.cfg = cfg or SceneConfig()
        self.gains = gains or ServoGains()
        self.grip = grip or GripRule()
        self.model = build_scene(wall, self.cfg, out=out).compile()
        self.data = mujoco.MjData(self.model)
        self.attachment = WeldAttachment(self.model, self.data)

        self._act_qpos = np.array(
            [self.model.jnt_qposadr[self.model.actuator(i).trnid[0]] for i in range(self.model.nu)]
        )
        self._act_qvel = np.array(
            [self.model.jnt_dofadr[self.model.actuator(i).trnid[0]] for i in range(self.model.nu)]
        )
        self.joint_names = [
            self.model.actuator(i).name.removeprefix(ROBOT_PREFIX) for i in range(self.model.nu)
        ]
        self._target = np.zeros(self.model.nu)
        self._scratch = mujoco.MjData(self.model)  # forward kinematics of the commanded pose
        self._intent: dict[str, str | None] = {}
        mujoco.mj_forward(self.model, self.data)

    # -- the contract ----------------------------------------------------
    def reset(self) -> RobotState:
        """Place the robot at wall.start and return its state.

        Needs kinematics we do not have yet: something must turn wall.start
        into a pose. Write that (an IK, or a recorded pose), then call
        `place()` with the result and `hold_start()` to close the grips.
        """
        raise NotImplementedError(
            "Sim.reset needs a pose for wall.start: solve it, then call "
            "Sim.place(joints, base_pos, base_quat) and Sim.hold_start()"
        )

    def place(
        self,
        joints: dict[str, float],
        base_pos: np.ndarray | tuple[float, float, float],
        base_quat_wxyz: np.ndarray | tuple[float, float, float, float],
    ) -> RobotState:
        """Teleport the robot onto a pose (setup only, never mid-run)."""
        self.data.qpos[0:3] = base_pos
        self.data.qpos[3:7] = base_quat_wxyz
        for i, name in enumerate(self.joint_names):
            self.data.qpos[self._act_qpos[i]] = joints.get(name, 0.0)
        self.data.qvel[:] = 0.0
        self._target = self.joint_angles().copy()
        mujoco.mj_forward(self.model, self.data)
        return self.state()

    def hold_start(self) -> None:
        """Grip the cups of wall.start wherever the balls currently are."""
        self._intent = dict(self.wall.start.contacts)
        for limb, hold_id in self.wall.start.contacts.items():
            if hold_id:
                self.attachment.attach(limb, hold_id)
        mujoco.mj_forward(self.model, self.data)

    def step(self, targets: JointTargets) -> RobotState:
        """Apply the targets, advance one control tick (20 ms), return the new state."""
        for i, name in enumerate(self.joint_names):
            if name in targets.positions:
                self._target[i] = targets.positions[name]
        self.data.ctrl[:] = self._servo_torque()
        for _ in range(int(round(CONTROL_DT / self.model.opt.timestep))):
            mujoco.mj_step(self.model, self.data)
        self._update_grip()
        return self.state()

    def state(self) -> RobotState:
        q, qd = self.joint_angles(), self.joint_velocities()
        trunk = self.data.body(f"{ROBOT_PREFIX}trunk")
        return RobotState(
            time=float(self.data.time),
            joint_positions={n: float(v) for n, v in zip(self.joint_names, q, strict=True)},
            joint_velocities={n: float(v) for n, v in zip(self.joint_names, qd, strict=True)},
            imu_quat=tuple(float(v) for v in trunk.xquat),
            gyro=tuple(float(v) for v in self.data.qvel[3:6]),
            base_pose=Pose(
                position=tuple(float(v) for v in trunk.xpos),
                quat=tuple(float(v) for v in trunk.xquat),
            ),
        )

    # -- grip -------------------------------------------------------------
    def grasp_targets(self, stance: Stance) -> dict[str, np.ndarray]:
        """Where each limb's ball sits for this stance, in the wall frame."""
        cup: CupHold = self.cfg.hold
        return {
            limb: np.array(cup.grasp_point(self.hold(hold_id).position, self.cfg.limb.ball_radius))
            for limb, hold_id in stance.contacts.items()
            if hold_id
        }

    def hold(self, hold_id: str):
        for h in self.wall.holds:
            if h.id == hold_id:
                return h
        raise KeyError(hold_id)

    def expect(self, stance: Stance) -> None:
        """Tell the sim which cups the plan wants held from now on [Ours].

        Our grip is a weld, so something has to decide when a ball leaves a cup.
        Inferring it from joint targets does not work: the commanded pose and
        the welded contacts disagree by more than the signal (see README.md).
        The runner knows the plan, so it says. A limb whose hold changed lets go
        at once; the new cup only catches when the ball actually arrives.
        """
        self._intent = dict(stance.contacts)
        for limb, hold_id in self._intent.items():
            if self.attachment.is_attached(limb) and self.attachment.held_hold(limb) != hold_id:
                self.attachment.detach(limb)

    def _update_grip(self) -> None:
        """Capture a ball that has arrived in the cup the plan wants it in."""
        for limb, hold_id in self._intent.items():
            if hold_id is None or self.attachment.is_attached(limb):
                continue
            if np.linalg.norm(self._tip_velocity(limb)) > self.grip.settle_speed:
                continue
            tip = self.site_pos(f"{ROBOT_PREFIX}{limb}_tip")
            grasp = np.array(
                self.cfg.hold.grasp_point(self.hold(hold_id).position, self.cfg.limb.ball_radius)
            )
            if np.linalg.norm(tip - grasp) <= self.grip.capture:
                self.attachment.attach(limb, hold_id)

    def _commanded_tips(self) -> dict[str, np.ndarray]:
        """Where the balls would be if the robot tracked its targets exactly."""
        self._scratch.qpos[:] = self.data.qpos
        self._scratch.qpos[self._act_qpos] = self._target
        mujoco.mj_kinematics(self.model, self._scratch)
        return {
            limb: self._scratch.site(f"{ROBOT_PREFIX}{limb}_tip").xpos.copy()
            for limb in self.attachment.limbs
        }

    def _tip_velocity(self, limb: str) -> np.ndarray:
        result = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_SITE,
            self.model.site(f"{ROBOT_PREFIX}{limb}_tip").id,
            result,
            False,
        )
        return result[3:]

    # -- helpers used by the optimiser and the scripts ---------------------
    def joint_angles(self) -> np.ndarray:
        return self.data.qpos[self._act_qpos]

    def joint_velocities(self) -> np.ndarray:
        return self.data.qvel[self._act_qvel]

    def joint_dict(self) -> dict[str, float]:
        return dict(zip(self.joint_names, (float(v) for v in self.joint_angles()), strict=True))

    def base_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.data.qpos[0:3].copy(), self.data.qpos[3:7].copy()

    def site_pos(self, name: str) -> np.ndarray:
        return self.data.site(name).xpos.copy()

    def torque_margin(self) -> float:
        return float(np.abs(self.data.actuator_force).max() / self.gains.torque_limit)

    def wall_penetration(self) -> float:
        worst = 0.0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            names = {self.model.geom(c.geom1).name, self.model.geom(c.geom2).name}
            if "wall_board" in names:
                worst = max(worst, -float(c.dist))
        return worst

    def height_climbed(self) -> float:
        """Highest hold currently held [Rules §8]."""
        heights = [
            self.hold(h).position[2]
            for h in (self.attachment.held_hold(limb) for limb in self.attachment.limbs)
            if h
        ]
        return max(heights, default=0.0)

    def _servo_torque(self) -> np.ndarray:
        err = self._target - self.joint_angles()
        tau = self.gains.kp * err - self.gains.kd * self.joint_velocities()
        return np.clip(tau, -self.gains.torque_limit, self.gains.torque_limit)
