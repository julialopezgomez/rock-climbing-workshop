"""Sim (laptop): a backend with the same role as the robot. See README.md.

Receives joint targets, returns robot state. First our emulation of the official simulator,
later the official one: only the inside of Sim changes.

Inside: a MuJoCo scene built from the wall model (scene.py), Microban's torque
actuators driven by a servo-like PD, a weld grip that makes a ball sit in a cup
(attach.py), and a start pose solved for wall.start (start.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from rcr.sim import start
from rcr.sim.attach import WeldAttachment
from rcr.sim.geometry import CupHold
from rcr.sim.scene import ROBOT_PREFIX, SERVO_TORQUE_LIMIT, SceneConfig, build_scene
from rcr.sim.start import NEUTRAL_POSE
from rcr.types import CONTROL_HZ, JointTargets, Pose, RobotState, Stance, WallModel

CONTROL_DT = 1.0 / CONTROL_HZ
SETTLE_TICKS = 25  # [ticks] reset() lets the servos take the load for 0.5 s before t = 0 [Ours]


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

    A ball that the plan wants on a cup (see Sim.expect) drops in once it is
    within `capture` of the cup's grasp point and moving slowly.
    """

    capture: float = 0.020  # [m] how close a ball must be to drop into a cup
    settle_speed: float = 0.25  # [m/s] below which a ball can be captured


@dataclass
class Score:
    """The run's score so far [Rules §8]."""

    height: float = 0.0  # [m] highest hold gripped so far (z of its reference point)
    top_time: float | None = None  # [s] when the wall's highest hold was first gripped


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
        self._index = {name: i for i, name in enumerate(self.joint_names)}
        self._neutral = np.array([NEUTRAL_POSE[name] for name in self.joint_names])
        self._target = self._neutral.copy()
        self._intent: dict[str, str | None] = {}
        self._top_z = max(h.position[2] for h in wall.holds)
        self.score = Score()
        self._renderer: mujoco.Renderer | None = None
        mujoco.mj_forward(self.model, self.data)

    # -- the contract ----------------------------------------------------
    def reset(self, stance: Stance | None = None) -> RobotState:
        """Hang the robot on a stance (default wall.start), let it settle, restart the clock.

        Free limbs (None) are left where the start-pose solve puts them. Raises ValueError if
        the stance is out of reach.
        """
        stance = stance or self.wall.start
        targets = self.grasp_targets(stance)
        sites = {limb: self.model.site(f"{ROBOT_PREFIX}{limb}_tip").id for limb in targets}
        qpos = start.solve(
            self.model, targets, sites, self._act_qpos, self._act_qvel, self._neutral
        )
        joints = {name: float(qpos[self._act_qpos[i]]) for i, name in enumerate(self.joint_names)}
        self.place(joints, qpos[0:3], qpos[3:7])
        self.hold_start(stance)
        for _ in range(SETTLE_TICKS):
            self._tick()
        self.data.time = 0.0
        self.score = Score()
        self._update_score()
        return self.state()

    def step(self, targets: JointTargets) -> RobotState:
        """Apply the targets, advance one control tick (20 ms), return the new state.

        Joints absent from `targets` go to Microban's neutral pose (INTERFACES.md).
        """
        unknown = targets.positions.keys() - self._index.keys()
        if unknown:
            raise KeyError(f"unknown joints: {sorted(unknown)}")
        self._target = self._neutral.copy()
        for name, value in targets.positions.items():
            self._target[self._index[name]] = value
        self._tick()
        return self.state()

    def state(self) -> RobotState:
        q, qd = self.joint_angles(), self.joint_velocities()
        trunk = self.data.body(f"{ROBOT_PREFIX}trunk")
        return RobotState(
            time=float(self.data.time),
            joint_positions={n: float(v) for n, v in zip(self.joint_names, q, strict=True)},
            joint_velocities={n: float(v) for n, v in zip(self.joint_names, qd, strict=True)},
            # Microban's IMU site sensors [Microban robot.xml]: orientation in the world
            # (= wall) frame, angular velocity in the IMU frame.
            imu_quat=tuple(float(v) for v in self._sensor("orientation")),
            gyro=tuple(float(v) for v in self._sensor("angular-velocity")),
            base_pose=Pose(
                position=tuple(float(v) for v in trunk.xpos),
                quat=tuple(float(v) for v in trunk.xquat),
            ),
        )

    # -- setup -------------------------------------------------------------
    def place(
        self,
        joints: dict[str, float],
        base_pos: np.ndarray | tuple[float, float, float],
        base_quat_wxyz: np.ndarray | tuple[float, float, float, float],
    ) -> RobotState:
        """Teleport the robot onto a pose, all grips open (setup only, never mid-run)."""
        mujoco.mj_resetData(self.model, self.data)
        self.attachment.detach_all()
        self._intent = {}
        self.data.qpos[0:3] = base_pos
        self.data.qpos[3:7] = base_quat_wxyz
        for i, name in enumerate(self.joint_names):
            self.data.qpos[self._act_qpos[i]] = joints.get(name, NEUTRAL_POSE[name])
        self._target = self.joint_angles().copy()
        mujoco.mj_forward(self.model, self.data)
        return self.state()

    def hold_start(self, stance: Stance | None = None) -> None:
        """Grip the cups of a stance (default wall.start) wherever the balls currently are."""
        stance = stance or self.wall.start
        self._intent = dict(stance.contacts)
        for limb, hold_id in stance.contacts.items():
            if hold_id:
                self.attachment.attach(limb, hold_id)
        mujoco.mj_forward(self.model, self.data)

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

    def _update_score(self) -> None:
        """Highest hold touched, and the time the top hold was reached [Rules §8]."""
        for limb in self.attachment.attached_limbs():
            z = self.hold(self.attachment.held_hold(limb)).position[2]
            self.score.height = max(self.score.height, z)
            if z >= self._top_z and self.score.top_time is None:
                self.score.top_time = float(self.data.time)

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

    # -- cameras ------------------------------------------------------------
    def images(self, width: int = 1280, height: int = 960) -> dict[str, np.ndarray]:
        """RGB images (height x width x 3, uint8) from the scene's cameras, by camera name."""
        if self._renderer is None or (self._renderer.width, self._renderer.height) != (
            width,
            height,
        ):
            self._renderer = mujoco.Renderer(self.model, height, width)
        out = {}
        for i in range(self.model.ncam):
            self._renderer.update_scene(self.data, camera=i)
            out[self.model.camera(i).name] = self._renderer.render().copy()
        return out

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
        """Peak servo torque as a fraction of the limit (1.0 = saturated)."""
        return float(np.abs(self.data.actuator_force).max() / self.gains.torque_limit)

    def wall_penetration(self) -> float:
        worst = 0.0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            names = {self.model.geom(c.geom1).name, self.model.geom(c.geom2).name}
            if "wall_board" in names:
                worst = max(worst, -float(c.dist))
        return worst

    def self_penetration(self) -> float:
        """Deepest overlap [m] between two parts of the robot (e.g. crossed legs)."""
        worst = 0.0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            bodies = (self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2])
            if all(self.model.body(b).name.startswith(ROBOT_PREFIX) for b in bodies):
                worst = max(worst, -float(c.dist))
        return worst

    # -- internals ------------------------------------------------------------
    def _tick(self) -> None:
        """One 20 ms control tick; the servo P loop runs every physics step, as on the servo."""
        for _ in range(int(round(CONTROL_DT / self.model.opt.timestep))):
            self.data.ctrl[:] = self._servo_torque()
            mujoco.mj_step(self.model, self.data)
        self._update_grip()
        self._update_score()

    def _sensor(self, name: str) -> np.ndarray:
        return self.data.sensor(f"{ROBOT_PREFIX}{name}").data

    def _servo_torque(self) -> np.ndarray:
        err = self._target - self.joint_angles()
        tau = self.gains.kp * err - self.gains.kd * self.joint_velocities()
        return np.clip(tau, -self.gains.torque_limit, self.gains.torque_limit)
