"""Controller (robot, 50 Hz): trajectory + robot state -> joint targets. See README.md.

Pi-safe: stdlib, numpy, and later placo / onnxruntime only. Never mujoco or opencv.
step() mirrors Microban's Move.step(); it does not know whether it runs on the sim or the robot.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rcr.controller import tracking
from rcr.types import JointTargets, RobotState, Trajectory


class Controller:
    """Playback of the trajectory's feedforward, plus a policy's residual correction if given.

    policy: a .onnx file from scripts/train_controller.py, with its .json (alpha, servo gain kp_fw,
    joint order and limits) next to it; the robot must run at that kp_fw. None plays the
    trajectory back.
    """

    def __init__(self, trajectory: Trajectory, policy: str | Path | None = None) -> None:
        self.trajectory = trajectory
        self.reference = tracking.reference(trajectory)
        self.policy = None
        if policy is not None:  # the policy's .json says at which servo gain it was trained
            if trajectory.planned_positions is None:
                raise ValueError("a policy tracks planned_positions: the trajectory has none")
            import onnxruntime

            meta = json.loads(Path(policy).with_suffix(".json").read_text())
            if tuple(meta["joints"]) != tracking.JOINTS:
                raise ValueError(f"{policy} was trained with another joint order")
            if meta.get("action") != tracking.ACTION:
                raise ValueError(f"{policy} was trained with another action (not residual)")
            self.alpha = meta["alpha"]
            self.reference = tracking.reference(trajectory, meta.get("kp_fw", tracking.KP_FW))
            self.low, self.high = np.array(meta["low"]), np.array(meta["high"])
            self.policy = onnxruntime.InferenceSession(str(policy))

    def step(self, state: RobotState) -> JointTargets:
        """Called every 20 ms with the latest state; must return well within the tick."""
        q_target, planned = tracking.at(self.reference, state.time)
        if self.policy is not None:
            q = [state.joint_positions[n] for n in tracking.JOINTS]
            dq = [state.joint_velocities[n] for n in tracking.JOINTS]
            ahead = tracking.at(self.reference, state.time + tracking.LOOKAHEAD_S)[1]
            obs = tracking.observation(planned, ahead, q, dq, state.imu_quat, state.gyro)
            action = self.policy.run(None, {"obs": obs[None]})[0][0]
            q_target = tracking.targets(q_target, action, self.alpha, self.low, self.high)
        return JointTargets(dict(zip(tracking.JOINTS, map(float, q_target), strict=True)))
