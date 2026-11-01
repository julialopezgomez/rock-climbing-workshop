"""Controller (robot, 50 Hz): trajectory + robot state -> joint targets. See README.md.

Pi-safe: stdlib, numpy, and later placo / onnxruntime only. Never mujoco or opencv.
step() mirrors Microban's Move.step(); it does not know whether it runs on the sim or the robot.
"""

from rcr.types import JointTargets, RobotState, Trajectory


class Controller:
    def __init__(self, trajectory: Trajectory) -> None:
        self.trajectory = trajectory

    def step(self, state: RobotState) -> JointTargets:
        """Called every 20 ms with the latest state; must return well within the tick."""
        raise NotImplementedError
