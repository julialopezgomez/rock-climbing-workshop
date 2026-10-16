"""Sim (laptop): a backend with the same role as the robot. See README.md.

Receives joint targets, returns robot state. First our emulation of the official simulator,
later the official one: only the inside of Sim changes.
"""

from rcr.types import JointTargets, RobotState, WallModel


class Sim:
    def __init__(self, wall: WallModel) -> None:
        self.wall = wall

    def reset(self) -> RobotState:
        """Place the robot at wall.start and return its state."""
        raise NotImplementedError

    def step(self, targets: JointTargets) -> RobotState:
        """Apply the targets, advance one control tick (20 ms), return the new state."""
        raise NotImplementedError
