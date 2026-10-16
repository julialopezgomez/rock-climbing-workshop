"""Link (both sides): the only thing that crosses laptop <-> robot during a run. See README.md.

Nothing time-critical crosses it: the 50 Hz loop lives on the robot.
RobotLink is Pi-safe and every method is non-blocking (it is called from the 50 Hz loop).
"""

from rcr.types import Pose, RobotState, Trajectory


class LaptopLink:
    def __init__(self, robot: str) -> None:
        self.robot = robot  # host name or address, e.g. "microban"

    def send_trajectory(self, trajectory: Trajectory) -> None:
        raise NotImplementedError

    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def send_pose(self, pose: Pose) -> None:
        """Optional: live robot pose from the cameras."""
        raise NotImplementedError

    def robot_state(self) -> RobotState | None:
        """Latest state received from the robot, or None."""
        raise NotImplementedError


class RobotLink:
    def trajectory(self) -> Trajectory | None:
        """The trajectory received from the laptop, or None if none yet."""
        raise NotImplementedError

    def run_requested(self) -> bool:
        """True between the laptop's start() and stop()."""
        raise NotImplementedError

    def pose(self) -> Pose | None:
        """Latest camera pose from the laptop, or None."""
        raise NotImplementedError

    def send_state(self, state: RobotState) -> None:
        raise NotImplementedError
