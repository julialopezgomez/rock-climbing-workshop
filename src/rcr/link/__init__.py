"""Link (both sides): the only thing that crosses laptop <-> robot during a run. See README.md.

Nothing time-critical crosses it: the 50 Hz loop lives on the robot. TCP for commands (trajectory,
start, stop: they arrive whole or raise), UDP for the live state and pose (newest wins) [Ours, D5].

    from rcr import link
    link.send_trajectory("data/examples/trajectory.json", "172.20.10.8")   # one call

Move by move: LaptopLink.send_trajectory(first, final=False), then append_segment(next) for each
next one, also while the robot runs; it holds at the end of what it has until more comes.

Laptop: LaptopLink (laptop.py). Robot: RobotLink (raspberry_pi/), Pi-safe, every method
non-blocking (it is called from the 50 Hz loop). The wire format: protocol.py.
"""

from __future__ import annotations

from pathlib import Path

from rcr import io
from rcr.link.laptop import LaptopLink
from rcr.link.protocol import COMMAND_PORT, POSE_PORT, LinkError
from rcr.link.raspberry_pi import RobotLink
from rcr.types import Trajectory

__all__ = [
    "COMMAND_PORT",
    "POSE_PORT",
    "LaptopLink",
    "LinkError",
    "RobotLink",
    "segment",
    "send_trajectory",
]


def send_trajectory(
    trajectory: Trajectory | Path | str, robot: str, port: int = COMMAND_PORT
) -> None:
    """Send a trajectory, or a trajectory file, to the robot at `robot` (address or host name).
    Returns once the robot has it whole and checked; raises LinkError saying why if not."""
    if not isinstance(trajectory, Trajectory):
        trajectory = io.load_trajectory(trajectory)
    with LaptopLink(robot, port) as link:
        link.send_trajectory(trajectory)


def segment(trajectory: Trajectory, start: int) -> Trajectory:
    """The samples from index `start` on: what is new in a trajectory that grew since `start`
    samples were sent, to send with LaptopLink.append_segment."""

    def tail(rows):
        return None if rows is None else rows[start:]

    return Trajectory(
        joint_names=trajectory.joint_names,
        times=trajectory.times[start:],
        positions=trajectory.positions[start:],
        planned_positions=tail(trajectory.planned_positions),
        trunk_poses=tail(trajectory.trunk_poses),
    )
