"""The link, both sides on this machine over loopback (127.0.0.1): what the Pi and the laptop do
over Wi-Fi, without them. scripts/link_pi.py check does the same with the real Pi."""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from rcr import io, link
from rcr.types import Pose, RobotState

ROOT = Path(__file__).resolve().parent.parent
TRAJECTORY = io.load_trajectory(ROOT / "data" / "examples" / "trajectory.json")


@pytest.fixture
def robot():
    with link.RobotLink(port=0, pose_port=0, host="127.0.0.1") as robot:
        yield robot


@pytest.fixture
def laptop(robot):
    with link.LaptopLink("127.0.0.1", robot.port, timeout=5.0) as laptop:
        yield laptop


def eventually(condition, seconds=2.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.01)
    return False


def test_the_trajectory_arrives_whole(robot, laptop):
    assert robot.trajectory() is None
    laptop.send_trajectory(TRAJECTORY)  # returns once the robot has it
    assert robot.trajectory() == TRAJECTORY
    assert robot.trajectories_received == 1


def test_one_call_sends_a_trajectory_file(robot):
    link.send_trajectory(ROOT / "data" / "examples" / "trajectory.json", "127.0.0.1", robot.port)
    assert robot.trajectory() == TRAJECTORY


def test_start_and_stop(robot, laptop):
    with pytest.raises(link.LinkError, match="no trajectory"):
        laptop.start()
    laptop.send_trajectory(TRAJECTORY)
    laptop.start()
    assert robot.run_requested()
    with pytest.raises(link.LinkError, match="stop before"):
        laptop.send_trajectory(TRAJECTORY)  # never swapped under a running climb
    laptop.stop()
    assert not robot.run_requested()


def test_a_broken_trajectory_is_refused(robot, laptop):
    bad = io.trajectory_from_dict(io.trajectory_to_dict(TRAJECTORY))
    bad.times = list(reversed(bad.times))
    with pytest.raises(link.LinkError, match="times must increase"):
        laptop.send_trajectory(bad)
    assert robot.trajectory() is None


def test_state_comes_back_and_pose_goes_out(robot, laptop):
    laptop.ping()  # connected: the robot now knows where to send the state
    state = RobotState(
        time=1.5,
        joint_positions={"head_yaw": 0.1},
        joint_velocities={"head_yaw": 0.0},
        imu_quat=(1.0, 0.0, 0.0, 0.0),
        gyro=(0.0, 0.0, 0.1),
        base_pose=Pose((0.5, -0.1, 0.4), (1.0, 0.0, 0.0, 0.0)),
    )
    for _ in range(5):
        robot.send_state(state)
    assert eventually(lambda: laptop.robot_state() == state)
    pose = Pose((0.6, -0.1, 0.5), (1.0, 0.0, 0.0, 0.0))
    laptop.send_pose(pose)
    assert eventually(lambda: robot.pose() == pose)


def test_the_robot_side_never_waits(robot):
    """Called from the 50 Hz loop: with no laptop at all, every call returns at once."""
    start = time.perf_counter()
    for _ in range(1000):
        robot.trajectory(), robot.run_requested(), robot.pose()
        robot.send_state(RobotState(0.0, {}, {}, (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    assert time.perf_counter() - start < 0.5


def test_the_laptop_reconnects_after_the_robot_restarts():
    with link.RobotLink(port=0, pose_port=0, host="127.0.0.1") as first:
        port = first.port
        laptop = link.LaptopLink("127.0.0.1", port, timeout=5.0)
        laptop.ping()
    with link.RobotLink(port=port, pose_port=0, host="127.0.0.1") as second:
        laptop.send_trajectory(TRAJECTORY)  # the old connection is gone: reconnects once
        assert second.trajectory() == TRAJECTORY
    laptop.close()


def test_no_robot_says_what_to_do():
    with link.RobotLink(port=0, pose_port=0, host="127.0.0.1") as robot:
        port = robot.port
    with pytest.raises(link.LinkError, match="raspberry_pi"):  # nothing listens there now
        link.send_trajectory(TRAJECTORY, "127.0.0.1", port)


def test_the_receiver_and_the_check_script(tmp_path):
    """What we run on the Pi (python -m rcr.link.raspberry_pi) and on the laptop
    (scripts/link_pi.py check), talking to each other."""
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    saved = tmp_path / "received.json"
    with link.RobotLink(port=0, pose_port=0, host="127.0.0.1") as probe:  # two free ports
        port, pose_port = probe.port, probe.pose_port
    receiver = subprocess.Popen(
        [sys.executable, "-m", "rcr.link.raspberry_pi", "--port", str(port),
         "--pose-port", str(pose_port), "--save", str(saved)],
        env=env, stdout=subprocess.PIPE, text=True,
    )  # fmt: skip
    try:
        assert "listening" in receiver.stdout.readline()
        check = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "link_pi.py"), "check", "--pi", "127.0.0.1",
             "--port", str(port), "--seconds", "1", "--pings", "5"],
            env=env, capture_output=True, text=True, timeout=60,
        )  # fmt: skip
        assert check.returncode == 0, check.stdout + check.stderr
        assert "link check passed" in check.stdout
        assert io.load_trajectory(saved) == TRAJECTORY
    finally:
        receiver.terminate()
        receiver.wait(timeout=10)
