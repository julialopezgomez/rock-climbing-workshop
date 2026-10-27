"""The link, both sides on this machine over loopback (127.0.0.1): what the Pi and the laptop do
over Wi-Fi, without them. scripts/link_pi.py check does the same with the real Pi."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from rcr import io, link
from rcr.types import Pose, RobotState, Trajectory

ROOT = Path(__file__).resolve().parent.parent
TRAJECTORY = io.load_trajectory(ROOT / "data" / "examples" / "trajectory.json")
# What the robot gets: its part only (the planned angles and trunk poses stay on the laptop)
ROBOT_PART = Trajectory(TRAJECTORY.joint_names, TRAJECTORY.times, TRAJECTORY.positions)


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
    assert robot.trajectory() == ROBOT_PART
    assert robot.trajectories_received == 1


def test_one_call_sends_a_trajectory_file(robot):
    link.send_trajectory(ROOT / "data" / "examples" / "trajectory.json", "127.0.0.1", robot.port)
    assert robot.trajectory() == ROBOT_PART


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
    laptop.ping()  # connected: subscribes, so the robot learns where to send the state
    state = RobotState(
        time=1.5,
        joint_positions={"head_yaw": 0.1},
        joint_velocities={"head_yaw": 0.0},
        imu_quat=(1.0, 0.0, 0.0, 0.0),
        gyro=(0.0, 0.0, 0.1),
        base_pose=Pose((0.5, -0.1, 0.4), (1.0, 0.0, 0.0, 0.0)),
    )

    def arrived():
        robot.send_state(state)  # as the 50 Hz loop would, until the subscribe has come
        return laptop.robot_state() == state

    assert eventually(arrived)
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
        assert second.trajectory() == ROBOT_PART
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
        assert io.load_trajectory(saved) == ROBOT_PART
    finally:
        receiver.terminate()
        receiver.wait(timeout=10)


# --- Move by move: segments appended while the robot runs --------------------------------------


def cut(lo: int, hi: int):
    """Samples lo..hi-1 of the example: one segment of it."""
    part = link.segment(TRAJECTORY, lo)
    n = hi - lo
    part.times, part.positions = part.times[:n], part.positions[:n]
    part.planned_positions, part.trunk_poses = part.planned_positions[:n], part.trunk_poses[:n]
    return part


def test_segments_grow_the_trajectory_while_it_runs(robot, laptop):
    cuts = [0, 300, 600, len(TRAJECTORY.times)]
    laptop.send_trajectory(cut(0, 300), final=False)
    assert not robot.trajectory_complete()
    laptop.start()  # runs what it has; more comes
    for lo, hi in zip(cuts[1:-1], cuts[2:], strict=True):
        laptop.append_segment(cut(lo, hi), final=hi == cuts[-1])
    assert robot.trajectory() == ROBOT_PART  # the segments, joined, are the whole
    assert robot.trajectory_complete()
    with pytest.raises(link.LinkError, match="complete"):
        laptop.append_segment(cut(len(TRAJECTORY.times) - 1, len(TRAJECTORY.times)))


def test_a_segment_must_go_on_from_the_end(robot, laptop):
    with pytest.raises(link.LinkError, match="no trajectory"):
        laptop.append_segment(TRAJECTORY)
    laptop.send_trajectory(TRAJECTORY, final=False)
    with pytest.raises(link.LinkError, match="not after the trajectory's end"):
        laptop.append_segment(cut(10, 20))  # overlaps what it has
    laptop.finish()
    assert robot.trajectory_complete()


def short_trajectory(seconds: float):
    """The example's first `seconds`, the robot's part only."""
    n = sum(1 for t in TRAJECTORY.times if t <= seconds)
    return TRAJECTORY.__class__(
        TRAJECTORY.joint_names, TRAJECTORY.times[:n], TRAJECTORY.positions[:n]
    )


@pytest.fixture
def receiver(tmp_path):
    """The receiver (python -m rcr.link.raspberry_pi) in its own process: (port, its output)."""
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    with link.RobotLink(port=0, pose_port=0, host="127.0.0.1") as probe:  # two free ports
        port, pose_port = probe.port, probe.pose_port
    process = subprocess.Popen(
        [sys.executable, "-m", "rcr.link.raspberry_pi", "--port", str(port),
         "--pose-port", str(pose_port), "--save", str(tmp_path / "received.json")],
        env=env, stdout=subprocess.PIPE, text=True,
    )  # fmt: skip
    assert "listening" in process.stdout.readline()
    output = []
    yield port, output
    process.terminate()
    output.append(process.communicate(timeout=10)[0])


def link_pi(*args, **kwargs):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    command = [sys.executable, str(ROOT / "scripts" / "link_pi.py"), *map(str, args)]
    return subprocess.Popen(command, env=env, stdout=subprocess.PIPE, text=True, **kwargs)


def test_stream_follows_the_file_as_the_optimiser_saves_it(tmp_path, receiver):
    """stream sends the file, then each time it has grown (as after each solved move with
    --save) what is new; Ctrl-C ends it, and the robot has the whole of it."""
    port, _ = receiver
    file = tmp_path / "trajectory.json"
    io.save_trajectory(short_trajectory(4.0), file)  # one move so far
    stream = link_pi("stream", file, "--pi", "127.0.0.1", "--port", port)
    try:
        assert "following" in stream.stdout.readline()
        assert "started" in stream.stdout.readline()
        for seconds in (8.0, 12.0):  # two more moves solved and saved
            time.sleep(0.3)
            io.save_trajectory(short_trajectory(seconds), file)
            assert "more, to" in stream.stdout.readline()
    finally:
        stream.send_signal(signal.SIGINT)
        out = stream.communicate(timeout=10)[0]
    assert "ended: the robot has all" in out
    assert io.load_trajectory(tmp_path / "received.json") == short_trajectory(12.0)


def test_replay_with_the_receiver(tmp_path):
    """scripts/link_pi.py replay against the receiver: a 3 s trajectory in 1 s segments, one
    every 1.5 s, so the robot must hold at the end of each and still reach the end."""
    n = 31
    short = TRAJECTORY.__class__(
        joint_names=TRAJECTORY.joint_names,
        times=[0.1 * i for i in range(n)],
        positions=TRAJECTORY.positions[:n],
    )
    file = tmp_path / "short.json"
    io.save_trajectory(short, file)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    with link.RobotLink(port=0, pose_port=0, host="127.0.0.1") as probe:  # two free ports
        port, pose_port = probe.port, probe.pose_port
    receiver = subprocess.Popen(
        [sys.executable, "-m", "rcr.link.raspberry_pi", "--port", str(port),
         "--pose-port", str(pose_port), "--save", str(tmp_path / "received.json")],
        env=env, stdout=subprocess.PIPE, text=True,
    )  # fmt: skip
    try:
        assert "listening" in receiver.stdout.readline()
        run = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "link_pi.py"), "replay", str(file),
             "--pi", "127.0.0.1", "--port", str(port), "--segment", "1", "--every", "1.5"],
            env=env, capture_output=True, text=True, timeout=60,
        )  # fmt: skip
        assert run.returncode == 0, run.stdout + run.stderr
        assert "robot reached the end" in run.stdout
        held = float(run.stdout.split("it held ")[1].split(" s")[0])
        assert held > 0.5  # segments came slower than it played them: it waited, then went on
        assert io.load_trajectory(tmp_path / "received.json").times == short.times
    finally:
        receiver.terminate()
        output = receiver.communicate(timeout=10)[0]
    assert "waiting at" in output and "going on from" in output


def test_the_bundle_runs_on_its_own(tmp_path):
    """scripts/link_pi.py bundle: the folder alone (no repository) holds all the Pi needs."""
    out = tmp_path / "raspberry_pi"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "link_pi.py"), "bundle", "--out", str(out)],
        check=True, capture_output=True,
    )  # fmt: skip
    assert (out / "run.sh").exists() and (tmp_path / "raspberry_pi.zip").exists()
    probe = "import rcr.link.raspberry_pi, rcr.controller; print(rcr.link.raspberry_pi.__file__)"
    where = subprocess.run(
        [sys.executable, "-c", probe], cwd=out, env={**os.environ, "PYTHONPATH": str(out / "src")},
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    assert where.startswith(str(out))  # the bundle's own files, not the repository's
