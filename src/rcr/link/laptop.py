"""The laptop's end of the link (rcr.link.LaptopLink). See README.md and protocol.py.

Commands (trajectory, start, stop) go over TCP and return once the robot has answered: when
send_trajectory returns, the robot has the whole trajectory, checked. The robot's state comes back
over UDP, received in a background thread; robot_state() returns the newest.
"""

from __future__ import annotations

import socket
import threading
import time

from rcr import io
from rcr.link import protocol as pr
from rcr.types import Pose, RobotState, Trajectory


class LaptopLink:
    """The link to one robot, at host name or address `robot` (e.g. "microban.local" or an IP)."""

    def __init__(self, robot: str, port: int = pr.COMMAND_PORT, timeout: float = 10.0) -> None:
        self.robot = robot
        self.port = port
        self.timeout = timeout  # [s] to connect, and for each command's answer
        self._tcp: socket.socket | None = None
        self._pose_port: int | None = None
        self._lock = threading.Lock()  # one command at a time
        # The robot's state, over UDP, as replies to our subscribes from this socket (protocol.py)
        self._udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._udp.bind(("", 0))
        self._udp.settimeout(0.2)
        self._state: RobotState | None = None
        self.states_received = 0
        self.states_lost = 0  # gaps in the robot's sequence numbers
        self._last_seq: int | None = None
        self._session = None
        self._pose_seq = 0
        self._closed = threading.Event()
        self._receiver = threading.Thread(target=self._receive_states, daemon=True)
        self._receiver.start()

    # --- set: laptop -> robot ------------------------------------------------------------------

    def send_trajectory(self, trajectory: Trajectory) -> None:
        """Send the trajectory; returns once the robot has it whole and checked (LinkError if
        not, e.g. while it is running one)."""
        answer = self._command(
            {"type": "trajectory", "trajectory": io.trajectory_to_dict(trajectory)}
        )
        if answer.get("samples") != len(trajectory.times):
            raise pr.LinkError(
                f"robot got {answer.get('samples')} samples of {len(trajectory.times)}"
            )

    def start(self) -> None:
        """Start the climb (the robot runs its trajectory). Returns once the robot has it."""
        self._command({"type": "start"})

    def stop(self) -> None:
        """Stop the climb. Returns once the robot has it."""
        self._command({"type": "stop"})

    def send_pose(self, pose: Pose) -> None:
        """Optional: the robot's pose from the cameras. Over UDP: never waits, may be lost."""
        if self._pose_port is None:
            self._command({"type": "ping"})  # connects, and learns the robot's pose port
        self._pose_seq += 1
        message = pr.datagram("pose", self._pose_seq, pr.pose_to_dict(pose))
        try:
            self._udp.sendto(message, (self.robot, self._pose_port))
        except OSError:
            pass  # the next pose replaces it

    # --- get: robot -> laptop ------------------------------------------------------------------

    def robot_state(self) -> RobotState | None:
        """The newest state received from the robot, or None."""
        return self._state

    def ping(self) -> float:
        """[s] the time for a command to reach the robot and its answer to come back."""
        start = time.perf_counter()
        self._command({"type": "ping"})
        return time.perf_counter() - start

    # --- the connection ------------------------------------------------------------------------

    def close(self) -> None:
        self._closed.set()
        with self._lock:
            if self._tcp is not None:
                self._tcp.close()
                self._tcp = None
        self._receiver.join()
        self._udp.close()

    def __enter__(self) -> LaptopLink:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _command(self, message: dict) -> dict:
        """Send a command and wait for the robot's answer; reconnects once if the connection
        was lost (the robot restarted, say)."""
        with self._lock:
            for attempt in (0, 1):
                fresh = self._tcp is None
                if fresh:
                    self._connect()
                try:
                    pr.send_message(self._tcp, message)
                    answer = pr.recv_message(self._tcp)
                except OSError as e:
                    answer, error = None, e
                else:
                    error = "the robot closed the connection"
                if answer is not None:
                    break
                self._tcp.close()
                self._tcp = None
                if fresh or attempt:
                    raise pr.LinkError(
                        f"no answer from the robot at {self.robot}:{self.port}: {error}"
                    )
        if not answer.get("ok"):
            raise pr.LinkError(f"the robot refused {message['type']!r}: {answer.get('error')}")
        return answer

    def _connect(self) -> None:
        try:
            tcp = socket.create_connection((self.robot, self.port), timeout=self.timeout)
        except OSError as e:
            raise pr.LinkError(
                f"cannot reach the robot at {self.robot}:{self.port} ({e}). Is "
                "`python -m rcr.link.raspberry_pi` running on the Pi, on the same network?"
            ) from e
        tcp.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        pr.send_message(tcp, {"type": "hello"})
        answer = pr.recv_message(tcp)
        if not answer or not answer.get("ok"):
            tcp.close()
            raise pr.LinkError(
                f"{self.robot}:{self.port} did not answer hello: not our robot link?"
            )
        self._pose_port = answer["pose_port"]
        self._tcp = tcp
        self._subscribe()

    def _subscribe(self) -> None:
        """Tell the robot where to send the state: wherever this datagram comes from."""
        try:
            self._udp.sendto(pr.datagram("subscribe", 0, {}), (self.robot, self._pose_port))
        except OSError:
            pass  # the next one, a second later

    def _receive_states(self) -> None:
        subscribed = 0.0
        while not self._closed.is_set():
            if self._pose_port is not None and time.monotonic() - subscribed > pr.SUBSCRIBE_EVERY:
                self._subscribe()
                subscribed = time.monotonic()
            try:
                data = self._udp.recv(65536)
            except socket.timeout:  # noqa: UP041 (TimeoutError from Python 3.10; the Pi may have 3.9)
                continue
            except OSError:
                return
            message = pr.parse_datagram(data)
            if message is None or message["type"] != "state":
                continue
            seq = message["seq"]
            if message.get("session") != self._session:  # the robot (re)started: count anew
                self._session, self._last_seq = message.get("session"), None
            if self._last_seq is not None and seq <= self._last_seq:
                continue  # late: older than what we have
            if self._last_seq is not None:
                self.states_lost += seq - self._last_seq - 1
            self._last_seq = seq
            self._state = pr.state_from_dict(message["state"])
            self.states_received += 1
