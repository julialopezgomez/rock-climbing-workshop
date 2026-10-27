"""The robot's end of the link (rcr.link.RobotLink), on the Pi. See ../README.md, ../protocol.py.

The network runs in background threads; every method here only reads or writes what they share,
so none of them ever waits: they are called from the 50 Hz loop.
"""

from __future__ import annotations

import socket
import threading
import time

from rcr import io
from rcr.link import protocol as pr
from rcr.types import Pose, RobotState, Trajectory


class RobotLink:
    """Listens for the laptop on `port` (TCP) and `pose_port` (UDP); 0 picks any free port."""

    def __init__(
        self, port: int = pr.COMMAND_PORT, pose_port: int = pr.POSE_PORT, host: str = ""
    ) -> None:
        self._lock = threading.Lock()
        self._trajectory: Trajectory | None = None
        self._run = False
        self._pose: Pose | None = None
        self._laptop: tuple[str, int] | None = None  # where to send the state (UDP)
        self._seq = 0
        self._session = time.time_ns() // 1_000_000  # tells the laptop when the robot restarted
        self._closed = threading.Event()
        self.trajectories_received = 0  # counts, so a loop can see that a new one came
        self._connections: set[socket.socket] = set()
        self._server = socket.create_server((host, port))  # SO_REUSEADDR: restarts at once
        self._server.settimeout(0.2)
        self.port = self._server.getsockname()[1]
        self._udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._udp.bind((host, pose_port))
        self._udp.settimeout(0.2)
        self.pose_port = self._udp.getsockname()[1]
        self._threads = [
            threading.Thread(target=self._accept, daemon=True),
            threading.Thread(target=self._receive_poses, daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    # --- get: laptop -> robot (never waits) ----------------------------------------------------

    def trajectory(self) -> Trajectory | None:
        """The trajectory received from the laptop, or None if none yet."""
        return self._trajectory

    def run_requested(self) -> bool:
        """True between the laptop's start() and stop()."""
        return self._run

    def pose(self) -> Pose | None:
        """Latest camera pose from the laptop, or None."""
        return self._pose

    def laptop(self) -> tuple[str, int] | None:
        """(address, state port) of the laptop connected last, or None."""
        return self._laptop

    # --- set: robot -> laptop (never waits) ----------------------------------------------------

    def send_state(self, state: RobotState) -> None:
        """Send the state to the laptop over UDP, if one has connected. Never waits; a lost one
        is replaced by the next."""
        laptop = self._laptop
        if laptop is None:
            return
        self._seq += 1
        message = pr.datagram("state", self._seq, pr.state_to_dict(state), self._session)
        try:
            self._udp.sendto(message, laptop)
        except OSError:
            pass  # the network is down or the buffer full: drop it

    # --- the connection ------------------------------------------------------------------------

    def close(self) -> None:
        """Stop listening and drop every connection: a closed link answers nothing more."""
        self._closed.set()
        with self._lock:
            for connection in self._connections:
                try:
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
        for thread in self._threads:
            thread.join()
        self._server.close()
        self._udp.close()

    def __enter__(self) -> RobotLink:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _accept(self) -> None:
        while not self._closed.is_set():
            try:
                connection, address = self._server.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            threading.Thread(target=self._serve, args=(connection, address), daemon=True).start()

    def _serve(self, connection: socket.socket, address: tuple) -> None:
        """One laptop connection, kept so that close() can drop it."""
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # Blocking, no timeout: a message is read whole however slow the Wi-Fi is (a timeout
        # mid-message would lose its place in the stream); close() unblocks it by a shutdown.
        connection.settimeout(None)
        with self._lock:
            if self._closed.is_set():
                connection.close()
                return
            self._connections.add(connection)
        try:
            self._answer(connection, address)
        finally:
            with self._lock:
                self._connections.discard(connection)
            connection.close()

    def _answer(self, connection: socket.socket, address: tuple) -> None:
        """Answer each command on this connection until it closes, or this link does."""
        while not self._closed.is_set():
            try:
                message = pr.recv_message(connection)
            except (OSError, ValueError, pr.LinkError):
                return
            if message is None or self._closed.is_set():
                return
            try:
                answer = {"type": "ack", "ok": True, **self._handle(message, address)}
            except Exception as e:  # a bad command is refused, never fatal
                answer = {"type": "ack", "ok": False, "error": f"{type(e).__name__}: {e}"}
            try:
                pr.send_message(connection, answer)
            except OSError:
                return

    def _handle(self, message: dict, address: tuple) -> dict:
        kind = message.get("type")
        if kind == "hello":
            self._laptop = (address[0], int(message["state_port"]))
            return {"pose_port": self.pose_port}
        if kind == "trajectory":
            trajectory = io.trajectory_from_dict(message["trajectory"])
            _check(trajectory)
            with self._lock:
                if self._run:
                    raise RuntimeError("running: stop before sending a new trajectory")
                self._trajectory = trajectory
                self.trajectories_received += 1
            return {"samples": len(trajectory.times), "duration": trajectory.times[-1]}
        if kind == "start":
            with self._lock:
                if self._trajectory is None:
                    raise RuntimeError("no trajectory yet")
                self._run = True
            return {}
        if kind == "stop":
            self._run = False
            return {}
        if kind == "ping":
            return {}
        raise ValueError(f"unknown message type {kind!r}")

    def _receive_poses(self) -> None:
        while not self._closed.is_set():
            try:
                data = self._udp.recv(65536)
            except TimeoutError:
                continue
            except OSError:
                return
            message = pr.parse_datagram(data)
            if message is not None and message["type"] == "pose":
                try:
                    self._pose = pr.pose_from_dict(message["pose"])
                except (KeyError, TypeError, ValueError):
                    pass


def _check(trajectory: Trajectory) -> None:
    """Refuse a trajectory the robot cannot run: same checks as run_pipeline's examples."""
    times, rows = trajectory.times, trajectory.positions
    if not times:
        raise ValueError("empty trajectory")
    if any(b <= a for a, b in zip(times, times[1:], strict=False)):
        raise ValueError("times must increase")
    if len(rows) != len(times) or any(len(r) != len(trajectory.joint_names) for r in rows):
        raise ValueError("one row of positions per time, one value per joint")
