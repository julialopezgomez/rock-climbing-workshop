"""What crosses the link, byte by byte: shared by the laptop and the Pi [Ours, D5]. Pi-safe: stdlib.

TCP, for what must arrive whole and in order: the laptop connects to the robot's COMMAND_PORT and
sends messages; the robot answers each one with an ack. A message is a 4-byte big-endian length,
then that many bytes of UTF-8 JSON, an object with a "type":

    laptop -> robot                          robot's ack (always "type": "ack", "ok", "error")
    hello       {"state_port": int}          {"pose_port": int}
    trajectory  {"trajectory": rcr.io JSON}  {"samples": int, "duration": float [s]}
    start, stop, ping                        {}

UDP, for what streams and where only the newest counts (a lost one is replaced by the next): one
JSON object per datagram, {"type", "seq", "session", and the payload}. "seq" counts up from 1 in
each "session" (one per start of the sender), so gaps show losses.

    state  robot -> laptop, to the laptop's address and the state_port it sent in hello
    pose   laptop -> robot, to the robot's address and the pose_port it sent back
"""

from __future__ import annotations

import json
import socket
import struct

from rcr.types import Pose, RobotState

COMMAND_PORT = 5005  # [Ours] TCP, on the robot
POSE_PORT = 5006  # [Ours] UDP, on the robot (the laptop picks any free port for the state)
MAX_MESSAGE = 64 * 2**20  # [bytes] a trajectory is ~1 MB; anything this big is a broken stream
_LENGTH = struct.Struct(">I")


class LinkError(RuntimeError):
    """The robot cannot be reached, or it refused a command (the message says why)."""


# --- TCP messages -------------------------------------------------------------------------------


def send_message(sock: socket.socket, message: dict) -> None:
    data = json.dumps(message, separators=(",", ":")).encode()
    sock.sendall(_LENGTH.pack(len(data)) + data)


def recv_message(sock: socket.socket) -> dict | None:
    """The next message, or None if the other side closed the connection."""
    header = _recv_exact(sock, _LENGTH.size)
    if header is None:
        return None
    (size,) = _LENGTH.unpack(header)
    if size > MAX_MESSAGE:
        raise LinkError(f"message of {size} bytes: not our protocol")
    data = _recv_exact(sock, size)
    if data is None:
        return None
    return json.loads(data)


def _recv_exact(sock: socket.socket, size: int) -> bytes | None:
    parts, left = [], size
    while left:
        part = sock.recv(min(left, 2**20))
        if not part:
            return None
        parts.append(part)
        left -= len(part)
    return b"".join(parts)


# --- UDP datagrams ------------------------------------------------------------------------------


def datagram(kind: str, seq: int, payload: dict, session: int = 0) -> bytes:
    message = {"type": kind, "seq": seq, "session": session, kind: payload}
    return json.dumps(message, separators=(",", ":")).encode()


def parse_datagram(data: bytes) -> dict | None:
    """The datagram's object, or None if it is not one of ours."""
    try:
        message = json.loads(data)
    except ValueError:
        return None
    return message if isinstance(message, dict) and "type" in message else None


# --- Payloads -----------------------------------------------------------------------------------


def pose_to_dict(pose: Pose) -> dict:
    return {"position": list(pose.position), "quat": list(pose.quat)}


def pose_from_dict(d: dict) -> Pose:
    return Pose(position=tuple(d["position"]), quat=tuple(d["quat"]))


def state_to_dict(state: RobotState) -> dict:
    return {
        "time": state.time,
        "joint_positions": state.joint_positions,
        "joint_velocities": state.joint_velocities,
        "imu_quat": list(state.imu_quat),
        "gyro": list(state.gyro),
        "base_pose": None if state.base_pose is None else pose_to_dict(state.base_pose),
    }


def state_from_dict(d: dict) -> RobotState:
    return RobotState(
        time=d["time"],
        joint_positions=dict(d["joint_positions"]),
        joint_velocities=dict(d["joint_velocities"]),
        imu_quat=tuple(d["imu_quat"]),
        gyro=tuple(d["gyro"]),
        base_pose=None if d.get("base_pose") is None else pose_from_dict(d["base_pose"]),
    )
