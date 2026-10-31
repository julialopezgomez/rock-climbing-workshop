"""PC side of the robot's dashboard link (Rhoban/microban dev_v2,
src/humanoid_dashboard/link.py), UDP port 7350:
* PC -> robot, JSON: {"op": "hello"} once a second subscribes for 3 s (the robot answers with its
  schema); {"op": "cmd", "seq", "name", "args"} runs a command (move_start, move_stop, slider, ...).
* robot -> PC: the schema (JSON: field names and labels), acks and events (JSON), and one binary
  frame per control tick (50 Hz): b"HDT1", uint32 schema CRC, uint32 sequence, float64 time, then
  float32 values in the schema's order.
"""

import json
import math
import socket
import struct
import time
import zlib

import numpy as np

LINK_PORT = 7350
MAGIC = b"HDT1"
HEADER = struct.Struct("<4sIId")  # magic, schema CRC, sequence, robot time
HELLO_PERIOD_S = 1.0
MOVE_ACTIVE = 2  # the "moves" field's code of a running move (STOPPING: 3, INACTIVE: 0)


class RobotLink:
    """The PC side of the robot's dashboard link (microban's src/humanoid_dashboard/link.py)."""

    def __init__(self, host: str, port: int = LINK_PORT):
        self.addr = (socket.gethostbyname(host), port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", 0))
        self.sock.setblocking(False)
        self.schema = None
        self.schema_id = None
        self.unpack = None
        self.frame = {}  # field -> np.ndarray of its labels' values (NaN: not given)
        self.frame_time = None  # robot time of the last frame
        self.frames_received = 0
        self.seq = 0
        self.pending = {}  # cmd seq -> name, for the acks
        self._last_hello = -math.inf

    def send(self, msg: dict) -> None:
        try:
            self.sock.sendto(json.dumps(msg).encode(), self.addr)
        except OSError as e:
            print(f"send failed: {e}")

    def command(self, name: str, **args) -> None:
        self.seq += 1
        if name != "slider":  # sliders are not answered
            self.pending[self.seq] = name
        self.send({"op": "cmd", "seq": self.seq, "name": name, "args": args})

    def labels(self, field: str) -> list[str]:
        return dict(self.schema["fields"]).get(field, []) if self.schema else []

    def poll(self) -> None:
        """Say hello when due and read everything received (never blocks)."""
        now = time.monotonic()
        if now - self._last_hello >= HELLO_PERIOD_S:
            self.send({"op": "hello"})
            self._last_hello = now
        while True:
            try:
                data, _ = self.sock.recvfrom(65536)
            except (BlockingIOError, InterruptedError):
                return
            except OSError:  # e.g. ICMP port unreachable while the robot's program restarts
                continue
            if data[:4] == MAGIC:
                self._frame(data)
            else:
                self._message(data)

    def _frame(self, data: bytes) -> None:
        if self.unpack is None or len(data) != HEADER.size + self.unpack.size:
            return
        _, sid, _, t = HEADER.unpack_from(data)
        if sid != self.schema_id:  # the robot restarted with another schema: wait for it
            return
        flat = np.array(self.unpack.unpack_from(data, HEADER.size))
        k = 0
        for name, labels in self.schema["fields"]:
            self.frame[name] = flat[k : k + len(labels)]
            k += len(labels)
        self.frame_time = t
        self.frames_received += 1

    def _message(self, data: bytes) -> None:
        try:
            msg = json.loads(data)
        except ValueError:
            return
        op = msg.get("op")
        if op == "schema":
            # The frames carry the CRC of the schema bytes they follow.
            self.schema_id = zlib.crc32(data)
            if msg != self.schema:
                self.schema = msg
                self.unpack = struct.Struct(f"<{sum(len(labels) for _, labels in msg['fields'])}f")
                self.frame = {}
        elif op == "ack":
            name = self.pending.pop(msg.get("seq"), msg.get("name"))
            print(f"[robot] {name}: {'ok' if msg.get('ok') else 'REFUSED'} {msg.get('text', '')}")
        elif op == "event":
            print(f"[robot {msg.get('level', 'info')}] {msg.get('text', '')}")

    def close(self) -> None:
        self.sock.close()

    def move_state(self, move: str) -> int:
        """The move's state in the last frame: 0 off, 2 running (MOVE_ACTIVE), 3 stopping."""
        moves, names = self.frame.get("moves"), self.labels("moves")
        if moves is None or move not in names or math.isnan(moves[names.index(move)]):
            return 0
        return int(moves[names.index(move)])

    def check_move(self, move: str, deploy_hint: str) -> dict:
        """The move's entry of the robot's meta; exits with deploy_hint when the robot lacks it."""
        meta = self.schema.get("meta", {})
        print(
            f"robot {self.schema.get('robot')} ({meta.get('host', '?')}, "
            f"{meta.get('variant', 'classic')}), moves: {self.labels('moves')}"
        )
        entry = next((m for m in meta.get("moves", []) if m["name"] == move), None)
        if entry is None:
            raise SystemExit(f"The robot has no move {move!r}: {deploy_hint}")
        if not entry.get("available", True):
            raise SystemExit(
                f"The move {move!r} is not available on this robot version ({meta.get('variant')})."
            )
        return entry


def wait_for_schema(link: RobotLink, timeout: float) -> None:
    end = time.monotonic() + timeout
    while link.schema is None or link.frames_received == 0:
        if time.monotonic() > end:
            raise SystemExit(
                f"No answer from {link.addr[0]}:{link.addr[1]} in {timeout:.0f} s: is the robot's "
                "program running (make robot-service-start) and the robot on this network "
                "(make discover-robots)?"
            )
        link.poll()
        time.sleep(0.05)
