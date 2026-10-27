"""Talk to the Raspberry Pi over the link (rcr.link), from the laptop.

    uv run scripts/link_pi.py deploy pi@172.20.10.8         # copy our Pi code there (ssh)
    uv run scripts/link_pi.py deploy pi@172.20.10.8 --run   # ... and start the receiver there
    uv run scripts/link_pi.py check --pi 172.20.10.8          # evaluate the link
    uv run scripts/link_pi.py send --pi 172.20.10.8 [FILE]    # send a trajectory (or the example)
    uv run scripts/link_pi.py start --pi 172.20.10.8
    uv run scripts/link_pi.py stop --pi 172.20.10.8

--pi defaults to $RCR_PI, else microban.local. On the Pi the receiver is
    cd ~/rcr && PYTHONPATH=src python3 -m rcr.link.raspberry_pi
(deploy --run does that for you, with Microban's Python if it is there).

check measures what the link README asks for: the round trip of a command, the time to send the
trajectory, and the rate and losses of the state coming back, during a dry run (no motors move).
"""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

from rcr import io
from rcr.link import COMMAND_PORT, LaptopLink, LinkError

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "data" / "examples" / "trajectory.json"
# What runs on the Pi [robot/README.md, "What goes to the Pi"]: Pi-safe, stdlib and numpy only
PI_FILES = ["rcr/__init__.py", "rcr/types.py", "rcr/io.py", "rcr/controller", "rcr/link"]
PI_DIR = "rcr"  # on the Pi, in the home folder: ~/rcr/src/rcr/...
# Microban's Python on the Pi if it is set up there [Microban Makefile], else the system one
PI_PYTHON = (
    "$( [ -x ~/microban/.venv/bin/python ] && echo ~/microban/.venv/bin/python || echo python3 )"
)


def deploy(host: str, run: bool) -> int:
    command = [
        "rsync",
        "-azR",
        "--delete",
        "--exclude",
        "__pycache__",
        f"--rsync-path=mkdir -p {PI_DIR}/src && rsync",  # the folder on a fresh Pi, same login
        *PI_FILES,  # -R, run from src/: each keeps its path from rcr/ on
        f"{host}:{PI_DIR}/src/",
    ]
    print(" ".join(command), flush=True)
    if subprocess.run(command, cwd=ROOT / "src").returncode:
        return 1
    print(f"copied to {host}:~/{PI_DIR}/src", flush=True)
    if not run:
        print(f"on the Pi: cd ~/{PI_DIR} && PYTHONPATH=src python3 -m rcr.link.raspberry_pi")
        return 0
    remote = f"cd ~/{PI_DIR} && PYTHONPATH=src {PI_PYTHON} -m rcr.link.raspberry_pi"
    return subprocess.run(["ssh", "-tt", host, remote]).returncode


def check(pi: str, port: int, file: Path, seconds: float, pings: int) -> int:
    trajectory = io.load_trajectory(file)
    results = []

    def result(ok: bool, what: str) -> None:
        results.append(ok)
        print(f"  {'ok  ' if ok else 'FAIL'} {what}", flush=True)

    print(f"link to {pi}:{port}")
    with LaptopLink(pi, port) as link:
        try:
            first = link.ping()
        except LinkError as e:
            result(False, str(e))
            return 1
        result(True, f"connected (first command {first * 1e3:.0f} ms, with connecting)")

        times = [link.ping() for _ in range(pings)]
        median = statistics.median(times)
        result(
            median < 0.1,
            f"round trip over {pings} commands: median {median * 1e3:.1f} ms, "
            f"min {min(times) * 1e3:.1f}, max {max(times) * 1e3:.1f}",
        )

        size = len(file.read_bytes())
        start = time.perf_counter()
        try:
            link.stop()  # in case a previous check left it running
            link.send_trajectory(trajectory)
            took = time.perf_counter() - start
            result(
                True,
                f"trajectory sent and checked by the robot: {len(trajectory.times)} samples, "
                f"{size / 1e6:.2f} MB in {took:.2f} s ({size / 1e6 / took:.1f} MB/s)",
            )
        except LinkError as e:
            result(False, f"trajectory: {e}")
            return 1

        link.start()
        received, lost = link.states_received, link.states_lost
        time.sleep(seconds)
        link.stop()
        got, missing = link.states_received - received, link.states_lost - lost
        state = link.robot_state()
        rate = got / seconds
        loss = missing / max(1, got + missing)
        result(
            got > 0 and loss < 0.05,
            f"state back during a {seconds:.0f} s dry run: {rate:.0f} per second, "
            f"{loss:.1%} lost ({missing} of {got + missing})",
        )
        if state is not None:
            result(
                set(state.joint_positions) == set(trajectory.joint_names),
                f"state has the trajectory's {len(trajectory.joint_names)} joints "
                f"(robot time {state.time:.2f} s)",
            )
        else:
            result(False, "no state came back: is a firewall on the laptop blocking UDP?")

    passed = all(results)
    print("link check passed" if passed else "link check FAILED")
    return 0 if passed else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    target = argparse.ArgumentParser(add_help=False)
    target.add_argument("--pi", default=os.environ.get("RCR_PI", "microban.local"))
    target.add_argument("--port", type=int, default=COMMAND_PORT)

    p = sub.add_parser("deploy", help="copy our Pi code to the Pi (rsync over ssh)")
    p.add_argument("host", help="user@address, e.g. pi@172.20.10.8")
    p.add_argument("--run", action="store_true", help="then start the receiver there")
    p = sub.add_parser("check", parents=[target], help="evaluate the link")
    p.add_argument("file", nargs="?", type=Path, default=EXAMPLE)
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--pings", type=int, default=20)
    p = sub.add_parser("send", parents=[target], help="send a trajectory")
    p.add_argument("file", nargs="?", type=Path, default=EXAMPLE)
    sub.add_parser("start", parents=[target], help="start the climb")
    sub.add_parser("stop", parents=[target], help="stop the climb")
    args = ap.parse_args()

    if args.command == "deploy":
        return deploy(args.host, args.run)
    if args.command == "check":
        return check(args.pi, args.port, args.file, args.seconds, args.pings)
    try:
        with LaptopLink(args.pi, args.port) as link:
            if args.command == "send":
                trajectory = io.load_trajectory(args.file)
                link.send_trajectory(trajectory)
                print(f"sent {args.file} ({len(trajectory.times)} samples) to {args.pi}")
            elif args.command == "start":
                link.start()
                print("started")
            else:
                link.stop()
                print("stopped")
    except LinkError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
