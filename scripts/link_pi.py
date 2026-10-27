"""Talk to the Raspberry Pi over the link (rcr.link), from the laptop.

    uv run scripts/link_pi.py bundle                          # the Pi's files in dist/raspberry_pi
    uv run scripts/link_pi.py deploy pi@172.20.10.8         # copy our Pi code there (ssh)
    uv run scripts/link_pi.py deploy pi@172.20.10.8 --run   # ... and start the receiver there
    uv run scripts/link_pi.py check --pi 172.20.10.8          # evaluate the link
    uv run scripts/link_pi.py send --pi 172.20.10.8 [FILE]    # send a trajectory (or the example)
    uv run scripts/link_pi.py stream --pi 172.20.10.8 [FILE]  # send it, then each move added to it
    uv run scripts/link_pi.py replay --pi 172.20.10.8 [FILE]  # a saved one, in timed segments
    uv run scripts/link_pi.py start --pi 172.20.10.8
    uv run scripts/link_pi.py stop --pi 172.20.10.8

--pi defaults to $RCR_PI, else microban.local. On the Pi the receiver is
    cd ~/rcr && PYTHONPATH=src python3 -m rcr.link.raspberry_pi
(deploy --run does that for you, with Microban's Python if it is there).

check measures what the link README asks for: the round trip of a command, the time to send the
trajectory, and the rate and losses of the state coming back, during a dry run (no motors move).
stream sends the file and starts the robot, then follows the file: each move the optimiser saves
into it (tests/test_optimiser.py --hard --save) goes to the robot as the next segment, until
Ctrl-C. replay sends a saved file in timed segments, as if solved live, and reports how long the
robot held waiting for them.
bundle writes the Pi's files (what deploy copies), a run.sh and a README into one folder and a
.zip, to copy to any Pi by hand (scp, a USB stick) without this repository.
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

from rcr import io, link
from rcr.link import COMMAND_PORT, LaptopLink, LinkError
from rcr.types import Trajectory

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


RUN_SH = """#!/bin/sh
# Start the link receiver on this Pi (it drives no motors). Ctrl-C to stop. Options: --help.
cd "$(dirname "$0")" || exit 1
PYTHON=python3
[ -x "$HOME/microban/.venv/bin/python" ] && PYTHON="$HOME/microban/.venv/bin/python"
PYTHONPATH=src exec "$PYTHON" -m rcr.link.raspberry_pi "$@"
"""

BUNDLE_README = """Our code for the Raspberry Pi: the link's receiver (rcr.link.raspberry_pi).
Built by `uv run scripts/link_pi.py bundle` from commit {commit}, {date}.

1. Copy what is in this folder to ~/rcr on the Pi, from the laptop (asks the Pi's password):
       rsync -a dist/raspberry_pi/ pi@<pi-address>:rcr/
   (the user may be another one: `user` on Microban's image). The "/" after raspberry_pi
   copies the contents. Or put raspberry_pi.zip on the Pi and there: unzip it, then
   mv raspberry_pi ~/rcr (if ~/rcr is not there yet).
2. On the Pi:
       sh ~/rcr/run.sh
   It prints "listening on <address>:5005". Needs Python 3.9 or newer, nothing to install.
3. On the laptop, in our repository:
       uv run scripts/link_pi.py check --pi <pi-address>
       uv run scripts/link_pi.py send --pi <pi-address> [trajectory.json]

Trajectories it receives are saved to ~/rcr/trajectory.json (run.sh --save FILE for another).
src/rcr/link/README.md has the rest.
"""


def bundle(out: Path) -> int:
    """The Pi's files in `out` (replaced) and out.zip: copy either to any Pi by hand."""
    if out.exists():
        shutil.rmtree(out)
    for path in PI_FILES:
        source, target = ROOT / "src" / path, out / "src" / path
        if source.is_dir():
            shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    (out / "run.sh").write_text(RUN_SH)
    (out / "run.sh").chmod(0o755)
    commit = subprocess.run(
        ["git", "describe", "--always", "--dirty"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    (out / "README.txt").write_text(
        BUNDLE_README.format(commit=commit, date=time.strftime("%d %b %Y"))
    )
    archive = shutil.make_archive(str(out), "zip", out.parent, out.name)
    files = sum(1 for f in out.rglob("*") if f.is_file())
    print(f"{files} files in {out}, and {archive}")
    print(f"to a Pi: rsync -a {out}/ pi@<pi-address>:rcr/   then on the Pi: sh ~/rcr/run.sh")
    return 0


def replay(pi: str, port: int, file: Path, seconds: float, every: float) -> int:
    """Send a saved trajectory in segments of `seconds`, one every `every` s, while the robot
    plays it: as if each were solved just then (to see the robot hold when they come late)."""
    trajectory = io.load_trajectory(file)
    times, end = trajectory.times, trajectory.times[-1]
    cuts = [0]  # the index each segment starts at
    for i, t in enumerate(times):
        if t >= times[cuts[-1]] + seconds:
            cuts.append(i)
    cuts.append(len(times))
    count = len(cuts) - 1

    def part(k: int) -> Trajectory:  # segment k: what the robot executes, positions only
        lo, hi = cuts[k], cuts[k + 1]
        return Trajectory(trajectory.joint_names, times[lo:hi], trajectory.positions[lo:hi])

    print(
        f"{file.name}: {end:.1f} s in {count} segments of ~{seconds:.0f} s, one every {every:.1f} s"
    )
    try:
        with LaptopLink(pi, port) as robot:
            robot.stop()
            robot.send_trajectory(part(0), final=count == 1)
            robot.start()
            began = time.monotonic()
            print(f"  segment 1 of {count}: to {times[cuts[1] - 1]:.1f} s; started", flush=True)
            for k in range(1, count):
                time.sleep(every)
                robot.append_segment(part(k), final=k == count - 1)
                state = robot.robot_state()
                at = f"robot at {state.time:.1f} s" if state else "no state yet"
                print(
                    f"  segment {k + 1} of {count}: to {times[cuts[k + 1] - 1]:.1f} s ({at})",
                    flush=True,
                )
            deadline = time.monotonic() + end + 10
            while time.monotonic() < deadline:
                state = robot.robot_state()
                if state is not None and state.time >= end - 0.05:
                    break
                time.sleep(0.1)
            took = time.monotonic() - began
            robot.stop()
    except LinkError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    reached = state is not None and state.time >= end - 0.05
    held = max(0.0, took - end)
    print(
        f"robot {'reached' if reached else 'did NOT reach'} the end ({end:.1f} s) in {took:.1f} s: "
        f"it held {held:.1f} s in all, waiting for segments"
    )
    return 0 if reached else 1


def stream(pi: str, port: int, file: Path, poll: float = 0.5) -> int:
    """Send the trajectory file and start the robot, then follow the file: each time it has grown
    (the optimiser saves it after every move: tests/test_optimiser.py --hard --save), send what
    is new as the next segment. Until Ctrl-C, which tells the robot nothing more will come."""

    def read() -> tuple[float, Trajectory | str] | None:
        """(when it was saved, the trajectory or why it cannot be read), or None if not there."""
        try:
            saved = file.stat().st_mtime
        except OSError:
            return None
        try:
            return saved, io.load_trajectory(file)
        except (ValueError, KeyError, TypeError) as e:  # bad JSON, or not a trajectory
            return saved, f"{type(e).__name__}: {e}"

    def interrupt(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)  # a plain kill ends it as Ctrl-C does
    print(f"following {file} (Ctrl-C to end)", flush=True)
    while (found := read()) is None or isinstance(found[1], str):
        time.sleep(poll)
    seen, sent = found
    try:
        with LaptopLink(pi, port) as robot:
            robot.stop()
            robot.send_trajectory(sent, final=False)
            robot.start()
            print(f"  sent {_moves(sent)}, to {sent.times[-1]:.1f} s; started", flush=True)
            warned = None
            try:
                while True:
                    time.sleep(poll)
                    found = read()
                    if found is None or found[0] in (seen, warned):
                        continue
                    if isinstance(found[1], str):  # said once per save; read again if it changes
                        print(f"  {file.name} changed but cannot be read: {found[1]}", flush=True)
                        warned = found[0]
                        continue
                    seen, now = found
                    n = len(sent.times)
                    same = now.times[:n] == sent.times and now.positions[:n] == sent.positions
                    if same and len(now.times) > n:
                        try:
                            robot.append_segment(link.segment(now, n))
                        except LinkError as e:
                            print(f"  the new samples were not sent: {e}", flush=True)
                            continue
                        state = robot.robot_state()
                        at = f"robot at {state.time:.1f} s" if state else "no state yet"
                        print(
                            f"  sent {_moves(now, n)} more, to {now.times[-1]:.1f} s ({at})",
                            flush=True,
                        )
                        sent, warned = now, None
                    elif same:
                        print(
                            f"  {file.name} changed, but `times` and `positions` (what the robot "
                            f"runs) have no new samples ({n}): nothing to send",
                            flush=True,
                        )
                    else:
                        print(
                            f"  {file.name} changed but does not go on from what was sent "
                            f"({len(now.times)} samples now, {n} sent): waiting for it to grow "
                            "past that. A different plan? Then --stop, and stream again.",
                            flush=True,
                        )
            except KeyboardInterrupt:
                robot.finish()
                print(
                    f"\nended: the robot has all {len(sent.times)} samples "
                    f"({sent.times[-1]:.1f} s); it plays to the end and holds there. "
                    "To stop it: --stop"
                )
    except LinkError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


def _moves(trajectory: Trajectory, start: int = 0) -> str:
    """'N samples (~M moves)': a move is 24 samples with the optimiser's TransitionConfig."""
    count = len(trajectory.times) - start
    moves = f" (~{count / 24:.0f} moves)" if count >= 24 else ""
    return f"{count} sample{'s' if count != 1 else ''}{moves}"


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

    p = sub.add_parser("bundle", help="the Pi's files in one folder (and a .zip)")
    p.add_argument("--out", type=Path, default=ROOT / "dist" / "raspberry_pi")
    p = sub.add_parser("deploy", help="copy our Pi code to the Pi (rsync over ssh)")
    p.add_argument("host", help="user@address, e.g. pi@172.20.10.8")
    p.add_argument("--run", action="store_true", help="then start the receiver there")
    p = sub.add_parser("check", parents=[target], help="evaluate the link")
    p.add_argument("file", nargs="?", type=Path, default=EXAMPLE)
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--pings", type=int, default=20)
    p = sub.add_parser("send", parents=[target], help="send a trajectory")
    p.add_argument("file", nargs="?", type=Path, default=EXAMPLE)
    p = sub.add_parser("stream", parents=[target], help="send a file, then what is added to it")
    p.add_argument("file", nargs="?", type=Path, default=EXAMPLE)
    p = sub.add_parser("replay", parents=[target], help="send a saved file in timed segments")
    p.add_argument("file", nargs="?", type=Path, default=EXAMPLE)
    # 4 s: one move with the optimiser's TransitionConfig (1 + 2 s, then a 1 s settle)
    p.add_argument("--segment", type=float, default=4.0, help="[s] of motion per segment")
    p.add_argument("--every", type=float, default=3.0, help="[s] between segments (solve time)")
    sub.add_parser("start", parents=[target], help="start the climb")
    sub.add_parser("stop", parents=[target], help="stop the climb")
    args = ap.parse_args()

    if args.command == "bundle":
        return bundle(args.out)
    if args.command == "stream":
        return stream(args.pi, args.port, args.file)
    if args.command == "replay":
        return replay(args.pi, args.port, args.file, args.segment, args.every)
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
