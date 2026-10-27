"""The receiver, on the Pi: waits for the laptop, saves each trajectory, reports start and stop.

    PYTHONPATH=src python3 -m rcr.link.raspberry_pi [--save FILE] [--rate HZ] [--port N]

It drives no motors. It stands in for our Move until that exists (it will hold a RobotLink the same
way): to check the link from the laptop (scripts/link_pi.py check, stream) and to put
trajectories on the Pi. After start it plays the trajectory on its own clock (a dry run) and sends
those positions back as the state, so the laptop sees them. Sent move by move, the trajectory
grows while it plays; at the end of what it has, the clock holds there until the next segment
comes, as a controller must [Ours]. Stop it with Ctrl-C.
"""

from __future__ import annotations

import argparse
import socket
import time
from pathlib import Path

from rcr import io
from rcr.link import protocol as pr
from rcr.link.raspberry_pi.robot_link import RobotLink
from rcr.types import RobotState, Trajectory


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--save", default="~/rcr/trajectory.json", help="where to save what arrives")
    ap.add_argument("--rate", type=float, default=50.0, help="[Hz] state sent back")
    ap.add_argument("--port", type=int, default=pr.COMMAND_PORT)
    ap.add_argument("--pose-port", type=int, default=pr.POSE_PORT)
    args = ap.parse_args()
    save = Path(args.save).expanduser()

    try:
        robot_link = RobotLink(args.port, args.pose_port)
    except OSError as e:
        raise SystemExit(
            f"cannot listen on port {args.port} or {args.pose_port}: {e}. Is the receiver already "
            "running? Stop it with: pkill -f rcr.link.raspberry_pi"
        ) from None
    with robot_link as link:
        print(f"listening on {_address()}:{link.port} (TCP), poses on {link.pose_port} (UDP)")
        print("drives no motors. Ctrl-C to stop.", flush=True)
        seen, running, laptop, clock, waiting, ended = 0, False, None, 0.0, False, False
        tick, next_tick = 1.0 / args.rate, time.monotonic()
        try:
            while True:
                if link.laptop() != laptop:
                    laptop = link.laptop()
                    print(f"laptop subscribed: state to {laptop[0]}:{laptop[1]} (UDP)", flush=True)
                if link.trajectories_received != seen:
                    seen = link.trajectories_received
                    trajectory = link.trajectory()
                    save.parent.mkdir(parents=True, exist_ok=True)
                    io.save_trajectory(trajectory, save)
                    more = "" if link.trajectory_complete() else ", more to come"
                    print(
                        f"trajectory {seen}: {len(trajectory.times)} samples, "
                        f"{trajectory.times[-1]:.1f} s, {len(trajectory.joint_names)} joints"
                        f"{more}; saved to {save}",
                        flush=True,
                    )
                if link.run_requested() != running:
                    running = link.run_requested()
                    if running:
                        clock, waiting, ended = 0.0, False, False
                    print("start" if running else f"stop at {clock:.2f} s", flush=True)
                trajectory = link.trajectory()
                if running and trajectory is not None:
                    end = trajectory.times[-1]
                    if clock + tick <= end:
                        clock += tick
                        if waiting:
                            print(f"next segment here: going on from {clock:.2f} s", flush=True)
                            waiting = False
                    elif not link.trajectory_complete() and not waiting:
                        print(f"waiting at {end:.2f} s for the next segment (holding)", flush=True)
                        waiting = True
                    elif link.trajectory_complete() and not ended:
                        print(f"end of the trajectory at {end:.2f} s (holding)", flush=True)
                        ended = True
                link.send_state(_state(trajectory, clock))
                next_tick += tick  # on schedule, whatever the work took
                time.sleep(max(0.0, next_tick - time.monotonic()))
        except KeyboardInterrupt:
            print()


def _state(trajectory: Trajectory | None, elapsed: float) -> RobotState:
    """The trajectory's positions at `elapsed` (the last sample before it), as a robot state."""
    positions = {}
    if trajectory is not None:
        i = 0
        while i + 1 < len(trajectory.times) and trajectory.times[i + 1] <= elapsed:
            i += 1
        row = trajectory.positions[i]
        positions = {name: row[j] for j, name in enumerate(trajectory.joint_names)}
    return RobotState(
        time=elapsed,
        joint_positions=positions,
        joint_velocities=dict.fromkeys(positions, 0.0),
        imu_quat=(1.0, 0.0, 0.0, 0.0),
        gyro=(0.0, 0.0, 0.0),
    )


def _address() -> str:
    """This machine's address on the network (what the laptop connects to)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("10.255.255.255", 1))  # sends nothing: only picks the outgoing interface
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


if __name__ == "__main__":
    main()
