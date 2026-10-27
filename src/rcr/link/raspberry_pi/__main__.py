"""The receiver, on the Pi: waits for the laptop, saves each trajectory, reports start and stop.

    PYTHONPATH=src python3 -m rcr.link.raspberry_pi [--save FILE] [--rate HZ] [--port N]

It drives no motors. It stands in for our Move until that exists (it will hold a RobotLink the same
way): to check the link from the laptop (scripts/link_pi.py check) and to put trajectories on the
Pi. As state it sends the trajectory's own positions at the time since start (a dry run), so the
laptop sees them come back. Stop it with Ctrl-C.
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

    with RobotLink(args.port, args.pose_port) as link:
        print(f"listening on {_address()}:{link.port} (TCP), poses on {link.pose_port} (UDP)")
        print("drives no motors. Ctrl-C to stop.", flush=True)
        seen, running, laptop, started = 0, False, None, 0.0
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
                    print(
                        f"trajectory {seen}: {len(trajectory.times)} samples, "
                        f"{trajectory.times[-1]:.1f} s, {len(trajectory.joint_names)} joints; "
                        f"saved to {save}",
                        flush=True,
                    )
                if link.run_requested() != running:
                    running = link.run_requested()
                    started = time.monotonic()
                    print("start" if running else "stop", flush=True)
                elapsed = time.monotonic() - started if running else 0.0
                link.send_state(_state(link.trajectory(), elapsed))
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
