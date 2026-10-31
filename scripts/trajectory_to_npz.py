"""Resample a saved trajectory at a fixed rate, as an .npz like iktrajv0.npz.

    uv run scripts/trajectory_to_npz.py build/trajectory_plan_N3.json  # writes the .npz beside it
    uv run scripts/trajectory_to_npz.py build/trajectory_plan_N3.json --rate 50 --out build/x.npz

The input is a trajectory the optimiser saved (tests/test_optimiser.py --save, or
src/rcr/optimiser/run.sh), with its planned motion (trunk_poses and planned_positions). The output:

    q          T x 26  FOR THE ROBOT (Microban's move reads it), as the IK trajectories: trunk
                       position [m], trunk quaternion (w, x, y, z), then the 19 joint targets [rad]
                       in rcr.types.MICROBAN_JOINTS order (the trajectory's `positions`)
    q_planned  T x 26  the planned configuration (the same as q now that the optimiser sends the
                       planned angles as they are; kept for the sim and older files)
    times      T       [s] 0, 1/rate, 2/rate, ... up to the trajectory's end
    rate       the rate [Hz]

Between the saved samples: linear for the trunk position and the joints (as the controller and the
sim's playback do), spherical (slerp) for the quaternion.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from rcr import io
from rcr.types import MICROBAN_JOINTS

RATE = 50.0  # [Hz] the robot's control loop (CLAUDE.md) [Ours]


def slerp(a: np.ndarray, b: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Unit quaternions between a and b (each n x 4) at fractions s (n): the shorter way round."""
    dot = np.sum(a * b, axis=1)
    b = np.where(dot[:, None] < 0, -b, b)
    dot = np.abs(dot)
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    near = angle < 1e-6  # nearly the same: linear, then normalised
    sin = np.where(near, 1.0, np.sin(angle))
    wa = np.where(near, 1 - s, np.sin((1 - s) * angle) / sin)
    wb = np.where(near, s, np.sin(s * angle) / sin)
    q = wa[:, None] * a + wb[:, None] * b
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def resample(path: Path, rate: float) -> dict[str, np.ndarray]:
    if not path.exists():
        sys.exit(f"no trajectory at {path}: solve and save one first (src/rcr/optimiser/run.sh)")
    traj = io.load_trajectory(path)
    if traj.trunk_poses is None or traj.planned_positions is None:
        sys.exit(
            f"{path} has no planned motion (trunk_poses, planned_positions): save it with --save"
        )
    order = [traj.joint_names.index(name) for name in MICROBAN_JOINTS]
    times = np.asarray(traj.times)
    trunk = np.asarray(traj.trunk_poses)
    planned = np.asarray(traj.planned_positions)[:, order]

    ticks = np.arange(0.0, times[-1] + 1e-9, 1.0 / rate)
    k = np.clip(np.searchsorted(times, ticks, side="right") - 1, 0, len(times) - 2)
    s = np.clip((ticks - times[k]) / (times[k + 1] - times[k]), 0.0, 1.0)

    def linear(rows):
        return (1 - s)[:, None] * rows[k] + s[:, None] * rows[k + 1]

    pose = np.hstack([linear(trunk[:, 0:3]), slerp(trunk[k, 3:7], trunk[k + 1, 3:7], s)])
    return {
        "q": np.hstack([pose, linear(planned)]),  # what the robot is sent: the planned angles
        "q_planned": np.hstack([pose, linear(planned)]),
        "times": ticks,
        "rate": np.array(rate),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "trajectory", type=Path, help="a saved trajectory, e.g. build/trajectory_plan_N3.json"
    )
    parser.add_argument("--rate", type=float, default=RATE, help=f"[Hz] default {RATE:g}")
    parser.add_argument("--out", type=Path, help="default: build/<the trajectory's name>.npz")
    args = parser.parse_args()
    out = args.out or Path("build") / f"{args.trajectory.stem}.npz"
    data = resample(args.trajectory, args.rate)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **data)
    ticks, seconds = len(data["q"]), data["times"][-1]
    print(f"{args.trajectory}: {ticks} ticks at {args.rate:g} Hz ({seconds:.2f} s) -> {out}")


if __name__ == "__main__":
    main()
