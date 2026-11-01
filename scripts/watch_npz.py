"""Play a replay .npz (q: trunk 7 + 19 joints, 0.02 s a frame) open loop in the organisers' sim,
in MuJoCo's viewer, as the robot replays it: the robot placed at the first frame, then one frame
a tick.

    uv run --no-sync mjpython scripts/watch_npz.py \
        data/wall1_best/ROBOT_wall1_first4moves_x13_to_top.npz
    uv run --no-sync mjpython scripts/watch_npz.py FILE.npz --wall build/rl/v3/n2ik/sim_wall.json

macOS needs mjpython for the viewer (Linux: python). --headless runs it without a window and prints
which hold each ball is in every --report seconds.
"""

import argparse
from pathlib import Path

import numpy as np
from climbing_sim import JOINTS as SIM_JOINTS
from climbing_sim import ClimbingSim, Wall

from rcr.types import MICROBAN_JOINTS

TO_SIM = [list(MICROBAN_JOINTS).index(n) for n in SIM_JOINTS]  # the file's order -> the sim's
SETTLE_S = 0.5  # [s] holding the first frame before the replay [Ours]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("npz", type=Path)
    parser.add_argument("--wall", default="N1", help="N1, N2, N3 or a wall .json (sim_wall.json)")
    parser.add_argument("--kp", type=float, default=3000.0, help="servo P gain (kp_fw)")
    parser.add_argument("--speed", type=float, default=1.0, help="1.25: 25 %% faster, as the robot")
    parser.add_argument("--headless", action="store_true", help="no window")
    parser.add_argument("--report", type=float, default=10.0, help="[s] between hold reports")
    args = parser.parse_args()

    q = np.load(args.npz)["q"]
    if args.speed != 1.0:  # the frames played faster, as robot_replay's speed does
        t = np.arange(len(q)) * 0.02
        tn = np.arange(0, t[-1] / args.speed + 1e-9, 0.02)
        q = np.array([np.interp(tn * args.speed, t, q[:, j]) for j in range(q.shape[1])]).T
    sim = ClimbingSim(Wall.load(str(args.wall)), cameras=[], kp_fw=args.kp)
    joints = dict(zip(MICROBAN_JOINTS, q[0, 7:], strict=True))
    sim.reset(q[0, :3], q[0, 3:7], joints)
    if not args.headless:
        viewer = sim.view()  # real time
    for _ in range(round(SETTLE_S / sim.control_dt)):
        sim.step(q[0, 7:][TO_SIM])
    print(f"{args.npz.name}: {len(q)} frames ({(len(q) - 1) * 0.02:.1f} s) at kp_fw {args.kp:g}, "
          f"speed {args.speed:g}; start: {sim.hooked()}", flush=True)  # fmt: skip
    every = round(args.report / 0.02)
    for i in range(1, len(q)):
        if not args.headless and not viewer.is_running():
            break
        sim.step(q[i, 7:][TO_SIM])
        if i % every == 0 or i == len(q) - 1:
            print(f"t {i * 0.02:6.1f} s  trunk z {sim.trunk_pose()[0][2]:.3f} m  {sim.hooked()}",
                  flush=True)  # fmt: skip
    if not args.headless:
        while viewer.is_running():  # keeps the last pose on screen until the window is closed
            viewer.sync()


if __name__ == "__main__":
    main()
