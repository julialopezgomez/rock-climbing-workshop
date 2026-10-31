"""Play an .npz trajectory (as sent to the robot) in the organisers' sim, with physics.

    uv run mjpython scripts/sim_npz.py build/trajectory_plan_N3.npz --wall l3_s41545
    uv run mjpython scripts/sim_npz.py build/trajectory_plan_N3.npz --wall l3_s41545 --key q_planned

Unlike the optimiser's playback (the planned poses shown exactly, no physics), here the servos
are sent the file's joint angles at its rate and the sim's servo model (BAM, at Microban's
firmware P gain) moves the robot under gravity, as on the real robot. The robot starts in the
file's first planned pose (q_planned, else q). After each move it prints which balls are hooked
and how far the trunk is from the plan. --wall: N1, N2, N3, or l<level>_s<seed> (a generated wall,
as scripts/plan_sim_wall.py names it). macOS needs mjpython for the window.
"""

import argparse
import re
import time
from pathlib import Path

import numpy as np
from climbing_sim import JOINTS, ClimbingSim, Wall
from climbing_sim.routes import generate_wall

from rcr.types import MICROBAN_JOINTS


def organisers_wall(name: str):
    """Their wall: a bundled one by name, or a generated one from l<level>_s<seed>."""
    seeded = re.fullmatch(r"l(\d)_s(\d+)", name)
    return generate_wall(int(seeded[1]), int(seeded[2])) if seeded else Wall.load(name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("npz", help="e.g. build/trajectory_plan_N3.npz")
    parser.add_argument("--wall", required=True, help="N1, N2, N3 or l<level>_s<seed>")
    parser.add_argument("--key", default="q", help="the array to send (default q, the robot's)")
    parser.add_argument("--headless", action="store_true", help="no window")
    args = parser.parse_args()

    if not Path(args.npz).exists():
        source = Path(args.npz).with_suffix(".json")
        hint = (
            f"uv run scripts/trajectory_to_npz.py {source}" if source.exists() else "solve it first"
        )
        raise SystemExit(f"no {args.npz}: {hint}")
    data = np.load(args.npz)
    sent = data[args.key]
    planned = data["q_planned"] if "q_planned" in data.files else sent
    rate = float(data["rate"]) if "rate" in data.files else 50.0
    order = [MICROBAN_JOINTS.index(name) for name in JOINTS]  # ours -> the sim's joint order

    sim = ClimbingSim(organisers_wall(args.wall), cameras=[], control_dt=1.0 / rate)
    start = planned[0]
    sim.reset(start[0:3], start[3:7], dict(zip(MICROBAN_JOINTS, start[7:], strict=True)))
    viewer = None if args.headless else sim.view()  # real time when shown
    hooked = sim.hooked()
    print(
        f"{args.npz} ({args.key}): {len(sent)} ticks at {rate:g} Hz on {args.wall}; start {hooked}"
    )
    for k in range(len(sent)):
        if viewer is not None and not viewer.is_running():
            break
        sim.step(sent[k, 7:][order])
        now = sim.hooked()
        if now != hooked:  # a ball left or reached a hold
            drop = (sim.data.qpos[2] - planned[k, 2]) * 1000
            print(f"  {k / rate:6.2f} s: hooked {now}; trunk {drop:+.0f} mm vs the plan in height")
            hooked = now
    print("done" + ("; close the window to quit" if viewer is not None else ""))
    while viewer is not None and viewer.is_running():
        time.sleep(0.1)


if __name__ == "__main__":
    main()
