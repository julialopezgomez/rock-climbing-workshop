# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Closed-loop check of an mjlab policy bundle in the organisers' simulator, before the real robot
[Ours].

The bundle comes from mjlab_microban (train_climbing.sh --deploy). Our move,
src/rcr/robot/climb_policy_move.py, is
loaded as Microban's program loads it (load_moves), with the bundle, and run with
run_move on the bundle's wall, the robot hanging on its start holds: the robot's servos (BAM, kp of
the move), its
sensors with their delays, the robot's observation (obs["q"], obs["gyro"], ...). Not the env the
policy was trained in:
it is the closest to the robot there is without the robot.

    uv sync --group sim --group rl   # the simulator, and onnxruntime (in the rl group)
    uv run --no-sync scripts/sim_check_climb_policy.py ../mjlab_microban/logs/deploy/<bundle>
    --seconds 30
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

MOVE = Path(__file__).resolve().parents[1] / "src" / "rcr" / "robot" / "climb_policy_move.py"


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("bundle", type=Path)
    p.add_argument(
        "--seconds",
        type=float,
        default=30.0,
        help="of climbing, after the ramp and the start delay",
    )
    p.add_argument("--view", action="store_true", help="MuJoCo's viewer, in real time (a display)")
    p.add_argument("--speed", type=float, default=1.0, help="with --view: x real time (default 1)")
    p.add_argument(
        "--no-adaptive-phase", action="store_true", help="the reference at its nominal speed"
    )
    p.add_argument(
        "--ideal-sensors", action="store_true", help="no sensor delays (to tell them apart)"
    )
    args = p.parse_args()

    from climbing_sim import ClimbingSim, Wall, load_moves, run_move

    bundle = args.bundle.resolve()
    if bundle.is_file():  # climb_bundle.npz (or policy.onnx) given: its folder
        bundle = bundle.parent
    if not (bundle / "climb_bundle.npz").is_file() or not (bundle / "policy.onnx").is_file():
        sys.exit(f"{bundle}: not a bundle folder (climb_bundle.npz + policy.onnx)")
    os.environ["MICROBAN_CLIMB_BUNDLE"] = str(bundle)  # the move looks there first
    moves = load_moves(str(MOVE))
    if not moves:
        sys.exit(f"{MOVE} gave no move (its bundle was not found in {bundle})")
    move = next(iter(moves.values()))
    if getattr(move, "bundle_dir", None) is None or Path(move.bundle_dir).resolve() != bundle:
        sys.exit(
            f"the move did not load the bundle {bundle} (found {getattr(move, 'bundle_dir', None)})"
        )
    move.options["adaptive_phase"] = not args.no_adaptive_phase
    move.options["save_log"] = False

    sim = ClimbingSim(
        Wall.from_json(bundle / "wall.json"), cameras=[], ideal_sensors=args.ideal_sensors
    )
    sim.reset_hanging()
    z0 = float(sim.data.qpos[2])
    lead = move.options["ramp_s"] + move.options["start_delay_s"]
    rec = {"t": [], "frame": [], "v_ref": [], "z": [], "hooked": [], "tilt": [], "stage": []}

    viewer = None
    if args.view:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(sim.model, sim.data)
    clock = {"wall": None, "sim": None}  # paces the run to real time in the viewer

    def on_step(s, obs) -> bool:
        c = move._ctrl
        climbing = move._stage == "climb"
        rec["t"].append(obs["time"])
        rec["stage"].append(move._stage)
        rec["frame"].append(c.t if climbing else 0)
        rec["v_ref"].append(c.v_ref if climbing else 1.0)
        rec["z"].append(float(s.data.qpos[2]) - z0)
        rec["hooked"].append(sum(h is not None for h in s.hooked().values()))
        g = obs["gravity"]
        rec["tilt"].append(
            math.degrees(math.acos(min(1.0, max(-1.0, -float(g[2])))))
            if np.isfinite(g).all()
            else 0
        )
        if viewer is not None:
            if not viewer.is_running():  # the window was closed
                return True
            viewer.sync()
            if clock["wall"] is None:
                clock["wall"], clock["sim"] = time.perf_counter(), obs["time"]
            elapsed = time.perf_counter() - clock["wall"]
            ahead = (obs["time"] - clock["sim"]) / args.speed - elapsed
            if ahead > 0:
                time.sleep(ahead)
        return move._stage == "aborted"

    run_move(move, sim, seconds=lead + args.seconds, on_step=on_step)
    r = {k: np.array(v) for k, v in rec.items()}
    climb = r["stage"] != "ramp"
    T = move._ctrl.T
    print(
        f"[SIM] wall {Path(bundle / 'wall.json').name}, {len(r['t'])} ticks, stage at the end: "
        f"{move._stage}"
        f"{' (' + move._reason + ')' if move._reason else ''}"
    )
    print(
        f"[SIM] reference frame reached {r['frame'].max()} / {T - 1} "
        f"(nominal at this time: {min(int(args.seconds / 0.02), T - 1)})"
    )
    print(
        f"[SIM] v_ref while climbing: mean {r['v_ref'][climb].mean():.3f}, min "
        f"{r['v_ref'][climb].min():.3f}, "
        f"time below 0.5: {(r['v_ref'][climb] < 0.5).mean() * 100:.0f} %"
    )
    print(
        f"[SIM] trunk height gained {r['z'][-1] * 100:.1f} cm (max {r['z'].max() * 100:.1f} cm); "
        "balls in sockets at "
        f"the end: {r['hooked'][-1]} / 4; max tilt {r['tilt'].max():.0f} deg"
    )
    np.savez(bundle / "sim_check.npz", **r)
    if viewer is not None:
        print("[SIM] done: close the viewer's window to exit")
        while viewer.is_running():
            time.sleep(0.05)


if __name__ == "__main__":
    main()
