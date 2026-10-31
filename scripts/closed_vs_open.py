"""The whole route from the reference's first pose, no resets, in the organisers' sim: playback
and a baked .npz open loop, against the policy closed loop with 0..3 extra ticks of observation
delay (a Wi-Fi round trip). Reports the moves completed before a held limb leaves
its hold (policy_to_npz.open_loop's rule), over a few sim seeds.

    uv run --no-sync python scripts/closed_vs_open.py --data build/rl/l2 \
        --policy build/rl/l2/sweep6/res_a0.1_kp3000_ls-1.5_s0_500k.onnx \
        --npz data/executable-files/policy_l2_kp3000_500k.npz
"""

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import onnxruntime

sys.path.insert(0, "scripts")
import train_controller as tc  # noqa: E402
from policy_to_npz import GRACE_S  # noqa: E402  (scripts/)

from rcr.controller import tracking  # noqa: E402
from rcr.types import LIMBS  # noqa: E402


def climb(env, times, targets_at, seed: int) -> int:
    """targets_at(i, obs) -> 19 targets (JOINTS order) at frame i; the moves done in a row."""
    sim, traj, moves = env.sim, env.trajectory, env.moves
    sim.rng = np.random.default_rng(seed)
    pose = traj.trunk_poses[0]
    start = dict(zip(traj.joint_names, traj.planned_positions[0], strict=True))
    sim.reset(pose[:3], pose[3:], start)
    first = tracking.at(env.ref, times[0])[0]
    for _ in range(round(tc.SETTLE_S / env.dt)):
        obs = sim.step(first[tc.TO_SIM])
    ends = [m["t1"] for m in moves]
    for i in range(1, len(times)):
        obs = sim.step(targets_at(i, obs)[tc.TO_SIM])
        k = int(np.searchsorted(ends, times[i], side="right"))
        if k >= len(moves):
            break
        move, hooked = moves[k], sim.hooked()
        settling = moves[k - 1]["limb"] if k > 0 and times[i] - move["t0"] < GRACE_S else None
        if any(hooked[limb] != move["before"][limb] for limb in move["fixed"] if limb != settling):
            return k
    hooked = sim.hooked()
    done = all(hooked[limb] == moves[-1]["after"][limb] for limb in LIMBS)
    return len(moves) if done else len(moves) - 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--npz", type=Path)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    meta = json.loads(Path(f"{args.policy.with_suffix('')}.json").read_text())
    session = onnxruntime.InferenceSession(str(args.policy))
    env = tc.TrackingEnv(args.data, meta["alpha"], meta["kp_fw"])
    times = np.asarray(env.trajectory.times)
    ff = np.array([tracking.at(env.ref, t)[0] for t in times])
    planned = np.array([tracking.at(env.ref, t)[1] for t in times])
    ahead = np.array([tracking.at(env.ref, t + tracking.LOOKAHEAD_S)[1] for t in times])

    def closed(delay: int):
        past = deque(maxlen=delay + 1)

        def targets_at(i, obs):
            past.append(obs)
            o = past[0]  # the observation `delay` ticks old
            q, dq = o["q"][tc.FROM_SIM], o["dq"][tc.FROM_SIM]
            x = tracking.observation(planned[i], ahead[i], q, dq, o["quat"], o["gyro"])
            a = session.run(None, {"obs": x[None]})[0][0]
            return tracking.targets(ff[i], a, env.alpha, env.low, env.high)

        return targets_at

    runs = {"playback (Julia's trajectory), open loop": lambda: lambda i, obs: ff[i]}
    if args.npz:
        baked = np.load(args.npz)["q"][:, 7:]
        runs[f"{args.npz.name}, open loop"] = lambda: lambda i, obs: baked[i]
    for d in range(4):
        runs[f"policy closed loop, +{d} tick delay ({20 * d} ms)"] = lambda d=d: closed(d)
    kp = meta["kp_fw"]
    print(f"{args.data}: {len(env.moves)} moves; kp_fw {kp:g} P only; {args.policy.name}")
    for name, make in runs.items():
        done = [climb(env, times, make(), seed) for seed in range(args.seeds)]
        print(f"{name:<55} moves in a row: {done}", flush=True)


if __name__ == "__main__":
    main()
