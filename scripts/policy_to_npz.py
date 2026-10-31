"""Bake a residual policy into an open-loop trajectory for the robot's replay (robot_replay.py on
feature/control): q (frames, 26) = trunk pose (x y z, w x y z; not driven) + 19 joint targets in
the robot's joint order, one frame every 0.02 s.

Each move is rolled out in the organisers' sim from its reference start with the policy; where it
slots in (and the held limbs stay), the frames of that move are the policy's targets, else the
reference's (playback). The result is then checked open loop in the sim, from the start, against
playback: how many moves in a row it gets through.

    uv run --no-sync python scripts/policy_to_npz.py --data build/rl/l2 \
        --policy build/rl/l2/sweep6/res_a0.1_kp3000_ls-1.5_s0.onnx --out build/rl/l2/policy_l2.npz

Replay it P only at the policy's kp_fw (saved in the npz as kp_fw): --kp <kp_fw> --ki 0 --kd 0.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime

sys.path.insert(0, "scripts")
import train_controller as tc  # noqa: E402

from rcr.controller import tracking  # noqa: E402
from rcr.types import LIMBS  # noqa: E402

GRACE_S = 0.5  # [s] for a moved ball to settle into its cup after the planned end [Ours]


def bake(env, act, times) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Targets (frames, 19, JOINTS order), which frames are the policy's, each move's outcome."""
    targets = np.array([tracking.at(env.ref, t)[0] for t in times])  # playback
    used = np.zeros(len(times), dtype=bool)
    outcomes = []
    for k, move in enumerate(env.moves):
        env.only, env.chain = [k], False
        try:
            obs = env.reset()[0]
        except RuntimeError:
            outcomes.append("start not held: playback")
            continue
        i0, i1 = int(np.searchsorted(times, move["t0"])), int(np.searchsorted(times, move["t1"]))
        rec, failed = [], False
        for i in range(i0 + 1, i1 + 1):
            env.t = times[i] - env.dt  # step() adds dt: stay on the reference's own frames
            a = act(obs)
            ff = tracking.at(env.ref, times[i])[0]
            rec.append(tracking.targets(ff, a, env.alpha, env.low, env.high))
            obs, _, _, _, info = env.step(a)
            if info["failed"]:
                failed = True
                break
        ok = False
        if not failed:  # the ball may settle into its cup just after t1: hold the last target
            for _ in range(round(GRACE_S / env.dt) + 1):
                hooked = env.sim.hooked()
                if any(hooked[limb] != move["before"][limb] for limb in move["fixed"]):
                    break
                if hooked[move["limb"]] == move["after"][move["limb"]]:
                    ok = True
                    break
                env.sim.step(rec[-1][tc.TO_SIM] if rec else None)
        if ok:
            targets[i0 + 1 : i1 + 1] = rec
            used[i0 + 1 : i1 + 1] = True
        outcomes.append("policy" if ok else "policy failed: playback")
    return targets, used, outcomes


def open_loop(env, trajectory, targets, times) -> int:
    """Play the targets from the reference's first pose, no resets; the moves completed before a
    held limb leaves its hold (the limb that just moved gets GRACE_S to settle in)."""
    sim = env.sim
    pose = trajectory.trunk_poses[0]
    start = dict(zip(trajectory.joint_names, trajectory.planned_positions[0], strict=True))
    sim.reset(pose[:3], pose[3:], start)
    for _ in range(round(tc.SETTLE_S / env.dt)):
        sim.step(targets[0][tc.TO_SIM])
    moves = env.moves
    for i in range(1, len(times)):
        sim.step(targets[i][tc.TO_SIM])
        k = int(np.searchsorted([m["t1"] for m in moves], times[i], side="right"))
        if k >= len(moves):
            break
        move, hooked = moves[k], sim.hooked()
        settling = moves[k - 1]["limb"] if k > 0 and times[i] - move["t0"] < GRACE_S else None
        if any(hooked[limb] != move["before"][limb] for limb in move["fixed"] if limb != settling):
            return k
    hooked = sim.hooked()  # the last move: its ball in its cup at the end
    done = all(hooked[limb] == moves[-1]["after"][limb] for limb in LIMBS)
    return len(moves) if done else len(moves) - 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    meta = json.loads(Path(f"{args.policy.with_suffix('')}.json").read_text())
    assert meta.get("action") == tracking.ACTION, "not a residual policy"
    session = onnxruntime.InferenceSession(str(args.policy))
    act = lambda obs: session.run(None, {"obs": obs[None]})[0][0]  # noqa: E731
    env = tc.TrackingEnv(args.data, meta["alpha"], meta["kp_fw"])
    times = np.asarray(env.trajectory.times)
    targets, used, outcomes = bake(env, act, times)
    for k, o in enumerate(outcomes):
        print(f"move {k} ({env.moves[k]['limb']}): {o}")

    playback = np.array([tracking.at(env.ref, t)[0] for t in times])
    n_policy = open_loop(env, env.trajectory, targets, times)
    n_play = open_loop(env, env.trajectory, playback, times)
    q = np.hstack([np.asarray(env.trajectory.trunk_poses), targets])
    np.savez(args.out, q=q, dt=0.02, kp_fw=meta["kp_fw"], ki=0, kd=0, policy=str(args.policy),
             policy_frames=used, alpha=meta["alpha"])  # fmt: skip
    print(
        f"moves from the policy: {outcomes.count('policy')}/{len(outcomes)}; frames from the "
        f"policy: {used.mean():.0%}"
    )
    print(
        f"open loop in the sim from the start, moves in a row: baked {n_policy}, playback "
        f"{n_play} (of {len(env.moves)}), both at kp_fw {meta['kp_fw']:g} P only"
    )
    print(f"saved {args.out}: q {q.shape}; replay with --kp {meta['kp_fw']:g} --ki 0 --kd 0")


if __name__ == "__main__":
    main()
