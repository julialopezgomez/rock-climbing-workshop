"""Record a chained climb in the organisers' sim (headless, EGL) to an MP4: the whole wall on the
left, a camera following the trunk on the right, the move and its outcome written on top.

    MUJOCO_GL=egl uv run --no-sync python scripts/render_climb.py \
        --policy build/rl/n1/sweep3/res_a0.2_kp1000_ls-1_s0.onnx --out climb_policy.mp4
    MUJOCO_GL=egl uv run --no-sync python scripts/render_climb.py --kp 1000 --out climb_playback.mp4

Moves run one after another as on the wall: after a slot-in the next move starts where this one
ended. After a fall the robot is put back at the next move's reference start (marked RESET), so the
whole route is shown either way.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, "scripts")
import train_controller as tc  # noqa: E402

from rcr.controller import tracking  # noqa: E402

W, H = 640, 540  # each half of the frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, help="ONNX policy (none: playback)")
    parser.add_argument("--kp", type=float, default=1000.0, help="playback only: kp_fw")
    parser.add_argument("--data", type=Path, default=Path("build/rl/n1"))
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--last", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=25, help="video fps (sim runs at 50 Hz)")
    args = parser.parse_args()

    act, alpha, kp, name = None, 0.0, args.kp, f"playback, kp_fw {args.kp:g}"
    if args.policy:
        import onnxruntime

        session = onnxruntime.InferenceSession(str(args.policy))
        act = lambda obs: session.run(None, {"obs": obs[None]})[0][0]  # noqa: E731
        meta = json.loads(Path(f"{args.policy.with_suffix('')}.json").read_text())
        alpha, kp = meta["alpha"], meta["kp_fw"]
        name = f"RL {args.policy.stem}, kp_fw {kp:g}"
    env = tc.TrackingEnv(args.data, alpha, kp)
    last = len(env.moves) - 1 if args.last is None else args.last
    every = round(50 / args.fps)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = 0.75, 65, -15
    near = mujoco.Renderer(env.sim.model, H, W)
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{2 * W}x{H}", "-r", str(args.fps), "-i", "-", "-pix_fmt", "yuv420p",
         "-vcodec", "libx264", "-crf", "23", "-movflags", "+faststart", str(args.out)],
        stdin=subprocess.PIPE,
    )  # fmt: skip

    def frame(text: str, color=(255, 255, 255)) -> None:
        cam.lookat[:] = env.sim.trunk_pose()[0]
        near.update_scene(env.sim.data, camera=cam)
        img = Image.fromarray(np.hstack([env.sim.render("overview", W, H), near.render()]))
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, 2 * W, 58], fill=(0, 0, 0))
        d.text((10, 6), name, fill=(200, 200, 200))
        d.text((10, 30), text, fill=color)
        ffmpeg.stdin.write(img.tobytes())

    def start(k: int) -> tuple[int, np.ndarray]:
        """Place the robot at move k's reference start; a start that cannot be held (all four
        balls hooked) is skipped, and the next move tried."""
        while True:
            env.only, env.chain = [k], False
            try:
                return k, env.reset()[0]
            except RuntimeError:
                print(f"move {k}: START NOT HELD (skipped)", flush=True)
                results.append("SKIPPED")
                for _ in range(args.fps // 2):
                    frame(f"move {k}: its reference start cannot be held, skipped", (240, 200, 60))
                k += 1

    results = []
    try:
        run(args, env, act, last, every, frame, start, results)
    finally:
        ffmpeg.stdin.close()
        ffmpeg.wait()
    n = sum(r == "SLOT-IN" for r in results)
    print(
        f"{name}: {n}/{len(results)} slot-ins, {results.count('FALL')} falls, "
        f"{results.count('SKIPPED')} skipped; saved {args.out}"
    )


def run(args, env, act, last, every, frame, start, results) -> None:
    """The chained climb from args.first to last."""
    steps = 0
    k, obs = start(args.first)
    while k <= last:
        move = env.moves[k]
        obs, _, terminated, truncated, info = env.step(
            act(obs) if act else np.zeros(len(tracking.JOINTS))
        )
        steps += 1
        done = terminated or truncated
        end = "SLOT-IN" if info["success"] else "FALL" if info["failed"] else "TIMEOUT"
        tally = f"{results.count('SLOT-IN')}/{len(results)} slot-ins so far"
        if steps % every == 0 or done:
            frame(f"move {k}/{last} ({move['limb']})   t = {env.t:5.2f} s   {tally}")
        if not done:
            continue
        results.append(end)
        print(f"move {k} ({move['limb']}): {end}", flush=True)
        color = (80, 220, 80) if end == "SLOT-IN" else (240, 80, 60)
        for _ in range(args.fps // 2):  # half a second on the outcome
            frame(f"move {k}/{last} ({move['limb']}): {end}   {tally}", color)
        k += 1
        if k > last:
            break
        if end == "SLOT-IN":  # carry on from where it is, as on the wall
            env.only, env.chain = None, True
            obs = env.reset()[0]
        else:
            k, obs = start(k)
            for _ in range(args.fps // 2):
                frame(f"RESET to the start of move {k}", (240, 200, 60))


if __name__ == "__main__":
    main()
