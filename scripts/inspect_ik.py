"""Inspect one climbing IK move: target tracking plots and optional MuJoCo replay.

Examples:
    uv run --with matplotlib python scripts/inspect_ik.py --plot build/ik.png
    uv run --with matplotlib python scripts/inspect_ik.py --viewer
    # On macOS, replace 'python' with 'mjpython' for the MuJoCo viewer.

Purely kinematic replay: MuJoCo receives qpos directly; no dynamics/servo tracking.
"""
import argparse
from pathlib import Path

import numpy as np

from rcr import io
from rcr.kinematics import ik
from rcr.types import LIMBS, Stance


def generate(wall, limb, destination):
    """One movement, returning sampled generalized configurations and target balls."""
    a = wall.start
    b = Stance({**a.contacts, limb: destination})
    q0 = ik.start_pose(wall, a)
    qs_move, worst = ik.track_move(wall, a, b, q0)
    qs = np.asarray([q0, *qs_move])
    n = round(ik.MOVE_TIME / ik.DT)
    times = np.arange(len(qs)) * ik.DT
    initial = ik.sockets(wall, a)
    target_b = ik.sockets(wall, b)
    B = ik.ball_path(initial[LIMBS.index(limb)], target_b[LIMBS.index(limb)])
    targets = np.repeat(initial[None, :, :], len(qs), axis=0)
    targets[:n + 1, LIMBS.index(limb)] = [
        B(min(k * ik.DT, ik.MOVE_TIME)) for k in range(n + 1)
    ]
    targets[n + 1:] = target_b
    actual = np.asarray([ik.balls(q) for q in qs])
    return a, b, qs, times, targets, actual, worst


def plot_diagnostics(qs, t, target, actual, moving, filename):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    i = LIMBS.index(moving)

    # Compare target vs achieved end-effector coordinates (wall x/y/z).
    for axis, label in enumerate("xyz"):
        axes[0, 0].plot(t, target[:, i, axis], "--", label=f"{label} target")
        axes[0, 0].plot(t, actual[:, i, axis], label=f"{label} actual")
    axes[0, 0].set_title(f"Moving limb: {moving} (m)")
    axes[0, 0].legend(ncol=3, fontsize=8)

    for j, limb in enumerate(LIMBS):
        err = np.linalg.norm(target[:, j] - actual[:, j], axis=1) * 1000
        axes[0, 1].plot(t, err, label=limb)
    axes[0, 1].axhline(ik.MAX_ERROR * 1000, color="red", linestyle="--", label="MAX_ERROR")
    axes[0, 1].set_title("End-effector tracking error (mm)")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(actual[:, i, 0], actual[:, i, 2], label="actual")
    axes[1, 0].plot(target[:, i, 0], target[:, i, 2], "--", label="target")
    axes[1, 0].set(xlabel="wall x (m)", ylabel="wall z (m)", title="Moving ball, wall view")
    axes[1, 0].axis("equal")
    axes[1, 0].legend()

    for j, limb in enumerate(LIMBS):
        if j != i:
            axes[1, 1].plot(t, np.linalg.norm(actual[:, j] - target[:, j], axis=1) * 1000,
                            label=limb)
    axes[1, 1].set_title("Held-contact drift (mm)")
    axes[1, 1].legend(fontsize=8)

    joints = qs[:, 7:]
    v = np.diff(joints, axis=0) / np.diff(t)[:, None]
    axes[2, 0].plot(t, joints, linewidth=0.8)
    axes[2, 0].set_title("19 joint positions (rad)")
    axes[2, 1].plot(t[1:], v, linewidth=0.8)
    axes[2, 1].set_title("Finite-difference joint velocities (rad/s)")
    for ax in axes.flat:
        ax.grid(alpha=0.2)
        if ax not in (axes[1, 0],):
            ax.set_xlabel("time (s)")
    out = Path(filename)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"Plots: {out}")
    plt.close(fig)


def replay(wall, qs, dt):
    """Set MuJoCo qpos for each sample; do not advance the physics integrator."""
    import time
    import mujoco
    import mujoco.viewer

    from rcr.planner.astar import qpos, scene

    model, data, *_ = scene(wall)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for q in qs:
            if not viewer.is_running():
                break
            data.qpos[:] = qpos(model, q)
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(dt)
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wall", default="data/examples/N1.json")
    parser.add_argument("--limb", choices=LIMBS, default="right_hand")
    parser.add_argument("--destination", default="c29_r27",
                        help="hold id of moving limb, on chosen wall")
    parser.add_argument("--plot", default="build/ik_diagnostics.png")
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()

    wall = io.load_wall(args.wall)
    a, b, qs, times, target, actual, worst = generate(
        wall, args.limb, args.destination
    )
    errors = np.linalg.norm(target - actual, axis=2)
    moving = LIMBS.index(args.limb)
    held = [j for j in range(len(LIMBS)) if j != moving]
    print(f"{a.contacts[args.limb]} -> {b.contacts[args.limb]}")
    print(f"Moving max error: {1000 * errors[:, moving].max():.2f} mm")
    print(f"Held max error: {1000 * errors[:, held].max():.2f} mm")
    print(f"Reported moving error: {1000 * worst:.2f} mm")
    print(f"Max joint speed: {np.max(np.abs(np.diff(qs[:, 7:], axis=0) / ik.DT)):.2f} rad/s")
    plot_diagnostics(qs, times, target, actual, args.limb, args.plot)
    if args.viewer:
        replay(wall, qs, ik.DT)


if __name__ == "__main__":
    main()
