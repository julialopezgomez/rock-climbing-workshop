"""Centre of mass of two motions of the same plan, move by move: the optimiser's and an IK one.

    uv run --group optimiser scripts/compare_com.py build/plan_N1.json \
        build/trajectory_plan_N1.json build/wall-1-ik.npz

Both motions are put on the same samples per move: the optimiser's own (k = 0..N, the moving ball
let go at K), and the IK's moves matched to the plan's by stance (rcr.optimiser.reference.guesses)
and resampled there. At every sample: the CoM of each, the supports (the balls still on their
holds: all four up to K, the three others after), and how far the CoM is inside them sideways.

Sideways (x) is what "the CoM inside the support polygon" means here: gravity is along z, so the
support polygon is the contacts' projection on the floor plane (x, y), and the holds are all on the
wall face, so it is a thin strip and only its extent in x tells anything. The margin is the
CoM's distance inside [min x, max x] of the supports (negative: outside, on the side of the
moving limb). Writes build/com_<plan name>.json (every sample, for plots) and prints, per move, the
CoM's shift towards the supports before the release and the margin at it.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from test_optimiser import plan_walls  # noqa: E402  the wall the plan is on

from rcr import io  # noqa: E402
from rcr.optimiser import reference  # noqa: E402
from rcr.optimiser import robot_model as rm  # noqa: E402
from rcr.optimiser import transition as tr  # noqa: E402
from rcr.types import LIMBS, MICROBAN_JOINTS  # noqa: E402


def optimiser_moves(path: Path, n_moves: int, n: int) -> np.ndarray:
    """The saved trajectory's configurations, per move: n_moves x (n + 1) x NQ."""
    traj = io.load_trajectory(path)
    order = [traj.joint_names.index(name) for name in MICROBAN_JOINTS]
    q = np.hstack([np.asarray(traj.trunk_poses), np.asarray(traj.planned_positions)[:, order]])
    solved = (len(q) - 1) // n
    if solved < n_moves:
        print(f"{path}: {solved} of the plan's {n_moves} moves solved; comparing those")
    return np.array([q[i * n : i * n + n + 1] for i in range(min(solved, n_moves))])


def margin(x: float, supports: np.ndarray) -> float:
    """[m] how far x is inside [min, max] of the supports' x (negative outside)."""
    return float(min(x - supports.min(), supports.max() - x))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("plan", type=Path, help="e.g. build/plan_N1.json")
    parser.add_argument("optimised", type=Path, help="e.g. build/trajectory_plan_N1.json")
    parser.add_argument("ik", type=Path, help="e.g. build/wall-1-ik.npz (q: T x 26 at 50 Hz)")
    parser.add_argument("--out", type=Path, help="default: build/com_<plan name>.json")
    args = parser.parse_args()

    cfg = tr.TransitionConfig()
    n, k_release = cfg.n_samples, cfg.release_sample
    plan = io.load_contact_plan(args.plan)
    wall = plan_walls(plan)
    model = rm.load_robot_model()
    ours = optimiser_moves(args.optimised, len(plan.stances) - 1, n)
    theirs = reference.guesses(plan, wall, reference.read(args.ik), cfg)

    def com(q):
        return np.array(model.com(q)).ravel()

    moves, rows = [], []
    for i, (a, b) in enumerate(zip(plan.stances, plan.stances[1 : len(ours) + 1], strict=False)):
        if theirs[i] is None:
            print(f"move {i}: not in the IK motion, skipped")
            continue
        limb = tr.moving_limb(a, b, i)
        held = [m for m in LIMBS if m != limb]
        point = {m: tr.contact_point(wall, a.contacts[m], cfg) for m in LIMBS}
        support_x = np.array([point[m][0] for m in held])
        # towards the supports: from the moving limb's hold to the others' middle, in x
        towards = np.sign(support_x.mean() - point[limb][0]) or 1.0
        per = {}
        for name, qs in (("optimiser", ours[i]), ("ik", theirs[i])):
            c = np.array([com(q) for q in qs])
            per[name] = {
                "com": c.tolist(),
                "shift": float((c[k_release, 0] - c[0, 0]) * towards),
                "margin_release": margin(c[k_release, 0], support_x),
                "margin_min_air": min(margin(x, support_x) for x in c[k_release:, 0]),
            }
        gap = np.linalg.norm(np.array(per["optimiser"]["com"]) - np.array(per["ik"]["com"]), axis=1)
        moves.append(
            {
                "move": i,
                "limb": limb,
                "holds": {m: point[m].tolist() for m in LIMBS},
                "from": a.contacts[limb],
                "to": b.contacts[limb],
                "target": tr.contact_point(wall, b.contacts[limb], cfg).tolist(),
                "gap": gap.tolist(),
                **per,
            }
        )
        o, k = per["optimiser"], per["ik"]
        rows.append(
            f"{i:4d} {limb:10s} {o['shift'] * 1000:+7.1f} {k['shift'] * 1000:+7.1f}   "
            f"{o['margin_release'] * 1000:+7.1f} {k['margin_release'] * 1000:+7.1f}   "
            f"{o['margin_min_air'] * 1000:+7.1f} {k['margin_min_air'] * 1000:+7.1f}   "
            f"{gap.max() * 1000:6.1f}"
        )

    print("[mm]          shift before release   margin at release   min margin in air   max gap")
    print("move limb          TO      IK          TO      IK          TO      IK    TO-IK")
    print("\n".join(rows))
    for name in ("optimiser", "ik"):
        shifts = [m[name]["shift"] for m in moves]
        inside = [m[name]["margin_release"] for m in moves]
        print(
            f"{name:9s}: shift median {np.median(shifts) * 1000:+.1f} mm; margin at release "
            f"median {np.median(inside) * 1000:+.1f} mm, outside in {sum(x < 0 for x in inside)}"
            f" of {len(moves)} moves"
        )
    out = args.out or Path("build") / f"com_{args.plan.stem}.json"
    out.write_text(json.dumps({"plan": str(args.plan), "release": k_release, "moves": moves}))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
