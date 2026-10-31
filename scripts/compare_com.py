"""Centre of mass of two motions of the same plan, move by move: the optimiser's and an IK one.

    uv run --group optimiser scripts/compare_com.py build/plan_N1.json \
        build/trajectory_plan_N1.json build/wall-1-ik.npz

Both motions are put on the same samples per move: the optimiser's own (k = 0..N, the moving ball
let go at K), and the IK's moves matched to the plan's by stance (rcr.optimiser.reference.guesses)
and resampled there. At every sample: the CoM of each, the supports (the balls still on their
holds: all four up to K, the three others after) and their static-equilibrium polygon.

The polygon is Stéphane Caron's, from the workshop's tutorial (static_balance; Bretl and Lall,
2008): the region, seen from above (x, y), where the CoM can be while the supporting balls hold
the robot still. Each ball is a point contact pushing up within a cone of half-angle 80° (the
cup's surface normals) with at most 5 N (what an XL330 can hold at a ball) [the tutorial's
values; --friction-deg, --max-force]. The margin is the CoM's signed distance to the polygon's
edge: positive inside, negative outside, NaN where no polygon exists (the supports cannot hold
the robot whatever the CoM). Writes build/com_<plan name>.json (every sample, for plots) and
prints, per move, the margin at the release and its smallest value in the air, both motions.
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

FRICTION_DEG = 80.0  # [deg] the cups' normals around the vertical [Caron's tutorial]
MAX_FORCE = 5.0  # [N] a ball's largest force [Caron's tutorial]


def optimiser_moves(path: Path, n_moves: int, n: int) -> np.ndarray:
    """The saved trajectory's configurations, per move: n_moves x (n + 1) x NQ."""
    traj = io.load_trajectory(path)
    order = [traj.joint_names.index(name) for name in MICROBAN_JOINTS]
    q = np.hstack([np.asarray(traj.trunk_poses), np.asarray(traj.planned_positions)[:, order]])
    solved = (len(q) - 1) // n
    if solved < n_moves:
        print(f"{path}: {solved} of the plan's {n_moves} moves solved; comparing those")
    return np.array([q[i * n : i * n + n + 1] for i in range(min(solved, n_moves))])


def polygon(balls: np.ndarray, mass: float, friction: float, max_force: float):
    """The static-equilibrium polygon (vertices x, y, counterclockwise) of balls (B x 3) in cups,
    or None if they cannot hold the robot."""
    import pypoman
    from scipy.spatial import QhullError
    from static_balance import ContactSet, PointContact, compute_static_equilibrium_polygon

    contacts = ContactSet(PointContact(p, [0.0, 0.0, 1.0], friction, max_force) for p in balls)
    try:
        return compute_static_equilibrium_polygon(contacts, mass)
    except (pypoman.Exception, QhullError, ValueError):
        return None


def signed_margin(point: np.ndarray, vertices) -> float:
    """[m] distance from point (x, y) to the polygon's edge: positive inside, negative outside."""
    if vertices is None:
        return float("nan")
    a, b = vertices, np.roll(vertices, -1, axis=0)
    edge = b - a
    s = np.clip(np.einsum("ij,ij->i", point - a, edge) / np.einsum("ij,ij->i", edge, edge), 0, 1)
    distance = np.linalg.norm(point - (a + s[:, None] * edge), axis=1).min()
    inside = np.all(edge[:, 0] * (point[1] - a[:, 1]) - edge[:, 1] * (point[0] - a[:, 0]) >= 0)
    return float(distance if inside else -distance)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("plan", type=Path, help="e.g. build/plan_N1.json")
    parser.add_argument("optimised", type=Path, help="e.g. build/trajectory_plan_N1.json")
    parser.add_argument("ik", type=Path, help="e.g. build/wall-1-ik.npz (q: T x 26 at 50 Hz)")
    parser.add_argument("--friction-deg", type=float, default=FRICTION_DEG, help="cone half-angle")
    parser.add_argument("--max-force", type=float, default=MAX_FORCE, help="[N] per ball")
    parser.add_argument("--out", type=Path, help="default: build/com_<plan name>.json")
    args = parser.parse_args()

    cfg = tr.TransitionConfig()
    n, k_release = cfg.n_samples, cfg.release_sample
    friction = float(np.tan(np.radians(args.friction_deg)))
    plan = io.load_contact_plan(args.plan)
    wall = plan_walls(plan)
    model = rm.load_robot_model()
    ours = optimiser_moves(args.optimised, len(plan.stances) - 1, n)
    theirs = reference.guesses(plan, wall, reference.read(args.ik), cfg)

    moves, rows = [], []
    for i, (a, b) in enumerate(zip(plan.stances, plan.stances[1 : len(ours) + 1], strict=False)):
        if theirs[i] is None:
            print(f"move {i}: not in the IK motion, skipped")
            continue
        limb = tr.moving_limb(a, b, i)
        held = [LIMBS.index(m) for m in LIMBS if m != limb]
        per = {}
        for name, qs in (("optimiser", ours[i]), ("ik", theirs[i])):
            com, polygons, margins = [], [], []
            for k, q in enumerate(qs):
                c = np.array(model.com(q)).ravel()
                balls = np.array(model.limb_positions(q)).T  # 4 x 3, LIMBS order
                vertices = polygon(
                    balls if k < k_release else balls[held], model.mass, friction, args.max_force
                )
                com.append(c.tolist())
                polygons.append(None if vertices is None else vertices.tolist())
                margins.append(signed_margin(c[:2], vertices))
            air = np.array(margins[k_release:])
            per[name] = {
                "com": com,
                "polygons": polygons,
                "margins": margins,
                "margin_release": margins[k_release],
                "margin_min_air": float(np.nanmin(air)) if np.isfinite(air).any() else None,
                "no_polygon": int(np.isnan(margins).sum()),
            }
        moves.append(
            {
                "move": i,
                "limb": limb,
                "from": a.contacts[limb],
                "to": b.contacts[limb],
                "holds": {m: tr.contact_point(wall, a.contacts[m], cfg).tolist() for m in LIMBS},
                "target": tr.contact_point(wall, b.contacts[limb], cfg).tolist(),
                **per,
            }
        )

        def mm(v):
            return "    -- " if v is None or np.isnan(v) else f"{v * 1000:+7.1f}"

        o, k = per["optimiser"], per["ik"]
        rows.append(
            f"{i:4d} {limb:10s} {mm(o['margin_release'])} {mm(k['margin_release'])}   "
            f"{mm(o['margin_min_air'])} {mm(k['margin_min_air'])}"
        )

    print(f"static-equilibrium polygon: cone {args.friction_deg:g} deg, {args.max_force:g} N/ball")
    print("[mm]          margin at release   min margin in air")
    print("move limb          TO      IK          TO      IK")
    print("\n".join(rows))
    for name in ("optimiser", "ik"):
        release = np.array([m[name]["margin_release"] for m in moves], dtype=float)
        air = np.array([np.nan if m[name]["margin_min_air"] is None else m[name]["margin_min_air"]
                        for m in moves])  # fmt: skip
        print(
            f"{name:9s}: at release median {np.nanmedian(release) * 1000:+.1f} mm, outside in "
            f"{int((release < 0).sum())} of {len(moves)} moves; in the air outside in "
            f"{int((air < 0).sum())}; samples with no polygon "
            f"{sum(m[name]['no_polygon'] for m in moves)}"
        )
    out = args.out or Path("build") / f"com_{args.plan.stem}.json"
    out.write_text(
        json.dumps(
            {
                "plan": str(args.plan),
                "release": k_release,
                "friction_deg": args.friction_deg,
                "max_force": args.max_force,
                "moves": moves,
            }
        )
    )
    print(f"-> {out}")


if __name__ == "__main__":
    main()
