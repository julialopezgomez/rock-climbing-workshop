"""Measure the planner against the optimiser: what its feasibility accepts, and how plans fare.

uv run scripts/plan_eval.py moves --per-bin 10          # sample reach-graph moves, ask feasibility
uv run scripts/plan_eval.py plans --seeds 3 --solve     # plan generated walls, then optimise them

Both need `uv sync --group optimiser` and MICROBAN_URDF, and share build/feasibility_cache.json
with scripts/plan_wall.py, so every verdict is computed once.

moves: random walks on the reach graph of generated walls; moves are binned by how far the limb
rises, and up to --per-bin per (hand or foot, bin) are checked. It prints, per bin, how many the
optimiser accepts and how long the checks take, to set the planner's move limits from data.
plans: each wall is planned with the cached feasibility; with --solve the plan is optimised in
full, and a move that fails there is a move feasibility passed wrongly. Results go to
build/plan_eval_<mode>.json.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

from plan_wall import optimiser_feasibility

from rcr import walls
from rcr.planner import astar
from rcr.types import LIMBS

ROOT = Path(__file__).resolve().parent.parent
BINS = (0.0, 0.06, 0.12, 0.18, 0.24, 0.30, 1.0)  # [m] rise of the moving limb


def sample_moves(levels, seeds, walk: int, rng) -> list:
    """(wall, a, b, limb, rise, dx) along random walks on the reach graph."""
    out = []
    for level in levels:
        for seed in seeds:
            wall = walls.generate(level, seed)
            graph = astar.Graph(wall, astar.rc.load())
            k = astar.key(wall.start)
            for _ in range(walk):
                children = [c for c in graph.children(k) if graph.valid(c)]
                if not children:
                    break
                for c in children:
                    (m,) = [n for n in range(4) if c[n] != k[n]]
                    rise = graph.z[c[m]] - graph.z[k[m]]
                    dx = graph.point[c[m]][0] - graph.point[k[m]][0]
                    out.append((wall, astar.stance(k), astar.stance(c), LIMBS[m], rise, dx))
                k = rng.choice(children)
    return out


def moves(args) -> dict:
    rng = random.Random(args.seed)
    pool = sample_moves(args.levels, range(args.walls), args.walk, rng)
    rng.shuffle(pool)
    feasible = optimiser_feasibility()
    taken: dict[tuple, int] = {}
    rows = []
    for wall, a, b, limb, rise, dx in pool:
        kind = "hand" if "hand" in limb else "foot"
        bin_ = next(i for i in range(len(BINS) - 1) if rise < BINS[i + 1])
        if taken.get((kind, bin_), 0) >= args.per_bin:
            continue
        taken[(kind, bin_)] = taken.get((kind, bin_), 0) + 1
        start = time.perf_counter()
        hits = feasible.hits
        verdict = feasible(wall, a, b)
        rows.append(
            dict(
                wall=wall.id,
                limb=limb,
                rise=round(rise, 4),
                dx=round(dx, 4),
                bin=bin_,
                feasible=verdict.feasible,
                cost=verdict.cost,
                reason=verdict.reason,
                seconds=round(time.perf_counter() - start, 2),
                cached=feasible.hits > hits,
            )
        )
        answer = "ok " if verdict.feasible else "NO "
        print(
            f"{len(rows):3d} {limb:<11} rise {rise:+.3f} dx {dx:+.3f}: "
            f"{answer} {rows[-1]['seconds']:5.1f} s  {verdict.reason[:60]}",
            flush=True,
        )
    print("\nrise [m]        hands ok/n     feet ok/n")
    for i in range(len(BINS) - 1):
        cells = []
        for kind in ("hand", "foot"):
            r = [x for x in rows if x["bin"] == i and kind in x["limb"]]
            cells.append(f"{sum(x['feasible'] for x in r):3d}/{len(r):<3d}")
        print(f"{BINS[i]:.2f}-{BINS[i + 1]:.2f}     {cells[0]:>10}     {cells[1]:>10}")
    return dict(rows=rows)


def plans(args) -> dict:
    feasible = optimiser_feasibility()
    rows = []
    for level in args.levels:
        for seed in range(args.seeds):
            wall = walls.generate(level, seed)
            hits = feasible.hits
            result = astar.solve(wall, feasible)
            row = dict(
                wall=wall.id,
                moves=len(result.plan.stances) - 1,
                top=result.reached_top,
                checks=result.checks,
                cached=feasible.hits - hits,
                rejected=len(result.rejected),
                searches=result.searches,
                expansions=result.expansions,
                seconds=round(result.seconds, 1),
            )
            if args.solve:
                from rcr.optimiser import optimise

                start = time.perf_counter()
                try:
                    optimise(wall, result.plan)
                    row["solve"] = "ok"
                except Exception as e:  # the optimiser names the failing transition
                    row["solve"] = f"failed: {e}"[:200]
                row["solve_seconds"] = round(time.perf_counter() - start, 1)
            rows.append(row)
            print(json.dumps(row), flush=True)
    return dict(rows=rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    m = sub.add_parser("moves")
    m.add_argument("--per-bin", type=int, default=8)
    m.add_argument("--walls", type=int, default=3, help="seeds per level to walk on")
    m.add_argument("--walk", type=int, default=12, help="steps per random walk")
    p = sub.add_parser("plans")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--solve", action="store_true")
    for s in (m, p):
        s.add_argument("--levels", type=lambda t: [int(x) for x in t.split(",")], default=[1, 2, 3])
        s.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = moves(args) if args.mode == "moves" else plans(args)
    path = ROOT / "build" / f"plan_eval_{args.mode}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"saved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
