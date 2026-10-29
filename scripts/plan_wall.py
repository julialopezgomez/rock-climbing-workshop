"""Plan a climb on a wall, with the optimiser's feasibility (cached) or the reach graph only.

uv run scripts/plan_wall.py --level 1 --seed 0                      # reach graph only
uv run scripts/plan_wall.py --level 1 --seed 0 --feasibility optimiser --save build/plan.json
uv run scripts/plan_wall.py --wall data/examples/wall_hard.json --feasibility optimiser
uv run scripts/plan_wall.py --warm 5 --feasibility optimiser    # 5 walls a level: fill the cache

`--feasibility optimiser` needs `uv sync --group optimiser` and MICROBAN_URDF (optimiser README).
Its verdicts are kept in build/feasibility_cache.json, keyed by the move's layout
(rcr.planner.cache): the same move anywhere, on any wall, is asked once. The cache's version is a
hash of the optimiser's code, the hooks and the hold mesh, so changing any of them starts it again.
"""

import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

from rcr import io, walls
from rcr.planner import astar
from rcr.planner.cache import Cache
from rcr.types import Feasibility

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "build" / "feasibility_cache.json"


def optimiser_feasibility(path: Path = CACHE) -> Cache:
    """The optimiser's feasibility behind the layout cache in `path`."""
    from rcr.optimiser import feasibility

    digest = hashlib.sha256()
    for f in sorted((ROOT / "src" / "rcr" / "optimiser").glob("*.py")):
        digest.update(f.read_bytes())
    for f in (ROOT / "src" / "rcr" / "robot" / "hooks.py", ROOT / "data" / "holds" / "hold.stl"):
        digest.update(f.read_bytes())
    return Cache(feasibility, path, version=digest.hexdigest()[:16])


def reach_only(wall, a, b) -> Feasibility:
    return Feasibility(True, reason="reach graph only")


def report(wall, result: astar.Result, cache: Cache | None) -> None:
    hits = f", {cache.hits} from the cache" if cache else ""
    top = "reached" if result.reached_top else "NOT reached"
    print(
        f"{wall.id}: {len(result.plan.stances) - 1} moves, top {top}; "
        f"{result.checks} feasibility checks{hits}, {len(result.rejected)} rejected; "
        f"{result.expansions} stances expanded; {result.seconds:.1f} s"
    )
    reasons = Counter(reason.split(":")[0] for _, _, reason in result.rejected)
    if reasons:
        print("  rejected at: " + ", ".join(f"{r} {n}" for r, n in reasons.most_common()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--level", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wall", type=Path, help="a wall/1 file instead of a generated wall")
    parser.add_argument("--feasibility", choices=("reach", "optimiser"), default="reach")
    parser.add_argument("--weight", type=float, default=astar.WEIGHT)
    parser.add_argument("--warm", type=int, default=0, help="plan this many walls per level")
    parser.add_argument("--save", type=Path, help="save the plan (contact_plan/1)")
    args = parser.parse_args()

    cache = optimiser_feasibility() if args.feasibility == "optimiser" else None
    feasible = cache or reach_only
    if args.warm:
        for level in (1, 2, 3):
            for seed in range(args.warm):
                wall = walls.generate(level, seed)
                report(wall, astar.solve(wall, feasible, weight=args.weight), cache)
        return 0
    wall = io.load_wall(args.wall) if args.wall else walls.generate(args.level, args.seed)
    result = astar.solve(wall, feasible, weight=args.weight)
    report(wall, result, cache)
    for a, b in zip(result.plan.stances, result.plan.stances[1:], strict=False):
        (limb,) = [x for x in a.contacts if a.contacts[x] != b.contacts[x]]
        print(f"  {limb:<11} {a.contacts[limb]:>4} -> {b.contacts[limb]}")
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        io.save_contact_plan(result.plan, args.save)
        print(f"saved {args.save}")
    return 0 if result.reached_top else 1


if __name__ == "__main__":
    sys.exit(main())
