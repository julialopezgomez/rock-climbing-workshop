"""Fill the optimiser's move cache (rcr.optimiser.move_cache) by climbing generated walls.

    uv run python scripts/fill_cache.py --levels 1 2 3 --seeds 0-9 --height 0.8 --hours 8
    uv run python scripts/fill_cache.py --levels 2 --seeds 10-14 --no-store   # hit rate only

For each wall (rcr.sim.levels: level 1 ladder, 2 ladder with gaps, 3 drifting [Rules §3]) it climbs
from the start stance with a simple search: limbs in turn, each to the highest hold on its side
first; every move through the cheap feasibility checks, then the full solve with settles through
the cache; back up from a dead end. Every outcome is stored, feasible or not. It stops a wall at
--height (the hands) or after --wall-minutes, and everything after --hours. Safe to stop and
rerun: the cache is on disk and solved moves are not solved again.

This is a tool to fill the cache with the moves a climb asks for, not our planner (D2) [Ours].
On competition day: put the real limbs in rcr.robot.hooks, run the tests, and run this again;
the new version starts with an empty cache.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

import numpy as np

from rcr.optimiser import edge_checks as fe
from rcr.optimiser import move_cache as mc
from rcr.optimiser import transition as tr
from rcr.sim import levels
from rcr.types import LIMBS, Stance, WallModel

ORDER = ("right_hand", "left_hand", "right_foot", "left_foot")
HANDS = ("right_hand", "left_hand")


def candidates(wall: WallModel, st: Stance, last: str | None) -> list[Stance]:
    """Next stances: limbs in turn after the last one moved, each to the holds above it on its
    side (feet also on middle holds), highest first, without crossing hands or feet."""
    pos = {h.id: np.array(h.position) for h in wall.holds}
    start = (ORDER.index(last) + 1) % 4 if last else 0
    out = []
    for limb in ORDER[start:] + ORDER[:start]:
        side = "R" if limb.startswith("right") else "L"
        sides = (side,) if limb in HANDS else (side, "M")
        above = [
            h
            for h in pos
            if h[0] in sides
            and pos[h][2] > pos[st.contacts[limb]][2]
            and h not in st.contacts.values()
        ]
        for h in sorted(above, key=lambda h: -pos[h][2]):
            nxt = Stance({**st.contacts, limb: h})
            x = {limb: pos[nxt.contacts[limb]][0] for limb in LIMBS}
            if x["left_hand"] < x["right_hand"] and x["left_foot"] < x["right_foot"]:
                out.append(nxt)
    return out


def climb(wall, cfg, cache, stats, height, deadline, store=True):
    """Climb one wall until the hands reach `height` [m] or `deadline` (time.time())."""
    pos = {h.id: h.position[2] for h in wall.holds}
    stack = [(wall.start, None, candidates(wall, wall.start, None))]
    while stack and time.time() < deadline:
        st, last, todo = stack[-1]
        if max(pos[st.contacts[h]] for h in HANDS) >= height - 1e-9:
            return len(stack) - 1
        if not todo:
            stack.pop()
            continue
        nxt = todo.pop(0)
        moving = tr.moving_limb(st, nxt, 0)
        entry = cache.get(wall, st, nxt, cfg)
        if entry is None:
            verdict = fe.feasibility(wall, st, nxt, cfg)
            stats["checks"] += 1
            if not verdict.feasible:
                stats["rejected"] += 1
                if store:
                    cache.put(wall, st, nxt, cfg, reason=f"feasibility: {verdict.reason}")
                continue
        try:
            _, hit = mc.solve_move_cached(
                wall, st, nxt, cfg, cache if store else _ReadOnly(cache), len(stack) - 1
            )
        except tr.OptimisationError:
            stats["hit" if entry is not None else "failed"] += 1
            continue
        stats["hit" if hit else "solved"] += 1
        top = max(pos[nxt.contacts[h]] for h in HANDS)
        print(f"  {moving} -> {nxt.contacts[moving]}: {'from the cache' if hit else 'solved'}"
              f" (hands at {top:.2f} m)", flush=True)  # fmt: skip
        stack.append((nxt, moving, candidates(wall, nxt, moving)))
    return len(stack) - 1 if stack else 0


class _ReadOnly:
    """The cache, read but never written (to measure how many moves it already answers)."""

    def __init__(self, cache):
        self.cache = cache

    def get(self, *args):
        return self.cache.get(*args)

    def put(self, *args, **kwargs):
        pass


def seeds(text: str) -> list[int]:
    if "-" in text:
        lo, hi = text.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(text)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--seeds", default="0")
    ap.add_argument(
        "--height", type=float, default=0.8, help="[m] stop a wall when the hands are here"
    )
    ap.add_argument("--wall-minutes", type=float, default=60.0)
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--cache", default=str(mc.DEFAULT_DIR))
    ap.add_argument("--no-store", action="store_true", help="measure only: do not add to the cache")
    args = ap.parse_args()

    cfg = replace(tr.TransitionConfig(), settle=True)
    cache = mc.MoveCache(args.cache)
    end = time.time() + args.hours * 3600
    print(f"cache {cache.directory / cache.version(cfg)}: {cache.size(cfg)} (feasible, infeasible)")
    for level in args.levels:
        for seed in seeds(args.seeds) if level != 1 else [0]:
            if time.time() >= end:
                break
            wall = levels.generate(level, seed)
            stats = dict.fromkeys(("checks", "rejected", "solved", "failed", "hit"), 0)
            started = time.time()
            deadline = min(end, started + args.wall_minutes * 60)
            moves = climb(wall, cfg, cache, stats, args.height, deadline, not args.no_store)
            asked = stats["solved"] + stats["failed"] + stats["hit"]
            print(
                f"level {level} seed {seed}: {moves} moves up, {time.time() - started:.0f} s; "
                f"full solves {stats['solved']} ok / {stats['failed']} failed, "
                f"answered by the cache {stats['hit']} of {asked}; "
                f"feasibility checks {stats['checks']} ({stats['rejected']} rejected)",
                flush=True,
            )
    print(f"cache now: {cache.size(cfg)} (feasible, infeasible)")


if __name__ == "__main__":
    main()
