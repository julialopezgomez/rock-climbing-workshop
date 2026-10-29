"""Remember feasibility verdicts by the move's layout, across walls and runs [Ours].

Holds are identical, on a 15 mm grid, under fixed gravity, so whether a move can be done depends
on where the holds are relative to each other, not on where they are on the wall. The key is the
moving limb, the grid offsets of the four holds of the stance before and of the new hold, and
those of every other hold within CACHE_RADIUS of them (a shin can hit the hold above a foot), all
relative to the moving limb's old hold. A move translated anywhere on any wall hits the same
entry. Walls off the grid are not cached.

    feasible = Cache(optimiser.feasibility, "build/feasibility_cache.json", version)
    plan = planner.plan(wall, feasible)

`version` names what the verdicts depend on (the hooks, the optimiser's settings, the hold);
a file with another version is ignored and overwritten. The planner never imports the optimiser:
the caller passes it in.
"""

import json
import math
import os
from pathlib import Path

from rcr.types import LIMBS, Feasibility, FeasibilityFn, Stance, WallModel
from rcr.walls import grid_index

SCHEMA = "feasibility_cache/1"
CACHE_RADIUS = 0.25  # [m] other holds this close to a hold of the move are part of the key [Ours]


def layout_key(wall: WallModel, a: Stance, b: Stance) -> str | None:
    """The move's layout, as a string; None if a hold is off the grid or the move is not one
    limb changing hold."""
    moved = [limb for limb in LIMBS if a.contacts[limb] != b.contacts[limb]]
    if len(moved) != 1:
        return None
    (limb,) = moved
    holds = {h.id: h for h in wall.holds}
    involved = [a.contacts[x] for x in LIMBS] + [b.contacts[limb]]
    cells = {i: grid_index(holds[i]) for i in involved if i in holds}
    if len(cells) < len(set(involved)) or None in cells.values():
        return None
    i0, j0 = cells[a.contacts[limb]]

    def offset(cell: tuple[int, int]) -> str:
        return f"{cell[0] - i0},{cell[1] - j0}"

    near = []
    for h in wall.holds:
        if h.id in cells:
            continue
        x, _, z = h.position
        if any(
            math.hypot(x - holds[i].position[0], z - holds[i].position[2]) <= CACHE_RADIUS
            for i in cells
        ):
            cell = grid_index(h)
            if cell is None:
                return None
            near.append(offset(cell))
    parts = [offset(cells[i]) for i in involved]
    return f"{limb}|{';'.join(parts)}|{';'.join(sorted(near))}"


class Cache:
    """A FeasibilityFn that asks `feasible` once per layout and remembers the verdict in a file."""

    def __init__(self, feasible: FeasibilityFn, path: Path | str | None = None, version: str = ""):
        self.feasible, self.version = feasible, version
        self.path = Path(path) if path else None
        self.entries: dict[str, list] = {}
        self.hits = self.misses = 0
        if self.path and self.path.exists():
            data = json.loads(self.path.read_text())
            if data.get("schema") == SCHEMA and data.get("version") == version:
                self.entries = data["entries"]

    def __call__(self, wall: WallModel, a: Stance, b: Stance) -> Feasibility:
        k = layout_key(wall, a, b)
        if k is not None and k in self.entries:
            self.hits += 1
            feasible, cost, reason = self.entries[k]
            return Feasibility(feasible, cost, f"cached: {reason}" if reason else "cached")
        self.misses += 1
        verdict = self.feasible(wall, a, b)
        if k is not None:
            self.entries[k] = [verdict.feasible, verdict.cost, verdict.reason]
            self.save()  # every verdict costs seconds: keep it even if the run stops
        return verdict

    def save(self) -> None:
        """Write the verdicts, merged with any another run saved meanwhile (same version), in
        one atomic step, so runs in parallel keep each other's verdicts."""
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                data = {}
            if data.get("schema") == SCHEMA and data.get("version") == self.version:
                self.entries = {**data["entries"], **self.entries}
        data = {"schema": SCHEMA, "version": self.version, "entries": self.entries}
        partial = self.path.with_suffix(f".{os.getpid()}.tmp")
        partial.write_text(json.dumps(data) + "\n")
        os.replace(partial, self.path)
