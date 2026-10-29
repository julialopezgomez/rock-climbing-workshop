"""Convert an organisers' wall (third_party/simulator, climbing_sim/walls) to our wall/1 format.

uv run scripts/wall_generator.py N2            # -> data/examples/N2.json
uv run scripts/wall_generator.py N1 N2 N3      # several at once
uv run scripts/wall_generator.py my_wall.json --level 1   # a wall file of theirs, by path

Their wall lists holds by grid hole (col, row): the hold's left bolt hole, the right one two
columns over. Ours gives each hold an id and its position, the bolt midpoint on the wall face.
Here the id is "c<col>_r<row>" (as the simulator names its hold bodies), and the position comes
from their own wall model (climbing_sim.wall.Hold.bolt_axis), in the same wall frame as ours
(x right, z up, y into the wall) [simulator].
"""

import argparse
import importlib
import re
import sys
import types
from pathlib import Path

from rcr import io
from rcr.types import Hold, Stance, WallModel

REPO = Path(__file__).resolve().parent.parent
SIM_PACKAGE = REPO / "third_party" / "simulator" / "climbing_sim"
OUT = REPO / "data" / "examples"


def organisers_wall_module():
    """Their climbing_sim/wall.py, imported without the package's __init__ (which needs MuJoCo
    and microban_sim): wall.py itself needs only numpy."""
    if not SIM_PACKAGE.exists():
        sys.exit(f"no simulator at {SIM_PACKAGE}: run `git submodule update --init`")
    package = types.ModuleType("climbing_sim")
    package.__path__ = [str(SIM_PACKAGE)]
    sys.modules.setdefault("climbing_sim", package)
    return importlib.import_module("climbing_sim.wall")


def hold_id(hold) -> str:
    return f"c{hold.col}_r{hold.row}"


def convert(name: str, level: int | None = None) -> WallModel:
    """Their wall `name` (a bundled name such as "N2", or a JSON path) as our WallModel."""
    theirs = organisers_wall_module().Wall.load(name)
    if level is None:  # their bundled walls say "Level k route" in their description
        found = re.search(r"Level (\d)", theirs.description)
        if not found:
            sys.exit(f"{name}: no level in its description, give one with --level")
        level = int(found.group(1))
    holds = [
        Hold(id=hold_id(h), position=tuple(round(float(v), 6) for v in h.bolt_axis))
        for h in theirs.holds
    ]
    start = Stance(contacts={limb: hold_id(h) for limb, h in theirs.start.items()})
    return WallModel(id=theirs.name, level=level, holds=holds, start=start)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("walls", nargs="+", help="their wall names (N1, N2, N3) or JSON files")
    parser.add_argument("--level", type=int, choices=(1, 2, 3), help="if the wall does not say")
    parser.add_argument("--out", type=Path, default=OUT, help=f"folder (default {OUT})")
    args = parser.parse_args()
    for name in args.walls:
        wall = convert(name, args.level)
        path = args.out / f"{wall.id}.json"
        io.save_wall(wall, path)
        print(f"{name}: {len(wall.holds)} holds, level {wall.level} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
