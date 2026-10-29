"""Convert an organisers' wall (third_party/simulator, climbing_sim/walls) to our wall/1 format.

uv run scripts/wall_generator.py N2            # -> data/examples/N2.json
uv run scripts/wall_generator.py N1 N2 N3      # several at once
uv run scripts/wall_generator.py my_wall.json --level 1   # a wall file of theirs, by path
uv run scripts/wall_generator.py --level 3 --seed 5   # a new route of theirs (climbing_sim.routes)

Their wall lists holds by grid hole (col, row): the hold's left bolt hole, the right one two
columns over. Ours gives each hold an id and its position, the bolt midpoint on the wall face.
The id is "c<col>_r<row>" (as the simulator names its hold bodies), and the position comes from
their own wall model (Hold.bolt_axis), in the same wall frame as ours (rcr.walls.from_organisers).
Needs `uv sync --group sim`.
"""

import argparse
import re
import sys
from pathlib import Path

from climbing_sim import Wall
from climbing_sim.routes import generate_wall

from rcr import io, walls
from rcr.types import WallModel

OUT = Path(__file__).resolve().parent.parent / "data" / "examples"


def convert(name: str, level: int | None = None) -> WallModel:
    """Their wall `name` (a bundled name such as "N2", or a JSON path) as our WallModel."""
    theirs = Wall.load(name)
    if level is None:  # their bundled walls say "Level k route" in their description
        found = re.search(r"Level (\d)", theirs.description)
        if not found:
            sys.exit(f"{name}: no level in its description, give one with --level")
        level = int(found.group(1))
    return walls.from_organisers(theirs, level)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("walls", nargs="*", help="their wall names (N1, N2, N3) or JSON files")
    parser.add_argument("--level", type=int, choices=(1, 2, 3), help="if the wall does not say")
    parser.add_argument("--seed", type=int, help="a new route of theirs for --level, this seed")
    parser.add_argument("--out", type=Path, default=OUT, help=f"folder (default {OUT})")
    args = parser.parse_args()
    if args.seed is not None:
        if args.level is None:
            parser.error("--seed needs --level")
        theirs = generate_wall(args.level, args.seed)
        made = [walls.from_organisers(theirs, args.level)]
    elif args.walls:
        made = [convert(name, args.level) for name in args.walls]
    else:
        parser.error("give wall names, or --level and --seed")
    for wall in made:
        path = args.out / f"{wall.id}.json"
        io.save_wall(wall, path)
        print(f"{wall.id}: {len(wall.holds)} holds, level {wall.level} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
