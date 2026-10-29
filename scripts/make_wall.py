"""Generate a random wall for a level (rcr.walls); show it, save it as wall/1, or check a wall file.

uv run scripts/make_wall.py --level 2 --seed 3 --show     # print the layout, save nothing
uv run scripts/make_wall.py --level 2 --seed 3 --save     # -> data/walls/gen_l2_s3.json
uv run scripts/make_wall.py --level 3 --out x.json        # save to a given path
uv run scripts/make_wall.py --check data/examples/N1.json   # rule violations of a file
"""

import argparse
import sys
from pathlib import Path

from rcr import io, walls

WALLS = Path(__file__).resolve().parent.parent / "data" / "walls"


def check(path: Path) -> int:
    wall = io.load_wall(path)
    problems, crowded = walls.violations(wall), walls.crowding(wall)
    print(walls.draw(wall))
    for line in problems:
        print(f"  rules: {line}")
    for line in crowded:
        print(f"  crowded [Ours]: {line}")
    print(f"{path}: {len(problems)} rule violations, {len(crowded)} crowded pairs")
    return 1 if problems else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--level", type=int, choices=(1, 2, 3))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show", action="store_true", help="print the layout")
    parser.add_argument("--save", action="store_true", help="save to data/walls/<wall id>.json")
    parser.add_argument("--out", type=Path, help="save to this path instead")
    parser.add_argument("--check", type=Path, help="check this wall/1 file instead")
    args = parser.parse_args()
    if args.check:
        return check(args.check)
    if args.level is None:
        parser.error("--level is required unless --check is given")

    wall = walls.generate(args.level, args.seed)
    print(
        walls.draw(wall) if args.show else f"{wall.id}: level {wall.level}, {len(wall.holds)} holds"
    )
    if args.save or args.out:
        out = args.out or WALLS / f"{wall.id}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        io.save_wall(wall, out)
        print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
