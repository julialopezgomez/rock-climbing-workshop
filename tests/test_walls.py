"""Generated walls: they keep the rules, each level looks like its rule, and the checks catch what
breaks a rule. Run directly to print walls: `uv run python tests/test_walls.py 3 0 1 2`."""

import dataclasses
import subprocess
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from rcr import io, walls
from rcr.types import GRID_PITCH, Hold

ROOT = Path(__file__).resolve().parent.parent
SEEDS = range(100)
CASES = [(level, seed) for level in (1, 2, 3) for seed in SEEDS]


def sides(wall):
    """The holds of the left and the right column, bottom to top (generated walls only): each hold
    joins the column whose last hold is nearest sideways, starting from the start feet."""
    holds = {h.id: h for h in wall.holds}
    left, right = [holds[wall.start.contacts[f]] for f in ("left_foot", "right_foot")]
    columns = [[left], [right]]
    for h in sorted(wall.holds, key=lambda h: h.position[2]):
        if h not in (left, right):
            min(columns, key=lambda c: abs(c[-1].position[0] - h.position[0])).append(h)
    return columns


@pytest.mark.parametrize(("level", "seed"), CASES)
def test_generated_walls_keep_the_rules(level, seed, tmp_path):
    wall = walls.generate(level, seed)
    assert walls.violations(wall) == []
    assert walls.crowding(wall) == []
    io.save_wall(wall, tmp_path / "wall.json")
    assert io.load_wall(tmp_path / "wall.json") == wall


@pytest.mark.parametrize(("level", "seed"), CASES)
def test_the_start_is_the_example_walls(level, seed):
    wall = walls.generate(level, seed)
    holds = {h.id: h.position for h in wall.holds}
    c = wall.start.contacts
    for side, (foot, hand) in zip(
        sides(wall), (("left_foot", "left_hand"), ("right_foot", "right_hand")), strict=True
    ):
        assert c[foot] == side[0].id, "feet on the lowest hold of their side"
        above = holds[c[hand]][2] - holds[c[foot]][2]
        assert 0.15 <= above <= 0.25
    assert holds[c["left_foot"]][0] < holds[c["right_foot"]][0]
    assert 1.27 - 0.075 < max(h.position[2] for h in wall.holds) <= 1.27 + 1e-9


@pytest.mark.parametrize("seed", SEEDS)
def test_level1_is_a_regular_ladder(seed):
    for side in sides(walls.generate(1, seed)):
        assert len({h.position[0] for h in side}) == 1, "one column"
        rises = {round(b.position[2] - a.position[2], 6) for a, b in pairwise(side)}
        assert len(rises) == 1 and rises <= {0.06, 0.075}


@pytest.mark.parametrize("seed", SEEDS)
def test_level2_is_a_ladder_with_missing_holds(seed):
    wall = walls.generate(2, seed)
    assert sum(len(side) for side in sides(wall)) == len(wall.holds)
    gaps = []
    for side in sides(wall):
        assert len({h.position[0] for h in side}) == 1, "one column"
        gaps += [round(b.position[2] - a.position[2], 6) for a, b in pairwise(side)]
    assert set(gaps) <= {0.06, 0.12}, "rungs 60 mm apart, never two missing in a row"
    assert 0.12 in gaps, "at least one hold missing"


@pytest.mark.parametrize("seed", SEEDS)
def test_level3_needs_lateral_moves(seed):
    wall = walls.generate(3, seed)
    for side in sides(wall):
        xs = [h.position[0] for h in side]
        assert max(abs(b - a) for a, b in pairwise(xs)) <= 2 * GRID_PITCH + 1e-9
        assert max(xs) - min(xs) >= 8 * GRID_PITCH - 1e-9, "the columns move at least 0.12 m"


def test_same_seed_same_wall_and_seeds_differ():
    assert walls.generate(3, 7) == walls.generate(3, 7)
    for level in (1, 2, 3):
        assert len({str(walls.generate(level, s).holds) for s in range(10)}) > 5


def test_bad_level():
    with pytest.raises(ValueError):
        walls.generate(4)


def test_the_hold_footprint_is_the_meshs():
    """HOLD_* bound hold.stl as the sim places it: socket centre 26 mm out, 14 mm below."""
    raw = (ROOT / "data" / "holds" / "hold.stl").read_bytes()
    n = int.from_bytes(raw[80:84], "little")
    tri = np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    v = np.frombuffer(raw, dtype=tri, count=n, offset=84)["v"].reshape(-1, 3) * 0.001
    x, z = v[:, 0], v[:, 2] - 0.014
    eps = 1e-6  # the STL is float32
    assert walls.HOLD_HALF_WIDTH - 2e-4 < np.abs(x).max() <= walls.HOLD_HALF_WIDTH + eps
    assert -walls.HOLD_BELOW - eps <= z.min() < -walls.HOLD_BELOW + 2e-4
    assert walls.HOLD_ABOVE - 2e-4 < z.max() <= walls.HOLD_ABOVE + eps


def test_the_example_wall_keeps_the_rules():
    assert walls.violations(io.load_wall(ROOT / "data" / "examples" / "wall_hard.json")) == []


def broken(wall, **change):
    """The wall with its first hold changed (Hold fields) and the others kept."""
    first = dataclasses.replace(wall.holds[0], **change)
    return dataclasses.replace(wall, holds=[first, *wall.holds[1:]])


def test_violations_name_what_breaks_a_rule():
    wall = walls.generate(1, 0)
    first = wall.holds[0]
    (i, j) = walls.grid_index(first)

    def problems(w):
        return " | ".join(walls.violations(w))

    assert "not on a grid hole" in problems(
        broken(wall, position=(first.position[0] + 0.007, 0, 0.16))
    )
    assert "off the grid" in problems(broken(wall, position=walls.position(0, j)))
    assert "sticks out" in problems(broken(wall, position=walls.position(25, 1)))
    assert "ArUco" in problems(broken(wall, position=walls.position(3, 90)))
    assert "repeated" in problems(broken(wall, id=wall.holds[1].id))
    assert "start: unknown hold" in problems(broken(wall, id="renamed"))
    beside = Hold("beside", walls.position(i + 2, j))
    assert "share a bolt hole" in problems(dataclasses.replace(wall, holds=[*wall.holds, beside]))
    above = Hold("above", walls.position(i, j + 2))
    assert "overlap" in problems(dataclasses.replace(wall, holds=[*wall.holds, above]))
    under = Hold("under", walls.position(i, j - 3))
    crowded = dataclasses.replace(wall, holds=[*wall.holds, under])
    assert walls.violations(crowded) == [] and walls.crowding(crowded), "legal but crowded"
    swapped = dict(wall.start.contacts, left_hand=wall.start.contacts["left_foot"])
    swapped["left_foot"] = wall.start.contacts["left_hand"]
    assert "hand below a foot" in problems(
        dataclasses.replace(wall, start=dataclasses.replace(wall.start, contacts=swapped))
    )


def test_no_sim_needed():
    code = "import sys, rcr.walls; assert 'mujoco' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize(("level", "seed"), CASES[::10])
def test_draw_shows_every_hold(level, seed):
    wall = walls.generate(level, seed)
    picture = walls.draw(wall).split("\n", 1)[1]  # skip the title line
    assert sum(picture.count(c) for c in "oHF") == len(wall.holds)


if __name__ == "__main__":
    level, *seeds = (int(a) for a in sys.argv[1:] or ["1", "0"])
    for seed in seeds or [0]:
        print(walls.draw(walls.generate(level, seed)))
