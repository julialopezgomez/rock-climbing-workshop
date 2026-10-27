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
EPS = 1e-9


def column(wall, side: str) -> list[Hold]:
    """The route's holds on one side ("L" or "R"), by rung."""
    return sorted((h for h in wall.holds if h.id[0] == side), key=lambda h: int(h.id[1:]))


def extras(wall) -> list[Hold]:
    return [h for h in wall.holds if h.id[0] == "M"]


def steps(holds: list[Hold]) -> list[tuple[float, float]]:
    """(dx, dz) between consecutive holds [m]."""
    return [
        (b.position[0] - a.position[0], b.position[2] - a.position[2]) for a, b in pairwise(holds)
    ]


@pytest.mark.parametrize(("level", "seed"), CASES)
def test_generated_walls_keep_the_rules(level, seed, tmp_path):
    wall = walls.generate(level, seed)
    assert walls.violations(wall) == []
    assert walls.crowding(wall) == []
    assert {h.id[0] for h in wall.holds} <= {"L", "R", "M"}
    io.save_wall(wall, tmp_path / "wall.json")
    assert io.load_wall(tmp_path / "wall.json") == wall


@pytest.mark.parametrize(("level", "seed"), CASES)
def test_the_start_is_the_example_walls(level, seed):
    wall = walls.generate(level, seed)
    holds = {h.id: h.position for h in wall.holds}
    c = wall.start.contacts
    assert (c["left_foot"], c["right_foot"]) == ("L0", "R0")
    assert c["left_hand"][1:] == c["right_hand"][1:], "hands on one rung"
    for foot, hand in (("left_foot", "left_hand"), ("right_foot", "right_hand")):
        assert 0.15 <= holds[c[hand]][2] - holds[c[foot]][2] <= 0.25
    assert holds["L0"][0] < holds["R0"][0]
    top = max(h.position[2] for h in column(wall, "L") + column(wall, "R"))
    assert 1.27 - 0.075 < top <= 1.27 + EPS


def single_columns(holds: list[Hold]) -> dict[float, list[Hold]]:
    """Holds grouped by x, each group bottom to top."""
    out: dict[float, list[Hold]] = {}
    for h in sorted(holds, key=lambda h: h.position[2]):
        out.setdefault(h.position[0], []).append(h)
    return out


@pytest.mark.parametrize("seed", SEEDS)
def test_level1_is_regular_with_holds_to_choose_from(seed):
    wall = walls.generate(1, seed)
    route = [column(wall, "L"), column(wall, "R")]
    (rise,) = {round(dz, 6) for side in route for _, dz in steps(side)}
    assert rise in (0.06, 0.075)
    for side in route:
        assert len({h.position[0] for h in side}) == 1, "one column"
    more = single_columns(extras(wall))
    assert len(more) <= 2
    for col in more.values():
        assert {round(dz, 6) for _, dz in steps(col)} <= {rise}, "regular extra column"


def missing(side: list[Hold]) -> list[float]:
    """Heights of the holds missing from a side of an L2 route (60 mm rungs, one column)."""
    return [a.position[2] + 0.06 for a, b in pairwise(side) if int(b.id[1:]) - int(a.id[1:]) == 2]


@pytest.mark.parametrize("seed", SEEDS)
def test_level2_has_gaps_that_stay_gaps(seed):
    wall = walls.generate(2, seed)
    route = [column(wall, "L"), column(wall, "R")]
    for side in route:
        assert len({h.position[0] for h in side}) == 1, "one column"
        assert {round(dz, 6) for _, dz in steps(side)} <= {0.06, 0.12}, "never two in a row"
    rungs = [{int(h.id[1:]) for h in side} for side in route]
    whole = set(range(max(rungs[0]))) - rungs[0] - rungs[1]
    assert 1 <= len(whole) <= 2, "one or two whole rungs missing"
    for z in missing(route[0]) + missing(route[1]):
        near = [h for h in wall.holds if abs(h.position[2] - z) < 0.06 - EPS]
        assert all(h.id[0] in "LR" for h in near), f"an extra hold fills the gap at z = {z}"
    assert len(single_columns(extras(wall))) <= 2


def traverse(side: list[Hold]) -> tuple[float, float]:
    """The longest run of consecutive steps each >= 45 mm sideways in one direction: (sideways,
    up) in all [m]."""
    best, run = (0.0, 0.0), (0.0, 0.0, 0)
    for dx, dz in steps(side):
        direction = (dx > 0) - (dx < 0)
        if abs(dx) >= 3 * GRID_PITCH - EPS:
            run = (
                (run[0] + abs(dx), run[1] + dz, direction)
                if direction == run[2]
                else (abs(dx), dz, direction)
            )
        else:
            run = (0.0, 0.0, 0)
        best = max(best, run[:2])
    return best


@pytest.mark.parametrize("seed", SEEDS)
def test_level3_traverses(seed):
    wall = walls.generate(3, seed)
    for side in (column(wall, "L"), column(wall, "R")):
        for dx, dz in steps(side):
            assert abs(dx) < EPS or 3 * GRID_PITCH - EPS <= abs(dx) <= 5 * GRID_PITCH + EPS
            assert dz >= -EPS
        sideways, up = traverse(side)
        assert sideways >= 0.15 - EPS, "a traverse of at least 0.15 m"
        assert up <= sideways + EPS, "mostly sideways"
    route_rows = {h.position[2] for h in wall.holds if h.id[0] in "LR"}
    per_row: dict[float, int] = {}
    for h in extras(wall):
        assert h.position[2] in route_rows, "extras only on the route's rows"
        per_row[h.position[2]] = per_row.get(h.position[2], 0) + 1
    assert max(per_row.values(), default=0) <= 2


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


def broken(wall, hold_id: str, **change):
    """The wall with one hold changed (Hold fields) and the others kept."""
    return dataclasses.replace(
        wall,
        holds=[dataclasses.replace(h, **change) if h.id == hold_id else h for h in wall.holds],
    )


def with_hold(wall, hold: Hold):
    return dataclasses.replace(wall, holds=[*wall.holds, hold])


def test_violations_name_what_breaks_a_rule():
    wall = walls.generate(1, 0)
    foot = next(h for h in wall.holds if h.id == "L0")
    (i, j) = walls.grid_index(foot)

    def problems(w):
        return " | ".join(walls.violations(w))

    off = (foot.position[0] + 0.007, 0.0, foot.position[2])
    assert "not on a grid hole" in problems(broken(wall, "L0", position=off))
    assert "off the grid" in problems(broken(wall, "L0", position=walls.position(0, j)))
    assert "sticks out" in problems(broken(wall, "L0", position=walls.position(25, 1)))
    assert "ArUco" in problems(broken(wall, "L0", position=walls.position(3, 90)))
    assert "repeated" in problems(broken(wall, "L0", id="R0"))
    assert "start: unknown hold" in problems(broken(wall, "L0", id="renamed"))
    assert "share a bolt hole" in problems(
        with_hold(wall, Hold("beside", walls.position(i + 2, j)))
    )
    assert "overlap" in problems(with_hold(wall, Hold("above", walls.position(i, j + 2))))
    crowded = with_hold(wall, Hold("under", walls.position(i, j - 3)))
    assert walls.violations(crowded) == [] and walls.crowding(crowded), "legal but crowded"
    swapped = dict(wall.start.contacts, left_hand="L0", left_foot=wall.start.contacts["left_hand"])
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
