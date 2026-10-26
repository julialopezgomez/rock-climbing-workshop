"""Seeded random walls for the three levels [Rules §3], and the checks that a wall keeps the rules.

    from rcr import walls
    wall = walls.generate(level=2, seed=3)
    assert not walls.violations(wall) and not walls.crowding(wall)
    print(walls.draw(wall))

The rulebook says only: L1 "regularly spaced holds", L2 "a ladder with missing holds", L3 "the robot
needs to move to the right or to the left" [Rules §3]. Our reading [Ours]:
  L1  a ladder: two columns, one hold every `rise` pitches on each, the right column optionally
      staggered by half a rise. Rise, column spacing and position vary with the seed.
  L2  an L1 ladder with rise 4 (60 mm), with holds missing: never two in a row on one side, so no
      side has a gap over 120 mm.
  L3  an L1 ladder whose columns drift sideways together, at most 2 pitches per rung, by 8 to 14
      pitches (0.12-0.21 m) in all, in one or two legs.
The start copies the example wall the optimiser has solved: feet on the lowest hold of each side,
hands on the hold closest to 0.225 m above them [Ours, data/examples/wall_hard.json].

(level, seed) reproduces a wall exactly; `scripts/make_wall.py --save` writes one to a file. Wall
ids are `gen_l<level>_s<seed>`, apart from those of `rcr.sim.levels` (an earlier, separate one).

Pi-safe: stdlib only. Independent of `rcr.sim` (which needs mujoco).
"""

import itertools
import random

from rcr.types import (
    GRID_MARGIN_X,
    GRID_MARGIN_Z,
    GRID_PITCH,
    LIMBS,
    WALL_HEIGHT,
    WALL_WIDTH,
    Hold,
    Stance,
    WallModel,
)

N_COLS = round((WALL_WIDTH - 2 * GRID_MARGIN_X) / GRID_PITCH) + 1  # 52 [Rules v1.1 Figure 2]
N_ROWS = round((WALL_HEIGHT - 2 * GRID_MARGIN_Z) / GRID_PITCH) + 1  # 93
ARUCO_PLATE = 0.136  # [m] square plate on each corner of the board [Rules §5.2]

# The provided hold on the wall face, around Hold.position (the bolt midpoint) [m], measured on
# data/holds/hold.stl [Rules v1.1 Figure 3]: 40.3 mm wide, from 29 mm below to 4 mm above. Two of
# them collide unless 3 pitches apart sideways or vertically. tests/test_walls.py checks the mesh.
HOLD_HALF_WIDTH = 0.0202
HOLD_ABOVE = 0.004
HOLD_BELOW = 0.029
# [m] a hold within a hold's width above it must be this much higher [Ours]: the ball leaves its
# socket (centre 14 mm below the bolt line) upwards by 35 mm (optimiser transition.LIFT), and its
# top (10 mm radius) must stay under the next hold (which reaches 29 mm below its bolt line):
# -14 + 35 + 10 + 29 = 60 mm, i.e. 4 pitches.
CLEAR_ABOVE = 0.060

# The generator, in grid pitches (0.015 m) [Ours].
FEET_ROWS = (8, 10)  # z = 0.13 .. 0.16 m, the start feet; the example wall's are at 0.16 m
HANDS_ABOVE = 15  # 0.225 m, start hands above the feet, as on the example wall
TOP_ROW = 84  # z = 1.27 m, no hold above it, as on the example wall
RISES = (4, 5)  # 60 or 75 mm between rungs on a side; the optimiser solved 75 mm
L2_RISE = 4  # L2 drops holds, so its rungs are close: a gap is at most 2 rises, 120 mm
L2_DROP = 0.4  # chance to drop a hold (if the one below on its side was kept)
SPACING = (10, 12)  # between the two columns: 0.15 .. 0.18 m; the example wall's is 0.165 m
LEFT_COL = (18, 22)  # the left column at the start: x = 0.29 .. 0.35 m
DRIFT = (8, 14)  # L3: net sideways move of a leg, 0.12 .. 0.21 m
DRIFT_STEP = 2  # L3: at most 30 mm sideways per rung
COLS = (10, 41)  # L3: columns stay here, so the top rungs clear the top ArUco plates
KIND = "unknown"  # CHANGE: hold kinds are not published [Rules §4]


# --- Grid ---------------------------------------------------------------------------------------


def position(i: int, j: int) -> tuple[float, float, float]:
    """Hold.position of the hold bolted through holes (i - 1, j) and (i + 1, j) [Rules §4]."""
    return (round(GRID_MARGIN_X + GRID_PITCH * i, 6), 0.0, round(GRID_MARGIN_Z + GRID_PITCH * j, 6))


def grid_index(hold: Hold) -> tuple[int, int] | None:
    """(i, j) of the grid hole at the hold's reference point, or None if it is not on one."""
    x, _, z = hold.position
    i, j = (x - GRID_MARGIN_X) / GRID_PITCH, (z - GRID_MARGIN_Z) / GRID_PITCH
    if abs(i - round(i)) > 1e-6 or abs(j - round(j)) > 1e-6:
        return None
    return round(i), round(j)


def hold_id(i: int, j: int) -> str:
    return f"h{j:02d}_{i:02d}"


# --- Checks -------------------------------------------------------------------------------------


def _footprint(hold: Hold) -> tuple[float, float, float, float]:
    x, _, z = hold.position
    return x - HOLD_HALF_WIDTH, x + HOLD_HALF_WIDTH, z - HOLD_BELOW, z + HOLD_ABOVE


def _overlap(a, b) -> bool:
    """Do two (x0, x1, z0, z1) boxes overlap?"""
    return a[0] < b[1] and b[0] < a[1] and a[2] < b[3] and b[2] < a[3]


PLATES = [
    (x, x + ARUCO_PLATE, z, z + ARUCO_PLATE)
    for x in (0.0, WALL_WIDTH - ARUCO_PLATE)
    for z in (0.0, WALL_HEIGHT - ARUCO_PLATE)
]


def violations(wall: WallModel) -> list[str]:
    """Why the wall could not be built as the rules say, one line per problem; [] if it can.

    Every hold on the grid [Rules §3] with both bolts in it [Rules §4], on the board, off the ArUco
    plates [Rules §5.2], clear of every other hold (the provided hold's footprint); unique ids;
    a start stance with every limb on its own hold, hands not below feet [Ours, as the optimiser].
    """
    out = []
    if wall.level not in (1, 2, 3):
        out.append(f"level {wall.level} is not 1, 2 or 3")
    ids = [h.id for h in wall.holds]
    repeated = sorted({i for i in ids if ids.count(i) > 1})
    if repeated:
        out.append(f"repeated hold ids: {repeated}")
    placed = []
    for h in wall.holds:
        cell = grid_index(h)
        if h.position[1] != 0.0:
            out.append(f"{h.id}: y = {h.position[1]}, not on the wall face (y = 0)")
        if cell is None:
            out.append(f"{h.id}: {h.position} is not on a grid hole")
            continue
        i, j = cell
        if not (1 <= i <= N_COLS - 2 and 0 <= j < N_ROWS):
            out.append(f"{h.id}: bolts at holes ({i - 1}, {j}), ({i + 1}, {j}), off the grid")
            continue
        box = _footprint(h)
        if box[0] < 0 or box[1] > WALL_WIDTH or box[2] < 0 or box[3] > WALL_HEIGHT:
            out.append(f"{h.id}: sticks out of the board")
        if any(_overlap(box, plate) for plate in PLATES):
            out.append(f"{h.id}: on an ArUco plate")
        placed.append((h, cell, box))
    for (a, (ia, ja), box_a), (b, (ib, jb), box_b) in itertools.combinations(placed, 2):
        if ja == jb and abs(ia - ib) in (0, 2):
            out.append(f"{a.id} and {b.id}: share a bolt hole")
        elif _overlap(box_a, box_b):
            out.append(f"{a.id} and {b.id}: overlap")
    out += _start_violations(wall)
    return out


def _start_violations(wall: WallModel) -> list[str]:
    holds = {h.id: h for h in wall.holds}
    contacts = wall.start.contacts
    if set(contacts) != set(LIMBS):
        return [f"start: limbs {sorted(contacts)}, not {sorted(LIMBS)}"]
    used = list(contacts.values())
    if any(h not in holds for h in used):
        return [f"start: unknown hold in {used}"]
    out = []
    if len(set(used)) < len(used):
        out.append("start: two limbs on one hold")
    hand = min(holds[contacts[limb]].position[2] for limb in ("right_hand", "left_hand"))
    foot = max(holds[contacts[limb]].position[2] for limb in ("right_foot", "left_foot"))
    if hand < foot:
        out.append("start: a hand below a foot")
    return out


def crowding(wall: WallModel) -> list[str]:
    """Pairs of holds where one sits so close above the other that a ball cannot leave the lower
    socket upwards (CLEAR_ABOVE) [Ours]. Legal, but our generator never does it."""
    out = []
    for a, b in itertools.combinations(wall.holds, 2):
        (xa, _, za), (xb, _, zb) = a.position, b.position
        if abs(xa - xb) < 2 * HOLD_HALF_WIDTH and 1e-9 < abs(za - zb) < CLEAR_ABOVE - 1e-9:
            low, high = (a, b) if za < zb else (b, a)
            out.append(f"{high.id} is {abs(za - zb) * 1e3:.0f} mm above {low.id}")
    return out


# --- Generator ----------------------------------------------------------------------------------


def generate(level: int, seed: int = 0) -> WallModel:
    """A random wall of this level; the same (level, seed) gives the same wall."""
    if level not in (1, 2, 3):
        raise ValueError(f"level must be 1, 2 or 3; got {level}")
    rng = random.Random(f"{level}/{seed}")
    rise = L2_RISE if level == 2 else rng.choice(RISES)
    left = rng.randint(*LEFT_COL)
    right = left + rng.randint(*SPACING)
    stagger = rng.choice((0, rise // 2))
    feet = rng.randint(*FEET_ROWS)
    sides = [
        list(range(feet, TOP_ROW + 1, rise)),  # left column rows
        list(range(feet + stagger, TOP_ROW + 1, rise)),  # right column rows
    ]
    hands = [_hands_rung(rows) for rows in sides]
    drift = None
    if level == 3:  # over the rungs both columns have; a staggered one may have one more on top
        drift = _drift(rng, min(map(len, sides)), hands, left, right)
        drift += [drift[-1]] * (max(map(len, sides)) - len(drift))
    holds = {}
    for side, (rows, col) in enumerate(zip(sides, (left, right), strict=True)):
        kept = _drop(rng, rows, hands[side]) if level == 2 else range(len(rows))
        for k in kept:
            i = col + (drift[k] if drift else 0)
            holds[(i, rows[k])] = Hold(
                id=hold_id(i, rows[k]), position=position(i, rows[k]), kind=KIND
            )

    def at(side: int, k: int) -> str:
        i = (left, right)[side] + (drift[k] if drift else 0)
        return hold_id(i, sides[side][k])

    start = Stance(
        contacts={
            "right_hand": at(1, hands[1]),
            "left_hand": at(0, hands[0]),
            "right_foot": at(1, 0),
            "left_foot": at(0, 0),
        }
    )
    ordered = sorted(holds.values(), key=lambda h: (h.position[2], h.position[0]))
    wall = WallModel(id=f"gen_l{level}_s{seed}", level=level, holds=ordered, start=start)
    problems = violations(wall) + crowding(wall)
    if problems:  # the construction rules these out; a failure here is a bug in this file
        raise AssertionError(f"{wall.id}: {problems}")
    return wall


def _hands_rung(rows: list[int]) -> int:
    """Index of the rung closest to HANDS_ABOVE above the side's first one (the lower if tied)."""
    return min(range(1, len(rows)), key=lambda k: abs(rows[k] - rows[0] - HANDS_ABOVE))


def _drop(rng: random.Random, rows: list[int], hands: int) -> list[int]:
    """L2: indices of the rungs kept on one side. The start rungs and below, and the top rung,
    stay; no two dropped in a row; at least one dropped."""
    kept, last_dropped = [], False
    droppable = range(hands + 1, len(rows) - 1)
    for k in range(len(rows)):
        if k in droppable and not last_dropped and rng.random() < L2_DROP:
            last_dropped = True
            continue
        kept.append(k)
        last_dropped = False
    if len(kept) == len(rows) and droppable:
        kept.remove(rng.choice(droppable))
    return kept


def _drift(rng: random.Random, n: int, hands: list[int], left: int, right: int) -> list[int]:
    """L3: the columns' sideways offset at each of n rungs, in pitches. None up to the start
    hands; then one or two legs (the second back the other way) of DRIFT pitches, at most
    DRIFT_STEP per rung, the columns staying within COLS."""
    first = max(hands) + 1
    legs = 2 if rng.random() < 0.5 and n - first >= 2 * DRIFT[1] // DRIFT_STEP else 1
    bounds = [first + (n - first) * leg // legs for leg in range(legs + 1)]
    direction, offset, steps = rng.choice((1, -1)), 0, [0] * n
    for leg in range(legs):
        room = {1: COLS[1] - right - offset, -1: left + offset - COLS[0]}
        if leg == 0 and room[direction] < DRIFT[0]:
            direction = -direction
        rungs = list(range(bounds[leg], bounds[leg + 1]))
        size = min(rng.randint(*DRIFT), room[direction], DRIFT_STEP * len(rungs))
        for _ in range(size):  # one pitch at a time, on a random rung with room for it
            k = rng.choice([k for k in rungs if abs(steps[k]) < DRIFT_STEP])
            steps[k] += direction
        offset += direction * size
        direction = -direction
    return list(itertools.accumulate(steps))


# --- Drawing ------------------------------------------------------------------------------------


def draw(wall: WallModel, dx: float = GRID_PITCH, dz: float = 2 * GRID_PITCH) -> str:
    """The wall as text, seen from the front: H/F = start hand/foot hold, o = other hold.

    One character is dx wide and dz tall; dz = 2 dx keeps the proportions of a terminal font.
    """
    cols, rows = round(WALL_WIDTH / dx), round(WALL_HEIGHT / dz)
    grid = [[" "] * cols for _ in range(rows)]
    start = wall.start.contacts
    for h in wall.holds:
        x, _, z = h.position
        limbs = [limb for limb, held in start.items() if held == h.id]
        mark = "H" if any("hand" in limb for limb in limbs) else "F" if limbs else "o"
        grid[rows - 1 - int(z / dz)][int(x / dx)] = mark
    border = "+" + "-" * cols + "+"
    lines = [f"{wall.id}: level {wall.level}, {len(wall.holds)} holds", border]
    lines += ["|" + "".join(row) + "|" for row in grid]
    return "\n".join([*lines, border])
