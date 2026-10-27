"""Seeded random walls for the three levels [Rules §3], and the checks that a wall keeps the rules.

    from rcr import walls
    wall = walls.generate(level=2, seed=3)
    assert not walls.violations(wall) and not walls.crowding(wall)
    print(walls.draw(wall))

The rulebook says only: L1 "regularly spaced holds", L2 "a ladder with missing holds", L3 "the robot
needs to move to the right or to the left" [Rules §3]. Our reading [Ours]:
  L1  a ladder, the route: two columns, a rung every 60 or 75 mm, the right column optionally
      staggered by half a rise; plus up to two more columns with the same rows, so the robot has
      holds to choose from and the wall stays regular. Rise, spacing and columns vary with the seed.
  L2  the same with a 60 mm rise and holds missing: one or two whole rungs (both sides at the same
      height) and single holds, never two in a row on a side, so every gap is at most 120 mm. The
      extra columns have no hold within a rise of a missing one, so the gaps stay gaps.
  L3  vertical ladder segments joined by one or two traverses: rungs moving 45-75 mm sideways and
      at most 30 mm up each, 0.15-0.25 m sideways in all (the second one back the other way); plus
      0-2 holds scattered on each row around the route, as on the organisers' own wall.
Holds are named as on the example wall: the route's `L<k>` and `R<k>` (k the rung, so a missing hold
leaves a gap in k), the other holds `M<n>` from the bottom up. The start copies the example wall the
optimiser has solved: feet on `L0` and `R0`, hands on the rung closest to 0.225 m above them
[Ours, data/examples/wall_hard.json].

Every hold is placed only where it keeps the rules and the crowding rule (`_fits`), and the wall is
checked again at the end. (level, seed) reproduces a wall exactly; `scripts/make_wall.py --save`
writes one to a file. Wall ids are `gen_l<level>_s<seed>`, apart from those of `rcr.sim.levels` (an
earlier, separate generator).

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
SPACING = (10, 12)  # between the two route columns: 0.15 .. 0.18 m; the example wall's is 0.165 m
LEFT_COL = (18, 22)  # the left route column at the start: x = 0.29 .. 0.35 m
COLS = (10, 41)  # every column stays here, so the top and bottom rows clear the ArUco plates
EXTRA_COLS = (0, 2)  # L1, L2: columns besides the route's
EXTRA_APART = 4  # an extra column is at least 60 mm from every other column
# L2: a gap is at most 2 rises, 120 mm. The optimiser found a hand could not reach up 150 mm rows
# at 70 % torque; raise L2_RISE to 5 only once feasibility() passes 150 mm gaps.
L2_RISE = 4
L2_RUNGS = (1, 2)  # whole rungs missing
L2_DROP = 0.3  # chance to drop a single hold (if its neighbours on its side are kept)
TRAVERSES = (1, 2)  # L3
TRAVERSE = (10, 17)  # L3: sideways in one traverse, 0.15 .. 0.255 m
TRAVERSE_DX = (3, 5)  # L3: sideways per rung in a traverse, 45 .. 75 mm (3 clears the hold beside)
TRAVERSE_DZ = (0, 2)  # L3: up per rung in a traverse, 0 .. 30 mm, if that fits; else up to a rise
LADDER_BETWEEN = (2, 4)  # L3: ladder rungs between traverses (and at least 2 before the top)
SCATTER = (0, 2)  # L3: extra holds per row
SCATTER_BAND = 6  # L3: up to 90 mm outside the route on their row
ATTEMPTS = 100  # L3: routes drawn before giving up (one fails if it runs out of height)
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


def _misplaced(h: Hold) -> str:
    """Why the hold cannot go where it is on its own, or "": on a grid hole [Rules §3] with both
    bolts on the grid [Rules §4], on the board, off the ArUco plates [Rules §5.2]."""
    if h.position[1] != 0.0:
        return f"{h.id}: y = {h.position[1]}, not on the wall face (y = 0)"
    cell = grid_index(h)
    if cell is None:
        return f"{h.id}: {h.position} is not on a grid hole"
    i, j = cell
    if not (1 <= i <= N_COLS - 2 and 0 <= j < N_ROWS):
        return f"{h.id}: bolts at holes ({i - 1}, {j}), ({i + 1}, {j}), off the grid"
    box = _footprint(h)
    if box[0] < 0 or box[1] > WALL_WIDTH or box[2] < 0 or box[3] > WALL_HEIGHT:
        return f"{h.id}: sticks out of the board"
    if any(_overlap(box, plate) for plate in PLATES):
        return f"{h.id}: on an ArUco plate"
    return ""


def _clash(a: Hold, b: Hold) -> str:
    """Why two holds cannot both be on the wall, or "": a shared bolt hole, or overlapping
    (the provided hold's footprint) [Rules §4]."""
    ca, cb = grid_index(a), grid_index(b)
    if ca and cb and ca[1] == cb[1] and abs(ca[0] - cb[0]) in (0, 2):
        return f"{a.id} and {b.id}: share a bolt hole"
    if _overlap(_footprint(a), _footprint(b)):
        return f"{a.id} and {b.id}: overlap"
    return ""


def _crowded(a: Hold, b: Hold) -> str:
    """Why a ball cannot leave the lower hold's socket upwards, or "": the other hold is within a
    hold's width sideways and less than CLEAR_ABOVE above it [Ours]."""
    (xa, _, za), (xb, _, zb) = a.position, b.position
    if abs(xa - xb) < 2 * HOLD_HALF_WIDTH and 1e-9 < abs(za - zb) < CLEAR_ABOVE - 1e-9:
        low, high = (a, b) if za < zb else (b, a)
        return f"{high.id} is {abs(za - zb) * 1e3:.0f} mm above {low.id}"
    return ""


def _fits(h: Hold, placed: list[Hold]) -> bool:
    """Can h join the holds placed so far, keeping the rules and the crowding rule?"""
    return not _misplaced(h) and not any(_clash(h, p) or _crowded(h, p) for p in placed)


def violations(wall: WallModel) -> list[str]:
    """Why the wall could not be built as the rules say, one line per problem; [] if it can.

    Every hold placed as `_misplaced` says, clear of every other hold (`_clash`); unique ids; a
    start stance with every limb on its own hold, hands not below feet [Ours, as the optimiser].
    """
    out = []
    if wall.level not in (1, 2, 3):
        out.append(f"level {wall.level} is not 1, 2 or 3")
    ids = [h.id for h in wall.holds]
    repeated = sorted({i for i in ids if ids.count(i) > 1})
    if repeated:
        out.append(f"repeated hold ids: {repeated}")
    out += [why for h in wall.holds if (why := _misplaced(h))]
    out += [why for a, b in itertools.combinations(wall.holds, 2) if (why := _clash(a, b))]
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
    socket upwards (`_crowded`) [Ours]. Legal, but our generator never does it."""
    return [why for a, b in itertools.combinations(wall.holds, 2) if (why := _crowded(a, b))]


# --- Generator ----------------------------------------------------------------------------------


def generate(level: int, seed: int = 0) -> WallModel:
    """A random wall of this level; the same (level, seed) gives the same wall."""
    if level not in (1, 2, 3):
        raise ValueError(f"level must be 1, 2 or 3; got {level}")
    rng = random.Random(f"{level}/{seed}")
    for _ in range(ATTEMPTS):
        if route := _route(level, rng):
            break
    else:
        raise AssertionError(f"level {level}, seed {seed}: no route in {ATTEMPTS} attempts")
    cells, hands, rise = route
    dropped = _drops(rng, len(cells), hands) if level == 2 else set()
    ladder = [
        Hold(id=f"{'LR'[s]}{k}", position=position(*cells[k][s]), kind=KIND)
        for k in range(len(cells))
        for s in (0, 1)
        if (s, k) not in dropped
    ]
    if level == 3:
        extras = _scatter(rng, ladder)
    else:
        missing = [position(*cells[k][s])[2] for s, k in dropped]
        extras = _extra_columns(rng, cells, ladder, missing, rise)
    extras.sort(key=lambda h: (h.position[2], h.position[0]))
    extras = [Hold(id=f"M{n}", position=h.position, kind=KIND) for n, h in enumerate(extras)]
    start = Stance(
        contacts={
            "right_hand": f"R{hands}",
            "left_hand": f"L{hands}",
            "right_foot": "R0",
            "left_foot": "L0",
        }
    )
    holds = sorted(ladder + extras, key=lambda h: (h.position[2], h.position[0]))
    wall = WallModel(id=f"gen_l{level}_s{seed}", level=level, holds=holds, start=start)
    problems = violations(wall) + crowding(wall)
    if problems:  # the construction rules these out; a failure here is a bug in this file
        raise AssertionError(f"{wall.id}: {problems}")
    return wall


def _route(level: int, rng: random.Random) -> tuple[list, int, int] | None:
    """The route's grid cells, rung by rung ([[(i, j) left, (i, j) right], ...]), the start
    hands' rung and the rise; or None if an L3 traverse ran out of height or room."""
    rise = L2_RISE if level == 2 else rng.choice(RISES)
    left = rng.randint(*LEFT_COL)
    right = left + rng.randint(*SPACING)
    # L3: no stagger, as a staggered column would sit just under the other one's holds once a
    # traverse has moved it by the spacing [Ours]
    stagger = 0 if level == 3 else rng.choice((0, rise // 2))
    feet = rng.randint(*FEET_ROWS)
    hands = min(range(1, 6), key=lambda k: abs(k * rise - HANDS_ABOVE))
    cells, placed = [], []

    def step(di: int, dj: int) -> bool:
        """Add a rung (di, dj) pitches from the last one, if it fits; False otherwise."""
        rung = (
            [(i + di, j + dj) for i, j in cells[-1]]
            if cells
            else [(left, feet), (right, feet + stagger)]
        )
        new = [
            Hold(f"{s}{len(cells)}", position(i, j), KIND)
            for s, (i, j) in zip("LR", rung, strict=True)
        ]
        inside = all(COLS[0] <= i <= COLS[1] and j <= TOP_ROW for i, j in rung)
        if not (inside and all(_fits(h, placed) for h in new)):
            return False
        cells.append(rung)
        placed.extend(new)
        return True

    step(0, 0)
    if level != 3:
        while step(0, rise):
            pass
        return cells, hands, rise
    for _ in range(hands + rng.randint(0, 2)):
        if not step(0, rise):
            return None
    direction = rng.choice((1, -1))
    for t in range(rng.randint(*TRAVERSES)):
        if t and not all(step(0, rise) for _ in range(rng.randint(*LADDER_BETWEEN))):
            return None
        (i_left, _), (i_right, _) = cells[-1]
        room = {1: COLS[1] - i_right, -1: i_left - COLS[0]}
        if room[direction] < TRAVERSE[0]:
            direction = -direction
        if not _traverse(rng, step, direction, min(rng.randint(*TRAVERSE), room[direction]), rise):
            return None
        direction = -direction
    top = 0
    while step(0, rise):
        top += 1
    return (cells, hands, rise) if top >= LADDER_BETWEEN[0] else None


def _traverse(rng: random.Random, step, direction: int, size: int, rise: int) -> bool:
    """Move the route `size` pitches sideways, TRAVERSE_DX per rung, rising TRAVERSE_DZ per rung
    where that fits (more, up to a rise, where not). False if a rung fits nowhere, or if the
    traverse rose more than it moved sideways."""
    moved = risen = 0
    while moved < size:
        left = size - moved
        # each width leaves either nothing or at least a whole step
        widths = [
            d
            for d in range(TRAVERSE_DX[0], min(TRAVERSE_DX[1], left) + 1)
            if left - d == 0 or left - d >= TRAVERSE_DX[0]
        ]
        low = list(range(TRAVERSE_DZ[0], TRAVERSE_DZ[1] + 1))
        rng.shuffle(widths)
        rng.shuffle(low)
        tries = [(d, dz) for dz in low + list(range(TRAVERSE_DZ[1] + 1, rise + 1)) for d in widths]
        for d, dz in tries:
            if step(direction * d, dz):
                moved, risen = moved + d, risen + dz
                break
        else:
            return False
    return risen <= moved


def _drops(rng: random.Random, n: int, hands: int) -> set[tuple[int, int]]:
    """L2: the (side, rung) dropped from n rungs: L2_RUNGS whole rungs, then single holds (one
    side of a rung only, so the whole rungs stay L2_RUNGS). Never the start rungs or below, nor
    the top rung, nor two in a row on one side."""
    droppable = range(hands + 1, n - 1)
    whole: list[int] = []
    for _ in range(rng.randint(*L2_RUNGS)):
        options = [k for k in droppable if all(abs(k - w) >= 2 for w in whole)]
        if options:
            whole.append(rng.choice(options))
    dropped = {(s, k) for k in whole for s in (0, 1)}
    for k in droppable:
        for s in (0, 1):
            free = not ({(s, k - 1), (s, k), (s, k + 1), (1 - s, k)} & dropped)
            if free and rng.random() < L2_DROP:
                dropped.add((s, k))
    return dropped


def _extra_columns(
    rng: random.Random, cells: list, ladder: list[Hold], missing: list[float], rise: int
) -> list[Hold]:
    """L1, L2: EXTRA_COLS more columns, each with the rows of the left or the right route column
    and EXTRA_APART from every other column; none within a rise of a missing hold's height."""
    taken = [cells[0][0][0], cells[0][1][0]]
    placed, out = list(ladder), []
    columns = list(range(COLS[0], COLS[1] + 1))
    rng.shuffle(columns)
    want = rng.randint(*EXTRA_COLS)
    for i in columns:
        if len(taken) - 2 == want:
            break
        if any(abs(i - c) < EXTRA_APART for c in taken):
            continue
        side = rng.randrange(2)
        for rung in cells:
            h = Hold("M", position(i, rung[side][1]), KIND)
            near_gap = any(abs(h.position[2] - z) < rise * GRID_PITCH - 1e-9 for z in missing)
            if not near_gap and _fits(h, placed):
                placed.append(h)
                out.append(h)
        taken.append(i)
    return out


def _scatter(rng: random.Random, ladder: list[Hold]) -> list[Hold]:
    """L3: on each row of the route, SCATTER holds at random columns up to SCATTER_BAND outside
    the route's holds on that row, where they fit."""
    rows: dict[int, list[int]] = {}
    for h in ladder:
        i, j = grid_index(h)
        rows.setdefault(j, []).append(i)
    placed, out = list(ladder), []
    for j in sorted(rows):
        lo = max(COLS[0], min(rows[j]) - SCATTER_BAND)
        hi = min(COLS[1], max(rows[j]) + SCATTER_BAND)
        columns = list(range(lo, hi + 1))
        rng.shuffle(columns)
        want, got = rng.randint(*SCATTER), 0
        for i in columns:
            if got == want:
                break
            h = Hold("M", position(i, j), KIND)
            if _fits(h, placed):
                placed.append(h)
                out.append(h)
                got += 1
    return out


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
