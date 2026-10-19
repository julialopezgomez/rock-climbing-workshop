"""Wall models for the three levels [Rules §3], as `wall/1` WallModels.

L1 ladder, L2 ladder with gaps, L3 lateral drift. Seeded, so (level, seed)
reproduces a wall exactly.
"""

from __future__ import annotations

import random

from rcr.sim.geometry import hold_position
from rcr.types import Hold, Stance, WallModel

ROW_STEP = 6  # 0.090 m between rungs [Ours]
COL_LEFT = 21  # the two ladder columns, 0.150 m apart [Ours]
COL_RIGHT = 31
FIRST_ROW = 6
LAST_ROW = 90
KIND = "cup"  # CHANGE: the provided holds are not published [Rules §4]


def _rung_rows() -> list[int]:
    return list(range(FIRST_ROW, LAST_ROW + 1, ROW_STEP))


def _start_stance(holds: list[Hold]) -> Stance:
    """Hands on the third rung, feet on the first [Ours]."""
    ids = {h.id for h in holds}
    for hand_row in (3, 2, 1):
        contacts = {
            "right_hand": f"R{hand_row}",
            "left_hand": f"L{hand_row}",
            "right_foot": "R0",
            "left_foot": "L0",
        }
        if set(contacts.values()) <= ids:
            return Stance(contacts=contacts)
    raise ValueError("no usable start stance on this wall")


def _ladder() -> list[Hold]:
    holds = []
    for k, row in enumerate(_rung_rows()):
        holds.append(Hold(id=f"L{k}", position=hold_position(COL_LEFT, row), kind=KIND))
        holds.append(Hold(id=f"R{k}", position=hold_position(COL_RIGHT, row), kind=KIND))
    return holds


def level1() -> WallModel:
    holds = _ladder()
    return WallModel(id="l1", level=1, holds=holds, start=_start_stance(holds))


def level2(seed: int = 0, drop_fraction: float = 0.25) -> WallModel:
    rng = random.Random(seed)
    holds = _ladder()
    rows = _rung_rows()
    protected = {f"{side}{k}" for k in (0, 1, 2, 3, len(rows) - 1) for side in "LR"}
    candidates = [h.id for h in holds if h.id not in protected]
    dropped = set(rng.sample(candidates, int(len(candidates) * drop_fraction)))
    kept = [h for h in holds if h.id not in dropped]
    return WallModel(id=f"l2_s{seed}", level=2, holds=kept, start=_start_stance(kept))


def level3(seed: int = 0, shift: int = 5) -> WallModel:
    """A ladder that drifts sideways, so the robot has to traverse [Rules §3]."""
    rng = random.Random(seed)
    holds, offset, direction = [], 0, 1
    for k, row in enumerate(_rung_rows()):
        if k > 3 and rng.random() < 0.7:
            offset += direction * shift
            if not -10 <= offset <= 10:
                direction *= -1
                offset += 2 * direction * shift
        holds.append(Hold(id=f"L{k}", position=hold_position(COL_LEFT + offset, row), kind=KIND))
        holds.append(Hold(id=f"R{k}", position=hold_position(COL_RIGHT + offset, row), kind=KIND))
    return WallModel(id=f"l3_s{seed}", level=3, holds=holds, start=_start_stance(holds))


def generate(level: int, seed: int = 0) -> WallModel:
    if level == 1:
        return level1()
    if level == 2:
        return level2(seed)
    if level == 3:
        return level3(seed)
    raise ValueError(f"level must be 1, 2 or 3; got {level}")


# --- The organisers' own wall -------------------------------------------------------------------
# Measured from videos/compressed/microban_climb.mp4 (frame 0) on the workshop site: the board quad
# gives a homography, orange blobs give cup centres, and those snap to the grid. Rows sit every 3
# or 4 pitches alternating (≈0.0525 m), with one to four cups per row scattered across a central
# band, columns 15..36 (x ≈ 0.24..0.56 m). Rows below j≈22 were behind the robot in that frame.
ORGANISERS_ROWS: dict[int, tuple[int, ...]] = {
    75: (15, 18, 28),
    71: (21, 25, 30, 33),
    67: (19, 28, 36),
    64: (22, 25, 31, 34),
    60: (16, 19, 28),
    57: (22, 26, 32),
    53: (17, 20, 28),
    50: (23, 26, 31, 34),
    46: (17, 24, 29),
    43: (21, 24, 29, 32),
    39: (16, 20, 27),
    36: (23, 30, 33),
    32: (16, 22, 26),
    29: (20, 23, 28, 31),
    25: (16, 25, 30),
    22: (23, 28, 30),
    18: (17, 22, 27),
    15: (18, 24, 31),
    11: (20, 26, 32),
    8: (19, 24, 29),
    4: (15, 20, 25),
    1: (20, 26, 31),
}


def organisers_wall() -> WallModel:
    """The layout from the organisers' render, as closely as we can measure it.

    Rows above j=22 are measured; the lower rows were occluded by the robot in
    that frame and follow the same spacing and density.
    """
    holds = [
        Hold(id=f"h{j:02d}_{i:02d}", position=hold_position(i, j), kind=KIND)
        for j in sorted(ORGANISERS_ROWS, reverse=True)
        for i in ORGANISERS_ROWS[j]
    ]
    by_row = {j: sorted(ORGANISERS_ROWS[j]) for j in ORGANISERS_ROWS}
    feet_row, hands_row = 1, 11
    start = Stance(
        contacts={
            "left_foot": f"h{feet_row:02d}_{by_row[feet_row][0]:02d}",
            "right_foot": f"h{feet_row:02d}_{by_row[feet_row][-1]:02d}",
            "left_hand": f"h{hands_row:02d}_{by_row[hands_row][0]:02d}",
            "right_hand": f"h{hands_row:02d}_{by_row[hands_row][-1]:02d}",
        }
    )
    return WallModel(id="organisers", level=3, holds=holds, start=start)
