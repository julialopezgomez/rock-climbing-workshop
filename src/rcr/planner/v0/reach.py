"""Reach sets and trunk regions: which stances the robot's kinematics allow [Ours, after SL1M §V
and CASSR Eq. 2].

Each limb's reach set K_limb (ball - trunk offsets, wall frame) comes from data/robot/reach.json
(written by scripts/make_reach.py), as support values h_limb along DIRECTIONS shared by all limbs:
K_limb = {o : n_k . o <= h_limb[k]}. A limb on a hold whose contact point is p allows the trunk
only in p - K_limb = {t : n_k . t <= h_limb[opp(k)] + n_k . p}, opp(k) being the opposite
direction. So a stance's trunk region, the trunk positions from which every limb reaches its
hold, is

    T = {t : N t <= u},  u = min over limbs of (h_limb[opp] + N p_limb),

capped by the trunk's distance from the wall (reach.json `trunk_y`). One small LP says whether it
is empty. A move of limb l to hold h keeps the other three, so its new region is
min(u_three, h_l[opp] + N p_h): every hold is tested at once (`moves`).
"""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from rcr.types import LIMBS, WallModel

REACH_FILE = Path(__file__).resolve().parents[4] / "data" / "robot" / "reach.json"
EPS = 1e-9


@dataclass(frozen=True)
class Reach:
    directions: np.ndarray  # (D, 3) unit vectors, closed under negation
    opposite: np.ndarray  # (D,) index of each direction's opposite
    reflected: dict[str, np.ndarray]  # limb -> (D,) h_limb[opposite]: the support of -K_limb
    contact_offset: np.ndarray  # [m] ball centre from Hold.position when it rests in the hold
    trunk_y: tuple[float, float]  # [m] the trunk's y range (y into the wall)
    cap: np.ndarray  # (D,) the trunk_y bound on each direction (inf where it doesn't apply)


@lru_cache(maxsize=4)
def load(path: Path = REACH_FILE) -> Reach:
    data = json.loads(Path(path).read_text())
    if data.get("schema") != "reach/1":
        raise ValueError(f"{path}: expected schema 'reach/1', found {data.get('schema')!r}")
    n = np.array(data["directions"], dtype=float)
    opposite = np.array([int(np.argmin(np.linalg.norm(n + d, axis=1))) for d in n])
    if not np.allclose(n[opposite], -n, atol=1e-5):
        raise ValueError(f"{path}: directions are not closed under negation")
    cap = np.full(len(n), np.inf)
    y_lo, y_hi = data["trunk_y"]
    cap[int(np.argmax(n[:, 1]))] = y_hi  # n = +y: t_y <= y_hi
    cap[int(np.argmin(n[:, 1]))] = -y_lo  # n = -y: -t_y <= -y_lo
    return Reach(
        directions=n,
        opposite=opposite,
        reflected={limb: np.array(data["support"][limb])[opposite] for limb in LIMBS},
        contact_offset=np.array(data["contact_offset"], dtype=float),
        trunk_y=(y_lo, y_hi),
        cap=cap,
    )


def contact_points(wall: WallModel, reach: Reach) -> dict[str, np.ndarray]:
    """Hold id -> where a resting ball's centre is [m], wall frame."""
    return {h.id: np.array(h.position) + reach.contact_offset for h in wall.holds}


def bound(reach: Reach, limb: str, points: np.ndarray) -> np.ndarray:
    """(..., D): the trunk region's bounds allowed by `limb` on each contact point in `points`."""
    return reach.reflected[limb] + points @ reach.directions.T


def region(reach: Reach, contacts: dict[str, np.ndarray]) -> np.ndarray:
    """u of the trunk region {t : N t <= u} for limbs on these contact points (any number)."""
    u = reach.cap.copy()
    for limb, p in contacts.items():
        u = np.minimum(u, bound(reach, limb, p))
    return u


def maybe_nonempty(reach: Reach, u: np.ndarray) -> np.ndarray:
    """Necessary for {N t <= u} to be non-empty: no direction's bound below its opposite's
    (u_k + u_opp(k) >= 0). Works on (..., D); cheap, so it filters before the LP."""
    return np.all(u + u[..., reach.opposite] >= -EPS, axis=-1)


def centre(reach: Reach, u: np.ndarray) -> tuple[np.ndarray, float] | None:
    """The point deepest in {N t <= u} and its depth [m] (the Chebyshev centre), or None if the
    region is empty. One LP."""
    finite = np.isfinite(u)
    n = reach.directions[finite]
    # maximise r subject to n_k . t + r <= u_k (the directions are unit vectors), r >= -1
    result = linprog(
        c=[0.0, 0.0, 0.0, -1.0],
        A_ub=np.hstack([n, np.ones((len(n), 1))]),
        b_ub=u[finite],
        bounds=[(None, None)] * 3 + [(-1.0, 1.0)],
        method="highs",
    )
    if result.status != 0 or result.x[3] < -EPS:
        return None
    return result.x[:3], float(result.x[3])
