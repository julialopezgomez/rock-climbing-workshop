"""Solved moves, stored and reused on any wall where the same holds come up [Ours, after FARO].

With TransitionConfig.settle, a move goes from stance a's own pose to stance b's (solve_move), so
its result depends on the two stances and the holds around them only, never on the moves before
it. And moving every hold by whole grid holes moves the whole problem with it: gravity points
down and the wall is flat, so the solved motion moves too (the trunk's position; the angles stay).
So one solved move serves every wall where its holds are laid out alike.

Key (a "mode transition" in FARO's terms, in grid holes, not hold names or heights):
    the moving limb;
    stance a's four holds and the moving limb's new hold, as offsets from where the moving limb
        starts (whole grid holes along the wall and up);
    every other hold in the box around those five, widened by NEAR_SIDE and NEAR_ABOVE, the
        same way (they can be in the way);
    the version: the hooks, the robot model, the hold mesh and its convex pieces, the settings
        and CACHE_VERSION.
Value: feasible or not, and why not; the solve time; for a feasible move its motion (q, tau and
forces per sample), stored relative to where the moving limb starts.

A new version (real limbs, other settings, a changed solver) never sees the old entries: the
cache starts empty for it. On reuse the stored motion is checked again for collisions on the
actual wall before it is trusted (verify), as a hold outside the box could still be in the way.

Storage: DIR/<version>/<key>.npz, one small file per move, written whole (safe to stop at any time).
Off by default for now: a run uses it only when asked (default_cache: RCR_MOVE_CACHE=on or a
folder; the runner's --cache). Then optimise, solve_plan and the runner store and reuse solved
moves, and feasibility() answers stored moves at once. Rejections (feasibility()'s
and failed solves) are stored and trusted only with REJECTIONS, off for now: a move that failed
once is checked or solved again.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rcr.types import GRID_PITCH, LIMBS, Stance, WallModel

CACHE_VERSION = 3  # bump when the solver or the key changes what a stored move means
# The region of other holds that belong to a key: the box around the move's five holds, widened
# by NEAR_SIDE to the sides and below and by NEAR_ABOVE above (the head, a hook lifted out of its
# socket) [Ours]. Smaller is reused more; a hold outside that is in the way anyway is caught when
# the stored motion is checked again on the actual wall (verify). 0.3 m all round found no
# repeat on a ladder climbed to 0.8 m: the rungs below the feet kept changing the key.
NEAR_SIDE = 0.12  # [m]
NEAR_ABOVE = 0.15  # [m]
DEFAULT_DIR = Path(__file__).resolve().parents[3] / "build" / "move_cache"
# Store infeasible verdicts (feasibility()'s rejections, failed solves) and answer them at once,
# "(cached)". Off for now: a coarse rejection stored by planning blocked the full solve of the
# same move, which could have succeeded [Ours]. Rejections already stored are then ignored.
REJECTIONS = False


def default_cache() -> MoveCache | None:
    """The cache a run uses, from the environment variable RCR_MOVE_CACHE: DEFAULT_DIR if it is
    "on", that folder if it is a path, and None (no cache) if it is unset or "off": off by
    default for now [Ours]."""
    import os

    where = os.environ.get("RCR_MOVE_CACHE", "off")
    if where.lower() in ("", "off"):
        return None
    return MoveCache(DEFAULT_DIR if where.lower() == "on" else where)


@dataclass
class Entry:
    feasible: bool
    reason: str  # why not, if infeasible
    solve_time: float  # [s] what it took to find out
    result: object | None = None  # transition.TransitionResult on this wall, if feasible


class MoveCache:
    def __init__(self, directory: Path | str = DEFAULT_DIR):
        self.directory = Path(directory)

    # --- keys ----------------------------------------------------------------------------------

    @staticmethod
    def version(cfg) -> str:
        """What the stored motions depend on besides the holds: change any and the cache is new."""
        from rcr.optimiser import collision as col
        from rcr.optimiser import robot_model as rm
        from rcr.robot import hooks

        parts = [
            f"cache {CACHE_VERSION}",
            repr((hooks.HOOKS, hooks.BALL_RADIUS)),
            _sha(rm.urdf_path()),
            _sha(rm.mjcf_path()),
            _sha(col.HOLD_STL),
            *sorted(_sha(p) for p in col.PIECES_CACHE.glob("hold_pieces_*.npz")),
            repr(cfg),
        ]  # fmt: skip
        return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]

    @staticmethod
    def key(wall: WallModel, a: Stance, b: Stance) -> tuple[str, np.ndarray] | None:
        """(key, reference point [m]) of the move a -> b, or None if a hold is off the grid
        relative to the others (then it is not cached)."""
        from rcr.optimiser import transition as tr

        moving = tr.moving_limb(a, b, 0)
        pos = {h.id: np.array(h.position, dtype=float) for h in wall.holds}
        ref = pos[a.contacts[moving]]
        involved = [a.contacts[limb] for limb in LIMBS] + [b.contacts[moving]]

        def cell(hold: str) -> tuple[int, int] | None:
            dx, dz = (pos[hold] - ref)[[0, 2]] / GRID_PITCH
            ix, iz = round(dx), round(dz)
            return (ix, iz) if abs(dx - ix) < 1e-6 and abs(dz - iz) < 1e-6 else None

        cells = [cell(h) for h in involved]
        xs = [pos[h][0] for h in involved]
        zs = [pos[h][2] for h in involved]
        near = sorted(
            cell(h)
            for h in pos
            if h not in involved
            and min(xs) - NEAR_SIDE <= pos[h][0] <= max(xs) + NEAR_SIDE
            and min(zs) - NEAR_SIDE <= pos[h][2] <= max(zs) + NEAR_ABOVE
        )
        if None in cells or None in near:
            return None
        text = json.dumps({"moving": moving, "holds": cells, "near": near})
        return hashlib.sha256(text.encode()).hexdigest()[:24], ref

    def _path(self, cfg, key: str) -> Path:
        return self.directory / self.version(cfg) / f"{key}.npz"

    # --- reading and writing -------------------------------------------------------------------

    def get(self, wall: WallModel, a: Stance, b: Stance, cfg) -> Entry | None:
        """The stored move, its motion moved onto this wall; None if not stored."""
        from rcr.optimiser import robot_model as rm
        from rcr.optimiser import transition as tr

        found = self.key(wall, a, b)
        if found is None:
            return None
        key, ref = found
        path = self._path(cfg, key)
        if not path.exists():
            return None
        with np.load(path) as stored:
            meta = json.loads(str(stored["meta"]))
            if not meta["feasible"]:
                return Entry(False, meta["reason"], meta["solve_time"]) if REJECTIONS else None
            q = stored["q"].copy()
            q[:, [0, 2]] += ref[[0, 2]]  # onto this wall
            result = tr.TransitionResult(
                q=q,
                dq=np.column_stack([rm.difference(q[0], qk) for qk in q]),
                tau=stored["tau"],
                forces={limb: stored[f"force_{limb}"] for limb in LIMBS},
                moving=meta["moving"],
                solve_time=0.0,
                iterations=0,
            )
        return Entry(True, "", meta["solve_time"], result)

    def put(self, wall, a, b, cfg, result=None, reason: str = "", solve_time: float = 0.0):
        """Store the move: its result if it was solved, else why not (only with REJECTIONS)."""
        if result is None and not REJECTIONS:
            return
        found = self.key(wall, a, b)
        if found is None:
            return
        key, ref = found
        path = self._path(cfg, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"feasible": result is not None, "reason": reason, "solve_time": solve_time}
        arrays = {}
        if result is not None:
            q = np.array(result.q, dtype=float)
            q[:, [0, 2]] -= ref[[0, 2]]  # relative to where the moving limb starts
            meta["moving"] = result.moving
            arrays = {"q": q, "tau": result.tau}
            arrays |= {f"force_{limb}": result.forces[limb] for limb in LIMBS}
        temporary = path.with_suffix(".tmp.npz")
        np.savez(temporary, meta=np.array(json.dumps(meta)), **arrays)
        temporary.replace(path)

    def size(self, cfg) -> tuple[int, int]:
        """(feasible, infeasible) moves stored for this version."""
        feasible = infeasible = 0
        for path in (self.directory / self.version(cfg)).glob("*.npz"):
            with np.load(path) as stored:
                if json.loads(str(stored["meta"]))["feasible"]:
                    feasible += 1
                else:
                    infeasible += 1
        return feasible, infeasible


def solve_move_cached(wall, a, b, cfg, cache: MoveCache, index: int = 0):
    """(TransitionResult, from_cache) of the move a -> b with cfg.settle: from the cache if it is
    stored and still clear on this wall, else solved (solve_move) and stored. With REJECTIONS, a
    stored infeasible move raises OptimisationError at once, naming it as cached."""
    import time

    from rcr.optimiser import transition as tr

    if not cfg.settle:
        raise ValueError("the cache holds moves solved with TransitionConfig.settle only")
    entry = cache.get(wall, a, b, cfg)
    if entry is not None and not entry.feasible:
        moving = tr.moving_limb(a, b, index)
        raise tr.OptimisationError(
            f"transition {index} ({moving} to {b.contacts[moving]}): {entry.reason} (cached)"
        )
    if entry is not None and verify(wall, a, b, entry.result, cfg):
        return entry.result, True
    start = time.perf_counter()
    try:
        result = tr.solve_move(wall, a, b, cfg, index)
    except tr.OptimisationError as e:
        reason = f"full solve: {str(e).split(': ', 1)[-1]}"
        cache.put(wall, a, b, cfg, reason=reason, solve_time=time.perf_counter() - start)
        raise
    cache.put(wall, a, b, cfg, result, solve_time=time.perf_counter() - start)
    return result, False


def verify(wall, a, b, result, cfg) -> bool:
    """Is a stored motion clear of this wall's holds (all of them), the wall and itself? Always
    yes without cfg.avoid_collisions (then nothing is checked)."""
    if not cfg.avoid_collisions:
        return True
    from rcr.optimiser import avoidance as av
    from rcr.optimiser import collision as col
    from rcr.optimiser import robot_model as rm

    moving = result.moving
    contacts = av.transition_contacts(a, b, moving, cfg)
    contacts += [dict(b.contacts)] * (len(result.q) - len(contacts))  # the settle's samples
    own = {limb: {a.contacts[limb], b.contacts[limb]} for limb in LIMBS}
    holds = {h.id: h for h in wall.holds}
    model = rm.load_robot_model()
    clearance = cfg.collision_clearance
    checks = col.check_trajectory(model, holds, result.q[0], result.dq, contacts, own, clearance)
    return all(c.gap >= -col.TOUCHING for c in checks)


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


# --- Seeds: a solved move reused as the start of the same move elsewhere ------------------------
# Used without TransitionConfig.settle only (iter_plan), where a move starts where the previous
# one ended: on a regular wall, most moves repeat a row up [Ours].

SEED_MATCH = 0.02  # [m] layouts this close make a solved move a good seed [Ours]


def layout(wall: WallModel, a: Stance, b: Stance, cfg) -> np.ndarray:
    """3x8: the contact points of a, then b, relative to the moving limb's start point. Two
    transitions with the same layout are the same move, shifted: one's dq can seed the other."""
    from rcr.optimiser import transition as tr

    start = tr.contact_point(wall, a.contacts[tr.moving_limb(a, b, 0)], cfg)
    both = np.hstack([tr.contact_points(wall, a, cfg), tr.contact_points(wall, b, cfg)])
    return both - start[:, None]


def same_layout(one: np.ndarray, other: np.ndarray) -> bool:
    return bool(np.abs(one - other).max() <= SEED_MATCH)


def shifted_seed(q0: np.ndarray, previous_q: np.ndarray, shift: np.ndarray, cfg):
    """A seed (NV x samples) from a solved motion of the same move elsewhere: its configurations
    moved by `shift` [m] (the trunk; joints and trunk turn as they were), in the tangent space at
    q0. The gap between q0 and the moved start fades out over phase A, so the seed starts at q0
    and follows the solved motion from the release on. Reusing dq instead (relative to the other
    start) put the arm 15 mm into a hold: the starts differ by up to ~20 deg in the joints."""
    from rcr.optimiser import robot_model as rm

    moved = np.array(previous_q, dtype=float)
    moved[:, 0:3] += shift
    dq = np.column_stack([rm.difference(q0, q) for q in moved])
    gap = dq[:, 0].copy()
    for k in range(dq.shape[1]):
        dq[:, k] -= max(0.0, 1.0 - k / cfg.release_sample) * gap
    return dq
