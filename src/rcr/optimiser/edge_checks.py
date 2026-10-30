"""Feasibility of one move, stance a -> b, from cheap checks at its edges [Ours, after FARO].

A full transition (transition.py) takes 10 s to minutes, and longer when it fails. Most moves that
cannot be done fail at an edge, where the contacts change, so these poses are checked
first, cheapest first, stopping at the first that fails:

    rules     every limb on a known hold, no two on one, hands not below feet, one limb moves
    reach     no two holds of b further apart than their limbs can span: each limb's
              reach from its shoulder or hip, plus the distance between those on the trunk
              (robot_model.limb_reach, an upper bound from the URDF). Microseconds, no solve.
    cache     answered before (move_cache, on this wall or any with the same holds around the
              move): a stored move is feasible, at once. With move_cache.REJECTIONS (off for now)
              a stored rejection is not, and every rejection below is stored there.
    stance    b can be held: four contacts, balance, torque limit, clear of the holds and wall
    release   the moving ball still in its socket of a but carrying nothing: the other three
              hold the robot (a hand reaching up a 0.15 m row fails here, on torque)
    arrival   the same at its socket of b (a foot that must end inside a hold fails here)
    lift      the moving ball out of its socket of a, at one of EXITS: the arm or leg can get
              out (a forearm that would hit the hold above fails here)
    path      the move itself, coarsely (transition.coarse: half the samples, no midpoints),
              from stance a's pose: a leg that must sweep through the hold above a foot while
              another limb moves fails here. Seconds, not the full solve's minutes.

Each pose is one sample of the transition's physics: the static pose NLP (transition.py), made
clear by the same exact collision loop. A local solve can miss a pose that exists, so each is
tried from two starts: the neighbouring pose of the motion (stance a's pose for the release, the
release for the lift, stance b's pose for the arrival), then a fresh guess. Passing does not
promise the full transition succeeds (the coarse path has fewer samples and no midpoints);
failing means it could not be done from these poses. cost: the peak |torque| over the poses and
the coarse path, as a fraction of the limit. trace() also gives the time each check took.

Tried and dropped [Ours]: a quick pass before the path, the straight motion (in q) through the
edge poses. On the example plans it passed none of 8 moves that solve: the ball's straight line
from above hold a to above hold b runs through hold b, and the loaded balls drift up to 30 mm
off their holds between poses. Mending both is what the coarse path does.
"""

from __future__ import annotations

import itertools
import time

import numpy as np

from rcr.optimiser import robot_model as rm
from rcr.optimiser import transition as tr
from rcr.optimiser.move_cache import default_cache
from rcr.types import LIMBS, Feasibility, Stance, WallModel

CHECKS = (
    "rules",
    "reach",
    "cache",
    "stance",
    "start",
    "release",
    "arrival",
    "lift",
    "path",
)  # in order
# Ways out of the socket for the lift pose [m, wall frame]: straight up, then up and out from the
# wall, all inside its ~60 deg opening cone. Straight up alone rejected a foot whose next hold is
# 75 mm above, which the transition solves by leaving up and out [Ours].
EXITS = ((0.0, 0.0, tr.LIFT), (0.0, -0.02, 0.03), (0.0, -0.035, 0.025))
_cache: dict[tuple, tuple[bool, float, str, np.ndarray | None]] = {}


def feasibility(
    wall: WallModel, a: Stance, b: Stance, cfg: tr.TransitionConfig | None = None
) -> Feasibility:
    """Can the robot go from stance a to stance b? See the module docstring."""
    return trace(wall, a, b, cfg)[0]


def trace(
    wall: WallModel, a: Stance, b: Stance, cfg: tr.TransitionConfig | None = None
) -> tuple[Feasibility, list[tuple[str, float]]]:
    """feasibility(wall, a, b), and (check, seconds) for each check that ran, in order: the
    last one is the one that rejected the move, or the one that passed it."""
    cfg = cfg or tr.TransitionConfig()
    ran = []

    def timed_check(name, check, *args):
        start = time.perf_counter()
        out = check(*args)
        ran.append((name, time.perf_counter() - start))
        return out

    why = timed_check("rules", rules, wall, a, b) or timed_check("reach", reach, wall, b, cfg)
    if ran[-1][0] == "reach" and why:
        why = f"reach: {why}"
    if why:
        return Feasibility(False, reason=why), ran
    cache = default_cache() if cfg.settle else None
    if cache is not None:  # answered before, on this wall or one with the same holds
        entry = timed_check("cache", cache.get, wall, a, b, cfg)
        if entry is not None and not entry.feasible:
            return Feasibility(False, reason=f"{entry.reason} (cached)"), ran
        if entry is not None:
            return Feasibility(True, cost=np.abs(entry.result.tau).max() / cfg.tau_limit), ran
    verdict = _checks(wall, a, b, cfg, timed_check)
    if cache is not None and not verdict.feasible:
        cache.put(wall, a, b, cfg, reason=verdict.reason)  # never checked again
    return verdict, ran


def _checks(wall, a, b, cfg, timed_check) -> Feasibility:
    """The edge poses, then the coarse path (the module docstring)."""
    moving = tr.moving_limb(a, b, 0)
    # (name, stance, unloaded limb, lift, the edge whose pose is tried first as the start)
    edges = (
        ("stance", b, None, (None,), None),
        ("start", a, None, (None,), None),
        ("release", a, moving, (None,), "start"),
        ("arrival", b, moving, (None,), "stance"),
        ("lift", a, moving, EXITS, "release"),
    )
    peak, poses = 0.0, {}
    for name, stance, unloaded, aways, near in edges:
        ok, torque, why, q = timed_check(
            name, _edge_any, wall, name, stance, unloaded, aways, poses.get(near), cfg
        )
        if not ok:
            return Feasibility(False, reason=f"{name}: {why}")
        peak, poses[name] = max(peak, torque), q
    ok, torque, why = timed_check("path", _path, wall, a, b, poses["start"], cfg)
    if not ok:
        return Feasibility(False, reason=f"path: {why}")
    return Feasibility(True, cost=max(peak, torque) / cfg.tau_limit)


def rules(wall: WallModel, a: Stance, b: Stance) -> str:
    """Why the move breaks a rule of ours, or "" [Ours: agreed for the sim's stances]."""
    holds = {h.id: h.position for h in wall.holds}
    for name, stance in (("a", a), ("b", b)):
        used = [stance.contacts[limb] for limb in LIMBS]
        if any(h is None for h in used):
            return f"stance {name}: every limb must be on a hold"
        if any(h not in holds for h in used):
            return f"stance {name}: unknown hold"
        if len(set(used)) < len(used):
            return f"stance {name}: two limbs on one hold"
        hand = min(holds[stance.contacts[limb]][2] for limb in ("right_hand", "left_hand"))
        foot = max(holds[stance.contacts[limb]][2] for limb in ("right_foot", "left_foot"))
        if hand < foot:
            return f"stance {name}: a hand below a foot"
    try:
        tr.moving_limb(a, b, 0)
    except ValueError as e:
        return str(e).split(": ", 1)[-1]
    return ""


def reach(wall: WallModel, stance: Stance, cfg: tr.TransitionConfig) -> str:
    """Why two of the stance's holds are further apart than their limbs can span, or "".
    Two balls can be no further apart than their reaches plus the distance between their limbs'
    first joints on the trunk (robot_model.limb_reach): never rejects a stance the robot can
    take."""
    points, limbs = tr.contact_points(wall, stance, cfg), rm.limb_reach()
    for i, j in itertools.combinations(range(len(LIMBS)), 2):
        (root_i, reach_i), (root_j, reach_j) = limbs[LIMBS[i]], limbs[LIMBS[j]]
        span = reach_i + reach_j + float(np.linalg.norm(root_i - root_j))
        apart = float(np.linalg.norm(points[:, i] - points[:, j]))
        if apart > span:
            return (
                f"{LIMBS[i]} on {stance.contacts[LIMBS[i]]} and {LIMBS[j]} on "
                f"{stance.contacts[LIMBS[j]]} are {apart:.3f} m apart, more than their "
                f"{span:.3f} m span"
            )
    return ""


def _edge_any(wall, name, stance, unloaded, aways, near, cfg):
    """_edge at the first of `aways` that works (each away: a ball offset, or None)."""
    for away in aways:
        ok, torque, why, q = _edge(wall, name, stance, unloaded, away, near, cfg)
        if ok:
            break
    return ok, torque, why, q


def _edge(wall, name, stance, unloaded, away, near, cfg):
    """(feasible, peak |torque|, why not, pose) of one edge pose, cached per wall and pose.
    Tried from `near` (a neighbouring pose of the motion) if given, then from a fresh guess."""
    wall_key = (wall.id, tuple((h.id, tuple(h.position)) for h in wall.holds))
    contacts = tuple(stance.contacts[limb] for limb in LIMBS)
    key = (wall_key, contacts, unloaded, away, repr(cfg))
    if key not in _cache:
        starts = [q for q in (near, tr.initial_guess(wall, stance, cfg)) if q is not None]
        why = ""
        for guess in starts:
            try:
                s = tr.static_pose(
                    wall, stance, guess, cfg, unloaded=unloaded, away=away, what=f"{name} pose"
                )
            except tr.OptimisationError as e:
                why = str(e).split(": ", 1)[-1]
                continue
            _cache[key] = (True, float(np.abs(s.tau).max()), "", s.q[0])
            break
        else:
            _cache[key] = (False, 0.0, why, None)
    return _cache[key]


def _path(wall, a, b, q_start, cfg) -> tuple[bool, float, str]:
    """(feasible, peak |torque|, why not) of the coarse move from stance a's pose, cached."""
    wall_key = (wall.id, tuple((h.id, tuple(h.position)) for h in wall.holds))
    key = (wall_key, "path", tuple(a.contacts.values()), tuple(b.contacts.values()), repr(cfg))
    if key not in _cache:
        try:
            r = tr.solve_transition(wall, a, b, q_start, tr.coarse(cfg), rescue=False)
            _cache[key] = (True, float(np.abs(r.tau).max()), "", None)
        except tr.OptimisationError as e:
            _cache[key] = (False, 0.0, str(e).split(": ", 1)[-1], None)
    return _cache[key][:3]
