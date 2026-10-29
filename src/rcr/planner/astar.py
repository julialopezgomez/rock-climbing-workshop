"""Reachability A* over stances, checked lazily with the optimiser's feasibility (D2).

Two levels, as FARO orders its checks cheap to expensive [Ours, after FARO and CASSR]:
  - the reach graph: a node is a stance (limb -> hold); a move changes one limb's hold. A move is
    in the graph if the new stance keeps our posture rules and its trunk region is not empty
    (reach.py). Building it is milliseconds per node;
  - `feasible(wall, a, b)` (1-17 s a move with the optimiser's): asked only about the moves of
    the reach graph's best path, from the start. A rejected move leaves the graph and the path is
    searched again (LazySP, Dellin & Srinivasa 2016), so feasible() sees a path's moves, not
    every node the search touches.
Cost 1 per move, so the plan has the fewest moves (time is the tie-break [Rules §8]). The goal is
a hand on one of the highest holds [Rules §8]; if no plan gets there, the plan goes as high as it
can (height first [Rules §8]).
"""

import heapq
import math
import time
from dataclasses import dataclass, field
from itertools import count, pairwise

import numpy as np

from rcr.planner import reach as rc
from rcr.planner.cache import Cache
from rcr.types import LIMBS, ContactPlan, Feasibility, FeasibilityFn, Stance, WallModel

HANDS = ("right_hand", "left_hand")
FEET = ("right_foot", "left_foot")
# The postures a stance may take [Ours]: the ranges of the 36 stances of the example plan, every
# move of which the optimiser solved (data/examples/contact_plan_hard.json), each widened by
# ENVELOPE_MARGIN. Kinematics alone allow far more, and the optimiser can hold almost none of it:
# of 72 moves sampled from the reach graph without this, it accepted 2, and rejected 58 because
# the new stance could not be held (scripts/plan_eval.py moves). A proxy for torque and
# collisions until a stance model is learned from the optimiser's static poses (README).
ENVELOPE = {  # [m] (min, max)
    "hands_apart": (0.150, 0.255),  # right hand x - left hand x
    "feet_apart": (0.075, 0.240),  # right foot x - left foot x
    "hand_above_feet": (0.097, 0.323),  # each hand's z - the feet's mean z
    "hands_level": (0.0, 0.150),  # |right hand z - left hand z|
    "feet_level": (0.0, 0.120),  # |right foot z - left foot z|
    # the hands' mean x - the feet's mean x. The example climbs straight up (-0.030..0.037); a
    # traverse needs more: with this, all of L3 seeds 0-9 have a plan in the reach graph (3 of 10
    # with the example's range). Not yet measured against the optimiser.
    "lean": (-0.100, 0.100),
}
ENVELOPE_MARGIN = 0.03  # [m] two grid pitches [Ours]
MAX_HAND_ABOVE_FEET = ENVELOPE["hand_above_feet"][1] + ENVELOPE_MARGIN
# [m] a limb never moves down more than this [Ours]: every solved move of the example plan goes up,
# and the generated walls' traverses never drop. It halves the moves to consider per stance.
MAX_DROP = 0.0
# A*'s heuristic weight [Ours]. The heuristic is a lower bound but a loose one (it allows a hand
# MAX_HAND_ABOVE_FEET, 0.35 m, a move; solved moves are 0.03-0.225 m), and with weight 1 the
# search visits most of the graph. With weight w a plan has at most w times the fewest moves of
# the graph (multi-cap A* uses 1.5).
WEIGHT = 3.0
MAX_REJECTIONS = 200  # feasible() rejections before giving up on the top [Ours]

Key = tuple[str, str, str, str]  # hold ids in LIMBS order


@dataclass
class Result:
    plan: ContactPlan
    reached_top: bool  # a hand ends on one of the highest holds
    seconds: float
    expansions: int = 0  # reach-graph nodes expanded, all searches together
    searches: int = 0
    checks: int = 0  # feasible() calls
    rejected: list[tuple[Stance, Stance, str]] = field(default_factory=list)


def posture(x: dict, z: dict):
    """Inside ENVELOPE (each range widened by ENVELOPE_MARGIN), and no hand below a foot (the
    optimiser's rule). x, z: limb -> hold coordinate [m], a number or an array (one per
    candidate hold): the answer is then an array too."""
    e = ENVELOPE_MARGIN + 1e-9
    feet_z = (z["right_foot"] + z["left_foot"]) / 2
    values = {
        "hands_apart": [x["right_hand"] - x["left_hand"]],
        "feet_apart": [x["right_foot"] - x["left_foot"]],
        "hand_above_feet": [z["right_hand"] - feet_z, z["left_hand"] - feet_z],
        "hands_level": [abs(z["right_hand"] - z["left_hand"])],
        "feet_level": [abs(z["right_foot"] - z["left_foot"])],
        "lean": [(x["right_hand"] + x["left_hand"]) / 2 - (x["right_foot"] + x["left_foot"]) / 2],
    }
    ok = (
        np.minimum(z["right_hand"], z["left_hand"])
        >= np.maximum(z["right_foot"], z["left_foot"]) - 1e-9
    )
    for name, vs in values.items():
        low, high = ENVELOPE[name]
        for v in vs:
            ok = ok & (v >= low - e) & (v <= high + e)
    return ok


def key(stance: Stance) -> Key:
    return tuple(stance.contacts[limb] for limb in LIMBS)


def stance(k: Key) -> Stance:
    return Stance(contacts=dict(zip(LIMBS, k, strict=True)))


class Graph:
    """The reach graph of one wall, built as the search asks for it."""

    def __init__(self, wall: WallModel, reach: rc.Reach):
        self.reach = reach
        self.ids = [h.id for h in wall.holds]
        self.x = {h.id: h.position[0] for h in wall.holds}
        self.z = {h.id: h.position[2] for h in wall.holds}
        points = rc.contact_points(wall, reach)
        self.point = points
        self.points = np.array([points[i] for i in self.ids])
        self.ids_array = np.array(self.ids)
        self.hold_x = np.array([self.x[i] for i in self.ids])
        self.hold_z = np.array([self.z[i] for i in self.ids])
        self.top_z = max(self.z.values())
        self.top = {i for i, z in self.z.items() if z >= self.top_z - 1e-9}
        self._children: dict[Key, list[Key]] = {}
        self._valid: dict[Key, bool] = {}
        self.lps = 0

    def posture_ok(self, k: Key) -> bool:
        """Every limb on its own hold, and the posture rules (`posture`)."""
        if len(set(k)) < 4:
            return False
        x = {limb: self.x[h] for limb, h in zip(LIMBS, k, strict=True)}
        z = {limb: self.z[h] for limb, h in zip(LIMBS, k, strict=True)}
        return bool(posture(x, z))

    def children(self, k: Key) -> list[Key]:
        """The stances one move from k that may be in the graph: a limb to another hold no more
        than MAX_DROP lower, keeping the posture rules, passing the cheap emptiness test of the
        new trunk region. All holds at once per limb. The exact test (an LP) is `valid`, which
        the search asks only of the stances it expands: the cheap test is right for 95 % of
        the stances it passes, and most children are never expanded."""
        if k in self._children:
            return self._children[k]
        out = []
        x = {limb: self.x[h] for limb, h in zip(LIMBS, k, strict=True)}
        z = {limb: self.z[h] for limb, h in zip(LIMBS, k, strict=True)}
        for m, limb in enumerate(LIMBS):
            allowed = (self.hold_z >= z[limb] - MAX_DROP - 1e-9) & ~np.isin(self.ids_array, k)
            allowed &= posture({**x, limb: self.hold_x}, {**z, limb: self.hold_z})
            if not allowed.any():
                continue
            three = {other: self.point[k[n]] for n, other in enumerate(LIMBS) if n != m}
            candidates = np.flatnonzero(allowed)
            u = np.minimum(
                rc.region(self.reach, three), rc.bound(self.reach, limb, self.points[candidates])
            )
            for index in candidates[rc.maybe_nonempty(self.reach, u)]:
                out.append(k[:m] + (self.ids[index],) + k[m + 1 :])
        self._children[k] = out
        return out

    def valid(self, k: Key) -> bool:
        """Is the stance's trunk region not empty (the exact test, one LP; cached)?"""
        if k not in self._valid:
            self.lps += 1
            u = rc.region(
                self.reach, {limb: self.point[h] for limb, h in zip(LIMBS, k, strict=True)}
            )
            self._valid[k] = rc.centre(self.reach, u) is not None
        return self._valid[k]

    def is_goal(self, k: Key) -> bool:
        return k[LIMBS.index("right_hand")] in self.top or k[LIMBS.index("left_hand")] in self.top

    def best_hand(self, k: Key) -> float:
        return max(self.z[k[LIMBS.index(h)]] for h in HANDS)

    def h(self, k: Key) -> int:
        """Moves still needed, a lower bound (admissible and consistent), d = MAX_HAND_ABOVE_FEET:
          - at the goal one hand is at the top, the other no lower than the feet's mean, which is
            at least top - d; each foot at least top - 2d (the feet's mean is, and a foot is no
            higher than the hands);
          - a hand rises at most d a move (it ends at most d above the feet's mean and started no
            lower than the feet); a foot at most 2d (it ends no higher than the lowest hand, and
            started no lower than the highest hand less 2d);
        a move changes one limb, so the limbs' counts add up."""
        d, top = MAX_HAND_ABOVE_FEET, self.top_z
        z = dict(zip(LIMBS, (self.z[i] for i in k), strict=True))

        def moves(rise: float, step: float) -> int:
            return math.ceil(max(0.0, rise) / step - 1e-9)

        a, b = (z[h] for h in HANDS)
        hands = min(
            moves(top - a, d) + moves(top - d - b, d), moves(top - b, d) + moves(top - d - a, d)
        )
        return hands + sum(moves(top - 2 * d - z[f], 2 * d) for f in FEET)


@dataclass
class Banned:
    """Moves feasible() ruled out. Its checks run in a fixed order (optimiser edge_checks.CHECKS)
    and its reason names the one that failed. What that check looks at says how far the
    verdict reaches [Ours, after FARO: a failure low in the cascade prunes all above it]:
      stance           stance b cannot be held                  -> every move into b
      start            stance a cannot be held                  -> every move out of a
      release, lift    the moving limb cannot leave a's hold    -> every move of that limb out of a
      arrival          the moving limb cannot settle on b's     -> every move of that limb into b
      anything else    the move itself                          -> that move
    """

    nodes: set = field(default_factory=set)
    leaving: set = field(default_factory=set)  # (stance, moving limb's index)
    arriving: set = field(default_factory=set)  # (stance, moving limb's index)
    moves: set = field(default_factory=set)

    def add(self, a: Key, b: Key, reason: str) -> None:
        (m,) = [n for n in range(4) if a[n] != b[n]]
        check = reason.removeprefix("cached: ").split(":")[0].strip()
        if check == "stance":
            self.nodes.add(b)
        elif check == "start":
            self.nodes.add(a)
        elif check in ("release", "lift"):
            self.leaving.add((a, m))
        elif check == "arrival":
            self.arriving.add((b, m))
        else:
            self.moves.add((a, b))

    def allows(self, a: Key, b: Key, m: int) -> bool:
        return not (
            b in self.nodes
            or a in self.nodes
            or (a, m) in self.leaving
            or (b, m) in self.arriving
            or (a, b) in self.moves
        )


def search(graph: Graph, start: Key, banned: Banned, weight: float = WEIGHT):
    """A* on the reach graph without the banned moves: (path to a goal, or to the highest
    stance reached if no goal can be; True if it is a goal; nodes expanded)."""
    tie = count()
    g = {start: 0}
    parent: dict[Key, Key] = {}
    heap = [(weight * graph.h(start), -graph.best_hand(start), next(tie), start)]
    closed = set()
    best = start
    while heap:
        _, _, _, k = heapq.heappop(heap)
        if k in closed or (k != start and not graph.valid(k)):
            continue
        closed.add(k)
        if graph.is_goal(k):
            return _path(parent, k), True, len(closed)
        if (graph.best_hand(k), -g[k]) > (graph.best_hand(best), -g[best]):
            best = k
        for child in graph.children(k):
            m = next(n for n in range(4) if child[n] != k[n])
            if g[k] + 1 >= g.get(child, math.inf) or not banned.allows(k, child, m):
                continue
            g[child] = g[k] + 1
            parent[child] = k
            f = g[child] + weight * graph.h(child)
            heapq.heappush(heap, (f, -graph.best_hand(child), next(tie), child))
    return _path(parent, best), False, len(closed)


def _path(parent: dict[Key, Key], k: Key) -> list[Key]:
    path = [k]
    while path[-1] in parent:
        path.append(parent[path[-1]])
    return path[::-1]


def solve(
    wall: WallModel,
    feasible: FeasibilityFn,
    reach: rc.Reach | None = None,
    weight: float = WEIGHT,
    max_rejections: int = MAX_REJECTIONS,
) -> Result:
    """The plan and how it was found. See the module docstring. `feasible` is asked through a
    layout cache (cache.Cache) unless it is one already, so a move rejected once is not asked
    again elsewhere on the wall."""
    clock = time.perf_counter()
    graph = Graph(wall, reach or rc.load())
    start = key(wall.start)
    if not isinstance(feasible, Cache):
        feasible = Cache(feasible)
    banned = Banned()
    verified: dict[tuple[Key, Key], Feasibility] = {}
    result = Result(plan=ContactPlan(wall.id, [wall.start]), reached_top=False, seconds=0.0)
    while True:
        path, reached_top, expanded = search(graph, start, banned, weight)
        result.searches += 1
        result.expansions += expanded
        good = [path[0]]
        for a, b in pairwise(path):
            if (a, b) not in verified:
                result.checks += 1
                verdict = feasible(wall, stance(a), stance(b))
                if not verdict.feasible:
                    banned.add(a, b, verdict.reason)
                    result.rejected.append((stance(a), stance(b), verdict.reason))
                    break
                verified[(a, b)] = verdict
            good.append(b)
        done = len(good) == len(path)
        if done or len(result.rejected) >= max_rejections:
            result.plan = ContactPlan(wall_id=wall.id, stances=[stance(k) for k in good])
            result.reached_top = done and reached_top
            result.seconds = time.perf_counter() - clock
            return result
