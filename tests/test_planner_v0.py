"""Planner (D2): plans keep the rules and reach the top, the search is A* on the reach graph, the
optimiser's feasibility is asked of each stance expanded, and the cache keys moves by their
layout. No optimiser needed: feasibility is a stand-in here (scripts/plan_eval.py runs the real
one)."""

import dataclasses
from collections import deque
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from rcr import io, walls
from rcr.planner.v0 import astar, plan
from rcr.planner.v0 import reach as rc
from rcr.planner.v0.cache import Cache, layout_key
from rcr.types import LIMBS, ContactPlan, Feasibility, Hold, Stance, WallModel

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "data" / "examples"
WALLS = [walls.generate(level, seed) for level in (1, 2, 3) for seed in range(3)]


def always(wall, a, b):
    return Feasibility(True)


def moved(a: Stance, b: Stance) -> list[str]:
    return [limb for limb in LIMBS if a.contacts[limb] != b.contacts[limb]]


def test_the_reach_sets_allow_every_stance_the_optimiser_solved():
    reach = rc.load()
    assert np.allclose(reach.directions[reach.opposite], -reach.directions)
    wall = io.load_wall(EXAMPLES / "wall_hard.json")
    points = rc.contact_points(wall, reach)
    for s in io.load_contact_plan(EXAMPLES / "contact_plan_hard.json").stances:
        u = rc.region(reach, {limb: points[h] for limb, h in s.contacts.items()})
        assert rc.maybe_nonempty(reach, u)
        assert rc.centre(reach, u) is not None


def test_a_stance_out_of_reach_has_no_trunk_region():
    reach = rc.load()
    wall = walls.generate(1, 0)
    points = rc.contact_points(wall, reach)
    lowest, highest = wall.holds[0].id, wall.holds[-1].id
    contacts = dict(wall.start.contacts, right_hand=highest, left_foot=lowest)  # 1.1 m apart
    u = rc.region(reach, {limb: points[h] for limb, h in contacts.items()})
    assert rc.centre(reach, u) is None


@pytest.mark.parametrize("wall", WALLS, ids=lambda w: w.id)
def test_plans_keep_the_rules_and_reach_the_top(wall):
    result = astar.solve(wall, always)
    stances = result.plan.stances
    graph = astar.Graph(wall, rc.load())
    assert result.reached_top and graph.is_goal(astar.key(stances[-1]))
    assert stances[0] == wall.start
    ids = {h.id for h in wall.holds}
    for s in stances:
        assert set(s.contacts) == set(LIMBS) and set(s.contacts.values()) <= ids
        assert graph.posture_ok(astar.key(s)) and graph.valid(astar.key(s))
    for a, b in pairwise(stances):
        (limb,) = moved(a, b)
        assert graph.z[b.contacts[limb]] >= graph.z[a.contacts[limb]] - astar.MAX_DROP - 1e-9
    assert result.checks == result.expansions - 1, "feasible() asked once per stance expanded"


def test_plan_returns_a_contact_plan_for_the_wall():
    wall = walls.generate(1, 1)
    out = plan(wall, always)
    assert isinstance(out, ContactPlan) and out.wall_id == wall.id and len(out.stances) > 1


def test_rejected_moves_are_planned_around(monkeypatch):
    """Take the holds the left and the right hand first move to when every move is allowed. Now
    no stance can have the left hand on its hold (as the optimiser's 'stance' check would say),
    and no limb can arrive on the right hand's (a 'path' failure, of that move only). These
    rules name holds, so the same layout elsewhere on the wall gets another answer: no layout
    cache here."""
    monkeypatch.setattr("rcr.planner.v0.cache.layout_key", lambda *args: None)
    wall = walls.generate(1, 0)
    free = astar.solve(wall, always).plan.stances
    first = {}
    for a, b in pairwise(free):
        (limb,) = moved(a, b)
        first.setdefault(limb, b.contacts[limb])
    no_left, no_right = first["left_hand"], first["right_hand"]
    calls = []

    def picky(w, a, b):
        calls.append((a, b))
        if b.contacts["left_hand"] == no_left:
            return Feasibility(False, reason="stance: torque over the limit")
        if no_right in b.contacts.values() and no_right not in a.contacts.values():
            return Feasibility(False, reason="path: shin through a hold")
        return Feasibility(True)

    result = astar.solve(wall, picky)
    assert result.reached_top and len(result.rejected) >= 2
    for s in result.plan.stances[1:]:
        assert s.contacts["left_hand"] != no_left and no_right not in s.contacts.values()
    assert len({(astar.key(a), astar.key(b)) for a, b in calls}) == len(calls), "never asked twice"


def test_a_rejection_reaches_as_far_as_its_check():
    a = ("R3", "L3", "R0", "L0")
    b = ("R3", "L4", "R0", "L0")  # the left hand moves
    other = ("R4", "L3", "R0", "L0")  # the right hand moves
    cases = {
        "stance: torque": [(a, b, 1, False), (other, b, 1, False), (a, other, 0, True)],
        "start: torque": [(a, b, 1, False), (a, other, 0, False)],
        "release: torque": [
            (a, b, 1, False),
            (a, ("R3", "L5", "R0", "L0"), 1, False),
            (a, other, 0, True),
        ],
        "arrival: collides": [
            (a, b, 1, False),
            (("R3", "L2", "R0", "L0"), b, 1, False),
            (a, other, 0, True),
        ],
        "cached: path: shin": [(a, b, 1, False), (other, ("R4", "L4", "R0", "L0"), 1, True)],
    }
    for reason, expected in cases.items():
        banned = astar.Banned()
        banned.add(a, b, reason)
        for x, y, m, allowed in expected:
            assert banned.allows(x, y, m) is allowed, (reason, x, y)


def test_the_checks_the_ban_knows_are_the_optimisers():
    pytest.importorskip("casadi")
    from rcr.optimiser.edge_checks import CHECKS

    assert {"stance", "start", "release", "lift", "arrival", "path"} <= set(CHECKS)


def test_with_no_way_to_the_top_the_plan_goes_as_high_as_it_can():
    wall = walls.generate(1, 0)
    cut = [h for h in wall.holds if not 0.7 < h.position[2] < 1.2]  # a 0.5 m gap
    gap = dataclasses.replace(wall, holds=cut)
    result = astar.solve(gap, always)
    assert not result.reached_top
    graph = astar.Graph(gap, rc.load())
    highest = graph.best_hand(astar.key(result.plan.stances[-1]))
    assert 0.6 < highest <= 0.7 + 1e-9


def small_wall() -> WallModel:
    """A short ladder, 0.42 m tall, small enough to search exhaustively."""
    holds = []
    for k, j in enumerate(range(10, 39, 4)):
        for side, i in (("L", 20), ("R", 31)):
            holds.append(Hold(f"{side}{k}", walls.position(i, j)))
    start = Stance({"right_hand": "R4", "left_hand": "L4", "right_foot": "R0", "left_foot": "L0"})
    return WallModel(id="small", level=1, holds=holds, start=start)


def distances_to_goal(graph: astar.Graph, start: astar.Key) -> dict[astar.Key, int]:
    """Exact moves to the goal from every stance reachable from start (breadth first)."""
    seen, queue, edges = {start}, deque([start]), []
    while queue:
        k = queue.popleft()
        for c in graph.children(k):
            if graph.valid(c):
                edges.append((k, c))
                if c not in seen:
                    seen.add(c)
                    queue.append(c)
    reverse: dict = {}
    for a, b in edges:
        reverse.setdefault(b, []).append(a)
    dist = {k: 0 for k in seen if graph.is_goal(k)}
    queue = deque(dist)
    while queue:
        k = queue.popleft()
        for a in reverse.get(k, []):
            if a not in dist:
                dist[a] = dist[k] + 1
                queue.append(a)
    return dist


def test_weight_one_finds_the_fewest_moves_and_the_heuristic_is_admissible():
    wall = small_wall()
    graph = astar.Graph(wall, rc.load())
    start = astar.key(wall.start)
    dist = distances_to_goal(graph, start)
    exact = astar.solve(wall, always, weight=1.0)
    assert exact.reached_top and len(exact.plan.stances) - 1 == dist[start]
    assert all(graph.h(k) <= d for k, d in dist.items()), "h never overestimates"
    weighted = astar.solve(wall, always, weight=astar.WEIGHT)
    assert len(weighted.plan.stances) - 1 <= astar.WEIGHT * dist[start]


def shifted(wall: WallModel, di: int, dj: int) -> WallModel:
    """The same wall moved di columns right and dj rows up."""
    holds = []
    for h in wall.holds:
        i, j = walls.grid_index(h)
        holds.append(dataclasses.replace(h, position=walls.position(i + di, j + dj)))
    return dataclasses.replace(wall, holds=holds)


def first_move(wall: WallModel) -> tuple[Stance, Stance]:
    a = wall.start
    b = Stance(dict(a.contacts, left_hand="L5"))
    return a, b


def test_the_cache_key_is_the_moves_layout():
    wall = small_wall()
    a, b = first_move(wall)
    k = layout_key(wall, a, b)
    assert k is not None and k.startswith("left_hand|")
    assert layout_key(shifted(wall, 3, 5), a, b) == k, "the same move elsewhere on the wall"
    near = dataclasses.replace(wall, holds=[*wall.holds, Hold("X", walls.position(25, 30))])
    assert layout_key(near, a, b) != k, "a hold nearby is part of the layout"
    far = dataclasses.replace(wall, holds=[*wall.holds, Hold("X", walls.position(25, 85))])
    assert layout_key(far, a, b) == k, "a hold far away is not"
    assert layout_key(wall, a, Stance(dict(b.contacts, right_foot="R1"))) is None, "two limbs"


def test_the_cache_asks_once_per_layout_and_keeps_its_verdicts(tmp_path):
    calls = []

    def counted(w, a, b):
        calls.append(1)
        return Feasibility(False, 0.5, "torque")

    wall = small_wall()
    a, b = first_move(wall)
    cache = Cache(counted, tmp_path / "cache.json", version="v1")
    assert not cache(wall, a, b).feasible
    assert not cache(shifted(wall, -4, 2), a, b).feasible
    assert (len(calls), cache.hits, cache.misses) == (1, 1, 1)
    again = Cache(counted, tmp_path / "cache.json", version="v1")
    verdict = again(wall, a, b)
    assert (verdict.feasible, verdict.cost, len(calls)) == (False, 0.5, 1)
    assert "torque" in verdict.reason
    other = Cache(counted, tmp_path / "cache.json", version="v2")
    other(wall, a, b)
    assert len(calls) == 2, "another version starts empty"


def test_runs_in_parallel_keep_each_others_verdicts(tmp_path):
    wall = small_wall()
    a, b = first_move(wall)
    c = Stance(dict(a.contacts, right_hand="R5"))
    first = Cache(always, tmp_path / "cache.json", version="v1")
    second = Cache(always, tmp_path / "cache.json", version="v1")  # opened before first saves
    first(wall, a, b)
    second(wall, a, c)
    merged = Cache(always, tmp_path / "cache.json", version="v1")
    assert {layout_key(wall, a, b), layout_key(wall, a, c)} <= set(merged.entries)
