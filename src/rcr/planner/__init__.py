"""Planner (laptop): wall model -> full contact plan. See README.md.

Imports rcr.types only: never rcr.optimiser or rcr.controller. Feasibility is passed in.
Method (D2): reachability A* over stances, the optimiser's feasibility asked lazily (astar.py).
"""

from rcr.types import ContactPlan, FeasibilityFn, WallModel


def plan(wall: WallModel, feasible: FeasibilityFn) -> ContactPlan:
    """Sequence of stances from wall.start to the top. May call feasible(wall, a, b)."""
    from rcr.planner.astar import solve

    return solve(wall, feasible).plan
