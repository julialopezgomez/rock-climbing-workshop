"""Planner (laptop): wall model -> full contact plan. See README.md.

Imports rcr.types only: never rcr.optimiser or rcr.controller. Feasibility is passed in.
"""

from rcr.planner.astar import feasible, solve
from rcr.types import ContactPlan, FeasibilityFn, WallModel

__all__ = ["feasible", "plan"]



def plan(wall: WallModel, feasible: FeasibilityFn, HEURISTIC_WEIGHT: float = 1.0) -> ContactPlan:
    """Sequence of stances from wall.start to the top. May call feasible(wall, a, b)."""
    return solve(wall, feasible, HEURISTIC_WEIGHT)