"""Planner (laptop): wall model -> full contact plan. See README.md.

Imports rcr.types only: never rcr.optimiser or rcr.controller. Feasibility is passed in.
"""

from rcr.planner.astar import feasible_ik_contact, solve
from rcr.types import ContactPlan, FeasibilityFn, Stance, WallModel

__all__ = ["feasible_ik_contact", "plan", "replan"]


def plan(wall: WallModel, feasible: FeasibilityFn, HEURISTIC_WEIGHT: float = 1.0) -> ContactPlan:
    """Sequence of stances from wall.start to the top. May call feasible(wall, a, b)."""
    return solve(wall, feasible, HEURISTIC_WEIGHT)

def replan(
    wall: WallModel,
    feasible: FeasibilityFn,
    start: Stance,
    failed_moves=(),
    failed_stances=(),
    HEURISTIC_WEIGHT: float = 1.0,
) -> ContactPlan:
    """Sequence of stances from `start` (a limb may be on None: it is placed first) to the top,
    never through a failed move (a, b) or a failed stance."""
    return solve(wall, feasible, HEURISTIC_WEIGHT, start, failed_moves, failed_stances)
