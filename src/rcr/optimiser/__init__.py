"""Optimiser (laptop): contact plan -> full trajectory; owns feasibility. See README.md."""

from rcr.types import ContactPlan, Feasibility, Stance, Trajectory, WallModel


def optimise(wall: WallModel, plan: ContactPlan) -> Trajectory:
    """The full joint trajectory the robot executes for this plan."""
    raise NotImplementedError


def feasibility(wall: WallModel, a: Stance, b: Stance) -> Feasibility:
    """Can the robot go from stance a to stance b? Passed to the planner as FeasibilityFn."""
    raise NotImplementedError
