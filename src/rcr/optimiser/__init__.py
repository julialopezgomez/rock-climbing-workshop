"""Optimiser (laptop): contact plan -> full trajectory; owns feasibility. See README.md.

Method (D3): a kinematic motion. Per move, the moving ball follows a polynomial path between its
sockets and the whole body follows by hierarchical IK with an error correction (ik.py). The CasADi
optimiser before it is in v0/, for reference.
"""

from rcr.types import (
    LIMBS,
    MICROBAN_JOINTS,
    ContactPlan,
    Feasibility,
    Stance,
    Trajectory,
    WallModel,
)


def optimise(wall: WallModel, plan: ContactPlan) -> Trajectory:
    """The full joint trajectory the robot executes for this plan: the start pose of its first
    stance, then each move's samples, every ik.DT. Kinematic: positions are the IK's angles."""
    from rcr.optimiser import ik

    validate_plan(wall, plan)
    qs = [ik.start_pose(wall, plan.stances[0])]
    for a, b in zip(plan.stances, plan.stances[1:], strict=False):
        qs += ik.track_move(wall, a, b, qs[-1])[0]
    joints = [ik.model.idx_qs[ik.model.getJointId(name)] for name in MICROBAN_JOINTS]
    positions = [[float(q[j]) for j in joints] for q in qs]
    return Trajectory(
        joint_names=list(MICROBAN_JOINTS),
        times=[round(k * ik.DT, 9) for k in range(len(qs))],
        positions=positions,
        planned_positions=[row.copy() for row in positions],
        # Pinocchio's quaternion is (x, y, z, w); ours (w, x, y, z)
        trunk_poses=[[float(v) for v in q[[0, 1, 2, 6, 3, 4, 5]]] for q in qs],
    )


def validate_plan(wall: WallModel, plan: ContactPlan) -> None:
    """ValueError unless the plan is for this wall, every stance is on its holds and exactly one
    limb changes between consecutive stances."""
    holds = {h.id for h in wall.holds}
    if plan.wall_id != wall.id:
        raise ValueError(f"the plan is for wall {plan.wall_id!r}, not {wall.id!r}")
    for n, s in enumerate(plan.stances):
        if unknown := {s.contacts[limb] for limb in LIMBS} - holds:
            raise ValueError(f"stance {n}: holds not on the wall: {sorted(unknown)}")
    for n, (a, b) in enumerate(zip(plan.stances, plan.stances[1:], strict=False)):
        moved = [limb for limb in LIMBS if a.contacts[limb] != b.contacts[limb]]
        if len(moved) != 1:
            raise ValueError(f"stances {n} -> {n + 1}: {len(moved)} limbs move, not 1")


def feasibility(wall: WallModel, a: Stance, b: Stance) -> Feasibility:
    """Can the robot go from stance a to stance b? Passed to the planner as FeasibilityFn."""
    raise NotImplementedError
