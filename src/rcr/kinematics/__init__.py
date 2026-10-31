"""Optimiser (laptop): contact plan -> full trajectory; owns feasibility. See README.md.

Method (D3): a kinematic motion. Per move, the moving ball follows a polynomial path between its
sockets and the whole body follows by hierarchical IK with an error correction (ik.py). The CasADi
optimiser is rcr.optimiser.
"""

import numpy as np

from rcr.kinematics.ik import feasible_ik_traj as feasible_ik_traj
from rcr.types import (
    LIMBS,
    ContactPlan,
    Trajectory,
    WallModel,
)


def optimise(wall: WallModel, plan: ContactPlan) -> Trajectory:
    """The full joint trajectory the robot executes for this plan: the start pose of its first
    stance, then each move's samples, every ik.DT. Kinematic: positions are the IK's angles."""
    from rcr.kinematics import ik

    validate_plan(wall, plan)
    return ik.to_trajectory(ik.generate_trajectory(wall, plan))


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


def convert_trajectory_to_npz(trajectory: Trajectory, filename: str) -> None:
    """Save the trajectory to a .npz with one array, `q` (N, 26), a row every ik.DT (0.02 s,
    50 Hz) from t = 0: the free flyer (trunk x, y, z [m], qw, qx, qy, qz, wall frame) then the 19
    joints [rad] in MICROBAN_JOINTS order [Ours]."""
    q = np.hstack([np.array(trajectory.trunk_poses), np.array(trajectory.positions)])
    np.savez(filename, q=q)
