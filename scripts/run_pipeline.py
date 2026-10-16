"""Run the pipeline.

    uv run scripts/run_pipeline.py --check

--check loads data/examples/, checks they are consistent, then passes them through every stage's
signature in pipeline order. A stage that is still a stub (raises NotImplementedError) is reported
as "stub" and the example output is used in its place, so the check exercises each v0 as it lands.

Later: --backend sim|robot runs the controller in a closed loop against rcr.sim.Sim or the robot.
"""

import argparse
import sys
import typing
from pathlib import Path
from types import NoneType

from rcr import controller, io, link, optimiser, perception, planner, sim
from rcr.types import (
    MICROBAN_JOINTS,
    ContactPlan,
    JointTargets,
    RobotState,
    Trajectory,
    WallModel,
)

EXAMPLES = Path(__file__).resolve().parent.parent / "data" / "examples"


def check_examples(wall: WallModel, plan: ContactPlan, traj: Trajectory) -> None:
    hold_ids = {h.id for h in wall.holds}
    assert plan.wall_id == wall.id, "contact plan is for another wall"
    assert plan.stances[0] == wall.start, "contact plan must start at wall.start"
    for i, stance in enumerate(plan.stances):
        for limb, hold in stance.contacts.items():
            assert hold is None or hold in hold_ids, f"stance {i}: {limb} on unknown hold {hold}"
    assert set(traj.joint_names) <= set(MICROBAN_JOINTS), "trajectory has unknown joints"
    assert all(a < b for a, b in zip(traj.times, traj.times[1:], strict=False)), (
        "times must increase"
    )
    assert len(traj.positions) == len(traj.times), "one row of positions per time"
    assert all(len(row) == len(traj.joint_names) for row in traj.positions), "row width"


def _name(t) -> str:
    return "None" if t is NoneType else getattr(t, "__name__", str(t).replace("rcr.types.", ""))


def stage(name: str, fn, expected, example, *args):
    """Check args/return against fn's annotations, call it, fall back to `example` if a stub."""
    hints = typing.get_type_hints(fn)
    returns = hints.pop("return", None)
    for (param, hint), arg in zip(hints.items(), args, strict=True):
        if isinstance(hint, type):
            assert isinstance(arg, hint), f"{name}: {param} should be {hint.__name__}"
    assert returns == expected, f"{name}: returns {_name(returns)}, expected {_name(expected)}"
    try:
        out = fn(*args)
        status = "ok"
    except NotImplementedError:
        out, status = example, "stub"
    assert isinstance(out, expected), f"{name}: returned {type(out).__name__}"
    print(f"  {name:<32} -> {_name(expected):<18} {status}")
    return out, status


def check() -> int:
    wall_ex = io.load_wall(EXAMPLES / "wall_l1.json")
    plan_ex = io.load_contact_plan(EXAMPLES / "contact_plan.json")
    traj_ex = io.load_trajectory(EXAMPLES / "trajectory.json")
    check_examples(wall_ex, plan_ex, traj_ex)
    print(
        f"examples ok: {len(wall_ex.holds)} holds, {len(plan_ex.stances)} stances, "
        f"{len(traj_ex.times)} trajectory samples"
    )

    first = dict(zip(traj_ex.joint_names, traj_ex.positions[0], strict=True))
    state_ex = RobotState(
        time=0.0,
        joint_positions=first,
        joint_velocities=dict.fromkeys(first, 0.0),
        imu_quat=(1.0, 0.0, 0.0, 0.0),
        gyro=(0.0, 0.0, 0.0),
    )

    print("laptop:")
    results = []
    wall, s = stage("perception.wall_model", perception.wall_model, WallModel, wall_ex, [])
    results.append(s)
    plan, s = stage("planner.plan", planner.plan, ContactPlan, plan_ex, wall, optimiser.feasibility)
    results.append(s)
    traj, s = stage("optimiser.optimise", optimiser.optimise, Trajectory, traj_ex, wall, plan)
    results.append(s)
    laptop = link.LaptopLink("microban")
    _, s = stage("link.LaptopLink.send_trajectory", laptop.send_trajectory, NoneType, None, traj)
    results.append(s)

    print("robot (or sim):")
    received, s = stage(
        "link.RobotLink.trajectory", link.RobotLink().trajectory, Trajectory | None, traj
    )
    results.append(s)
    ctrl = controller.Controller(received)
    targets_ex = JointTargets(positions=first)
    targets, s = stage("controller.Controller.step", ctrl.step, JointTargets, targets_ex, state_ex)
    results.append(s)
    backend = sim.Sim(wall)
    _, s = stage("sim.Sim.step", backend.step, RobotState, state_ex, targets)
    results.append(s)

    print(f"check passed: {results.count('ok')} implemented, {results.count('stub')} stubs")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="run the examples through every stage")
    args = parser.parse_args()
    if not args.check:
        parser.error("only --check exists so far")
    return check()


if __name__ == "__main__":
    sys.exit(main())
