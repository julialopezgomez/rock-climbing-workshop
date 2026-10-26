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

from rcr import controller, io, link, optimiser, perception, planner
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
    """Check args/return against fn's annotations, call it, fall back to `example` if a stub,
    or if this machine lacks what the stage needs (an optional dependency group or Microban)."""
    hints = typing.get_type_hints(fn)
    returns = hints.pop("return", None)
    for (param, hint), arg in zip(hints.items(), args, strict=True):
        if isinstance(hint, type):
            assert isinstance(arg, hint), f"{name}: {param} should be {hint.__name__}"
    assert returns == expected, f"{name}: returns {_name(returns)}, expected {_name(expected)}"
    why = ""
    try:
        out = fn(*args)
        status = "ok"
    except NotImplementedError:
        out, status = example, "stub"
    except (ImportError, FileNotFoundError) as e:
        out, status, why = example, "skip", f" ({e})"
    assert isinstance(out, expected), f"{name}: returned {type(out).__name__}"
    print(f"  {name:<32} -> {_name(expected):<18} {status}{why}")
    return out, status


def _sim(wall: WallModel):
    """The sim backend, or None if this machine cannot build it (no --group sim, no Microban)."""
    try:
        from rcr import sim

        return sim.Sim(wall)
    except (ImportError, FileNotFoundError) as e:
        print(f"  {'sim.Sim':<32} -> {'RobotState':<18} skip ({e})")
        return None


def check() -> int:
    wall_ex = io.load_wall(EXAMPLES / "wall_hard.json")
    plan_ex = io.load_contact_plan(EXAMPLES / "contact_plan_hard.json")
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
    backend = _sim(wall)
    if backend is None:
        results.append("skip")
    else:
        state, s = stage("sim.Sim.reset", backend.reset, RobotState, state_ex, wall.start)
        results.append(s)
        _, s = stage("sim.Sim.step", backend.step, RobotState, state, targets)
        results.append(s)

    print(
        f"check passed: {results.count('ok')} implemented, {results.count('stub')} stubs, "
        f"{results.count('skip')} skipped"
    )
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
