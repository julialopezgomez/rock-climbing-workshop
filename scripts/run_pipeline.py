"""Run the pipeline.

    uv run scripts/run_pipeline.py --check

--check takes the organisers' wall N1 (data/examples/N1.json, from their simulator) and the link's
sample trajectory (data/examples/trajectory.json), then passes them through every stage's signature
in pipeline order (the optimiser on the plan's first move only, to keep it quick). A stage that is
still a stub (raises NotImplementedError), or that this machine cannot run (a dependency group
missing), is reported as "stub" or "skip" and the example output is used in its place: for the
plan, the start stance alone (there is no example contact plan yet).

The sim is the organisers' (third_party/simulator, climbing_sim), on their wall of the same name;
their observation and targets to and from our types by rcr.robot.
Later: --backend sim|robot runs the controller in a closed loop against their sim or the robot.
"""

import argparse
import sys
import typing
from pathlib import Path
from types import NoneType

from rcr import controller, io, link, optimiser, perception, planner, robot
from rcr.types import (
    MICROBAN_JOINTS,
    ContactPlan,
    JointTargets,
    RobotState,
    Trajectory,
    WallModel,
)

EXAMPLES = Path(__file__).resolve().parent.parent / "data" / "examples"


def check_examples(wall: WallModel, traj: Trajectory) -> None:
    hold_ids = {h.id for h in wall.holds}
    for limb, hold in wall.start.contacts.items():
        assert hold is None or hold in hold_ids, f"start: {limb} on unknown hold {hold}"
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
    given = list(hints.items())[: len(args)]  # the rest have defaults
    for (param, hint), arg in zip(given, args, strict=True):
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


class OrganisersSim:
    """Their sim on their wall of this name, in our types: hang() -> RobotState, step(targets)."""

    def __init__(self, wall: WallModel):
        from climbing_sim import JOINTS, ClimbingSim, Wall

        self.joints = JOINTS
        self.sim = ClimbingSim(Wall.load(wall.id), cameras=[])

    def hang(self) -> RobotState:
        self.last = list(self.sim.joint_positions())
        return robot.to_state(self.sim.reset_hanging(), self.joints)

    def step(self, targets: JointTargets) -> RobotState:
        self.last = robot.to_targets(targets, self.joints, self.last)
        return robot.to_state(self.sim.step(self.last), self.joints)


def _sim(wall: WallModel) -> OrganisersSim | None:
    """Their sim, or None if this machine cannot build it (no --group sim, no submodule)."""
    try:
        return OrganisersSim(wall)
    except (ImportError, FileNotFoundError) as e:
        print(f"  {'climbing_sim.ClimbingSim':<32} -> {'RobotState':<18} skip ({e})")
        return None


def check() -> int:
    wall_ex = io.load_wall(EXAMPLES / "N1.json")
    plan_ex = ContactPlan(wall_ex.id, [wall_ex.start])  # no example plan yet: the start alone
    traj_ex = io.load_trajectory(EXAMPLES / "trajectory.json")
    check_examples(wall_ex, traj_ex)
    print(f"examples ok: {wall_ex.id}, {len(wall_ex.holds)} holds; {len(traj_ex.times)} samples")

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
    # The plan's first move only: a check of the stage, not the climb (the whole example takes an
    # hour on an empty move cache; tests/test_optimiser_v0.py PLAN solves and plays it) [Ours]
    if len(plan.stances) < 2:
        print(f"  {'optimiser.optimise':<32} -> {'Trajectory':<18} skip (a plan with no move)")
        traj, s = traj_ex, "skip"
    else:
        first_move = ContactPlan(plan.wall_id, plan.stances[:2])
        traj, s = stage(
            "optimiser.optimise", optimiser.optimise, Trajectory, traj_ex, wall, first_move
        )
    results.append(s)
    # Both ends of the link on this machine, over loopback: the real code, without the Pi
    robot_end = link.RobotLink(port=0, pose_port=0, host="127.0.0.1")
    laptop = link.LaptopLink("127.0.0.1", robot_end.port)
    try:
        _, s = stage(
            "link.LaptopLink.send_trajectory", laptop.send_trajectory, NoneType, None, traj
        )
        results.append(s)

        print("robot (or sim):")
        received, s = stage(
            "link.RobotLink.trajectory", robot_end.trajectory, Trajectory | None, traj
        )
        results.append(s)
    finally:
        laptop.close()
        robot_end.close()
    ctrl = controller.Controller(received)
    targets_ex = JointTargets(positions=first)
    targets, s = stage("controller.Controller.step", ctrl.step, JointTargets, targets_ex, state_ex)
    results.append(s)
    backend = _sim(wall)
    if backend is None:
        results.append("skip")
    else:
        state, s = stage("sim (theirs): hang", backend.hang, RobotState, state_ex)
        results.append(s)
        _, s = stage("sim (theirs): step", backend.step, RobotState, state, targets)
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
