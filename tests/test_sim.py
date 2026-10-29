"""The organisers' simulator (third_party/simulator, climbing_sim) on their walls, with our types:
watch the robot on a wall, and check that our side fits their sim.

    uv run mjpython tests/test_sim.py                  # their wall N1
    uv run mjpython tests/test_sim.py N2               # or N3, or a wall JSON file of theirs
    uv run mjpython tests/test_sim.py stances          # apply the STANCES below, in turn (on N1)
    uv run pytest tests/test_sim.py                    # the checks

macOS needs `mjpython` for the viewer (Linux: plain `python`). Needs `uv sync --group sim` and the
submodule (`git submodule update --init`); without them pytest skips this file. Their sim and walls
are used as they are: our side converts (their wall to ours: rcr.walls.from_organisers; their
observation and targets: rcr.robot.observation).

`run()` is the loop the other parts connect to: today the targets hold the start pose; later
`targets = controller.step(state)`.
"""

import sys
from pathlib import Path

import pytest

climbing_sim = pytest.importorskip("climbing_sim")

from climbing_sim import JOINTS, ClimbingSim, Hold, Wall  # noqa: E402

from rcr import io, walls  # noqa: E402
from rcr.robot import to_state, to_targets  # noqa: E402
from rcr.types import MICROBAN_JOINTS, JointTargets, Pose  # noqa: E402

EXAMPLES = Path(__file__).resolve().parent.parent / "data" / "examples"


def load_wall(name: str = "N1") -> Wall:
    """One of their walls: a bundled name (N1, N2, N3) or a wall JSON file of theirs."""
    return Wall.load(name)


def base_pose(sim: ClimbingSim) -> Pose:
    """The trunk, ground truth: what the cameras would give."""
    position, quat = sim.trunk_pose()
    return Pose(tuple(position), tuple(quat))


# --- Run: the robot in their sim, in the viewer ---------------------------------------------------


def run(wall: Wall) -> None:
    sim = ClimbingSim(wall)
    obs = sim.reset_hanging()  # hanging on wall.start, settled
    t0, last = obs["time"], list(sim.joint_positions())
    state = to_state(obs, JOINTS, t0, base_pose(sim))
    hold = JointTargets(positions=dict(state.joint_positions))
    viewer = sim.view()  # every step shows, in real time
    while viewer.is_running():
        targets = hold  # later: targets = controller.step(state)
        last = to_targets(targets, JOINTS, last)
        obs = sim.step(last)  # one 20 ms tick
        state = to_state(obs, JOINTS, t0, base_pose(sim))
    print(f"t = {state.time:.1f} s, hooked: {sim.hooked()}")


# --- Stances: define them here, apply them, look at them ----------------------------------------
# A stance here is a hold of N1 per limb, as their (col, row): their sim hangs the robot on four
# holds, so every limb has one. N1's rungs are 5 rows (75 mm) apart.

STANCES = {
    "start": {
        "right_hand": (29, 22), "left_hand": (18, 22), "right_foot": (26, 2), "left_foot": (21, 2)
    },
    "right hand up": {
        "right_hand": (29, 27), "left_hand": (18, 22), "right_foot": (26, 2), "left_foot": (21, 2)
    },
    "one rung higher": {
        "right_hand": (29, 27), "left_hand": (18, 27), "right_foot": (26, 7), "left_foot": (21, 7)
    },
}  # fmt: skip


def verify(sim: ClimbingSim, stance: dict) -> str:
    """Hang the robot on the stance: does every ball stay in its hold? "" if so, else the reason."""
    holds = {limb: Hold(*cell) for limb, cell in stance.items()}
    sim.reset_hanging(holds)
    wanted = {limb: sim.wall.holds.index(h) for limb, h in holds.items()}
    missed = [limb for limb, k in sim.hooked().items() if k != wanted[limb]]
    return f"not in their holds: {', '.join(missed)}" if missed else ""


def show_stances(seconds: float = 3.0) -> None:
    """Apply each stance in STANCES in the viewer, `seconds` each, round and round until closed."""
    sim = ClimbingSim(load_wall())
    viewer = sim.view()
    while viewer.is_running():
        for name, stance in STANCES.items():
            problem = verify(sim, stance)
            print(f"{name:<16} {'ok' if not problem else problem}")
            hold = sim.joint_positions()
            for _ in range(round(seconds / sim.control_dt)):
                if not viewer.is_running():
                    return
                sim.step(hold)


# --- Checks (pytest) ------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sim():
    return ClimbingSim(load_wall(), cameras=[])


def test_their_joints_are_ours():
    assert sorted(JOINTS) == sorted(MICROBAN_JOINTS)


@pytest.mark.parametrize("name", ["N1", "N2", "N3"])
def test_our_walls_are_theirs(name):
    """data/examples/N*.json (scripts/wall_generator.py) are their walls: each hold is theirs,
    named c<col>_r<row>, at their bolt midpoint, and the start is theirs."""
    theirs = Wall.load(name)
    ours = io.load_wall(EXAMPLES / f"{name}.json")
    assert ours == walls.from_organisers(theirs, level=int(name[1]))
    for hold, h in zip(ours.holds, theirs.holds, strict=True):
        assert hold.id == f"c{h.col}_r{h.row}"
        assert hold.position == pytest.approx(tuple(h.bolt_axis), abs=1e-9)
    assert ours.start.contacts == {k: f"c{h.col}_r{h.row}" for k, h in theirs.start.items()}


def test_the_robot_hangs_on_the_example_start(sim):
    obs = sim.reset_hanging()
    for _ in range(50):  # 1 s more, holding the pose
        obs = sim.step(sim.joint_positions())
    start = {limb: sim.wall.holds.index(h) for limb, h in sim.wall.start.items()}
    assert sim.hooked() == start
    state = to_state(obs, JOINTS, base=base_pose(sim))
    assert set(state.joint_positions) == set(MICROBAN_JOINTS)
    assert state.base_pose.position[1] < 0  # in front of the wall


def test_targets_follow_the_joint_targets_contract(sim):
    sim.reset_hanging()
    last = list(sim.joint_positions())
    target = to_targets(JointTargets(positions={"head": 0.3}), JOINTS, last)
    assert target[JOINTS.index("head")] == 0.3
    knee = JOINTS.index("left_knee")
    assert target[knee] == last[knee]  # absent: keeps its last target, as a Microban move
    sim.step(target)
    with pytest.raises(KeyError, match="unknown joints"):
        to_targets(JointTargets(positions={"tail": 0.0}), JOINTS, last)


def test_verify_accepts_the_start_and_rejects_a_stretch(sim):
    assert verify(sim, STANCES["start"]) == ""
    too_far = {**STANCES["start"], "right_hand": (29, 42), "left_hand": (18, 42)}  # hands 0.6 m up
    assert "not in their holds" in verify(sim, too_far)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["stances"]:
        show_stances()
    else:
        run(load_wall(*args[:1]))
