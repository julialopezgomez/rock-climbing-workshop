"""The sim in one place: watch the robot on the wall, and check that the sim works.

    uv run mjpython tests/test_sim.py               # robot on the example L1 wall, until closed
    uv run mjpython tests/test_sim.py organisers    # or l1 | l2 | l3 | organisers | a wall/1 file
    uv run mjpython tests/test_sim.py stances       # apply the STANCES below, one after another
    uv run pytest tests/test_sim.py                 # the checks

macOS needs `mjpython` for the viewer (Linux: plain `python`). Needs `uv sync --group sim` and
Microban (src/rcr/sim/README.md); without them pytest skips this file.

`run()` is the loop the other parts connect to: today the targets just hold the start pose; later
`targets = controller.step(state)`, and `sim.expect(stance)` when the plan moves on.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from rcr import io  # noqa: E402
from rcr.sim import Sim, levels  # noqa: E402
from rcr.sim.scene import MICROBAN_XML  # noqa: E402
from rcr.sim.start import NEUTRAL_POSE  # noqa: E402
from rcr.types import MICROBAN_JOINTS, JointTargets, Stance, WallModel  # noqa: E402

EXAMPLE_WALL = Path(__file__).resolve().parent.parent / "data" / "examples" / "wall_l1.json"
WALLS = {"l1": levels.level1, "l2": levels.level2, "l3": levels.level3}
WALLS["organisers"] = levels.organisers_wall


def load_wall(name: str | None = None) -> WallModel:
    if name in WALLS:
        return WALLS[name]()
    return io.load_wall(Path(name) if name else EXAMPLE_WALL)


# --- Run: the robot in the sim, in the viewer -----------------------------------------------------


def run(wall: WallModel) -> None:
    sim = Sim(wall)
    state = sim.reset()  # hanging on wall.start, time = 0
    hold = JointTargets(positions=dict(state.joint_positions))

    import mujoco.viewer

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        while viewer.is_running():
            tick = time.perf_counter()
            targets = hold  # later: targets = controller.step(state)
            state = sim.step(targets)  # one 20 ms tick
            viewer.sync()
            time.sleep(max(0.0, 0.02 - (time.perf_counter() - tick)))  # real time

    print(f"t = {state.time:.1f} s, gripping: {', '.join(sim.attachment.attached_limbs())}")
    print(f"score: height {sim.score.height:.3f} m, top reached at {sim.score.top_time}")


# --- Stances: define them here, apply them, look at them ----------------------------------------
# A Stance (rcr.types) is a hold id or None (free) per limb. These use the example L1 wall's holds:
# L0..L3 left column, R0..R3 right column, rows at z = 0.16 / 0.31 / 0.46 / 0.61 m.

STANCES = {
    "start": Stance({"right_hand": "R2", "left_hand": "L2", "right_foot": "R0", "left_foot": "L0"}),
    "right hand free": Stance(
        {"right_hand": None, "left_hand": "L2", "right_foot": "R0", "left_foot": "L0"}
    ),
    "right hand up": Stance(
        {"right_hand": "R3", "left_hand": "L2", "right_foot": "R0", "left_foot": "L0"}
    ),
    "one row higher": Stance(
        {"right_hand": "R3", "left_hand": "L3", "right_foot": "R1", "left_foot": "L1"}
    ),
}


def verify(sim: Sim, stance: Stance) -> str:
    """Apply the stance: can every ball reach its cup without hitting the wall or itself?

    Returns "" if it can, else the reason.
    """
    try:
        sim.reset(stance)  # solves the joint angles for the stance, grips, settles
    except ValueError as e:
        return str(e)
    if sim.wall_penetration() > 0.001:
        return f"hits the wall by {sim.wall_penetration() * 1e3:.0f} mm"
    if sim.self_penetration() > 0.001:
        return f"hits itself by {sim.self_penetration() * 1e3:.0f} mm"
    return ""


def show_stances(wall: WallModel, seconds: float = 3.0) -> None:
    """Apply each stance in STANCES in the viewer, `seconds` each, round and round until closed."""
    sim = Sim(wall)

    import mujoco.viewer

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        while viewer.is_running():
            for name, stance in STANCES.items():
                problem = verify(sim, stance)
                print(f"{name:<16} {'ok' if not problem else problem}")
                if problem.startswith("stance not reachable") or not viewer.is_running():
                    continue  # nothing to show
                state = sim.state()
                hold = JointTargets(positions=dict(state.joint_positions))
                while viewer.is_running() and state.time < seconds:
                    tick = time.perf_counter()
                    state = sim.step(hold)
                    viewer.sync()
                    time.sleep(max(0.0, 0.02 - (time.perf_counter() - tick)))


# --- Checks (pytest) ------------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(not MICROBAN_XML.exists(), reason=f"no Microban at {MICROBAN_XML}")


def test_grid_and_hold_follow_the_rules():
    from rcr.sim.geometry import BOLT_SPACING, N_COLS, N_ROWS, CupHold, hold_position

    assert (N_COLS, N_ROWS) == (52, 93)  # [Rules v1.1 Figure 2]
    assert BOLT_SPACING == pytest.approx(0.030)  # [Rules v1.1 Figure 3]
    assert hold_position(20, 10) == pytest.approx((0.3175, 0.0, 0.16))  # the bolt midpoint
    for i in (0, N_COLS - 1):
        with pytest.raises(ValueError, match="bolt off the grid"):
            hold_position(i, 10)
    hold = CupHold()  # the organisers' hold.stl
    hold.validate()  # inside the Ø0.060 x 0.100 m envelope
    v = hold.vertices()
    assert -v[:, 1].min() == pytest.approx(0.043)  # depth [Rules Figure 3]
    assert v[:, 1].max() == pytest.approx(0.0)  # back face on the wall
    socket = np.linalg.norm(v - hold.mesh_pos, axis=1)  # distances from the socket centre
    assert np.isclose(socket, hold.socket_radius, atol=1e-4).sum() > 1000  # the spherical socket
    # a ball as big as the socket sits on its centre: 14 mm down, 26 mm out [Rules Figure 3]
    assert hold.grasp_point((0.3, 0.0, 0.5), hold.socket_radius) == pytest.approx(
        (0.3, -0.026, 0.486)
    )


@pytest.fixture
def sim(tmp_path):
    return Sim(load_wall(), out=tmp_path / "scene.xml")


def test_reset_hangs_the_robot_on_the_start_stance(sim):
    state = sim.reset()
    assert state.time == 0.0
    assert set(state.joint_positions) == set(MICROBAN_JOINTS)
    for limb, hold in sim.wall.start.contacts.items():
        assert sim.attachment.held_hold(limb) == hold
    right, left = (sim.site_pos(f"rob_{side}_hand_tip") for side in ("right", "left"))
    assert right[0] > left[0]  # facing the wall: right hand at larger x
    assert state.base_pose.position[1] < 0  # in front of the wall


def test_holding_the_start_pose_is_stable(sim):
    state = sim.reset()
    before = np.array(state.base_pose.position)
    hold = JointTargets(positions=dict(state.joint_positions))
    for _ in range(100):  # 2 s
        state = sim.step(hold)
    assert state.time == pytest.approx(2.0)
    assert len(sim.attachment.attached_limbs()) == 4
    assert np.linalg.norm(np.array(state.base_pose.position) - before) < 0.005
    assert sim.wall_penetration() < 0.001


def test_step_follows_the_joint_targets_contract(sim):
    sim.reset()
    sim.step(JointTargets(positions={"head": 0.3}))
    assert sim._target[sim._index["head"]] == 0.3
    assert sim._target[sim._index["left_knee"]] == NEUTRAL_POSE["left_knee"]  # absent: neutral
    with pytest.raises(KeyError, match="unknown joints"):
        sim.step(JointTargets(positions={"tail": 0.0}))


def test_expect_releases_the_limb_that_moves_on(sim):
    sim.reset()
    sim.expect(Stance(contacts={**sim.wall.start.contacts, "right_hand": "R3"}))
    assert not sim.attachment.is_attached("right_hand")
    assert sim.attachment.is_attached("left_hand")
    assert sim.score.height == pytest.approx(0.46)  # the start hands, R2/L2
    assert sim.score.top_time is None


@pytest.mark.parametrize("name", ["l1", "organisers"])
def test_other_walls_have_a_start_pose(name, tmp_path):
    Sim(load_wall(name), out=tmp_path / "scene.xml").reset()


def test_verify_accepts_a_reachable_stance(sim):
    assert verify(sim, sim.wall.start) == ""
    assert verify(sim, Stance({**sim.wall.start.contacts, "right_hand": None})) == ""  # 3 limbs


def test_verify_rejects_what_the_robot_cannot_do(sim):
    too_far = {"right_hand": "R3", "left_hand": "L3", "right_foot": "R0", "left_foot": "L0"}
    assert "not reachable" in verify(sim, Stance(too_far))  # a 0.45 m stretch
    crossed = {"right_hand": None, "left_hand": "L2", "right_foot": "L0", "left_foot": "R0"}
    assert "hits itself" in verify(sim, Stance(crossed))


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["stances"]:
        show_stances(load_wall(args[1] if len(args) > 1 else None))
    else:
        run(load_wall(args[0] if args else None))
