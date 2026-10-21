"""Optimiser tests, and a viewer to watch the motion it generates.

    uv sync --group optimiser --group sim
    export MICROBAN_URDF=/path/to/microban/src/model/urdf/robot.urdf
    uv run pytest tests/test_optimiser.py            # the checks
    uv run mjpython tests/test_optimiser.py [rows]   # solve a climb of `rows` rows (default 1) on
                                                     # the test ladder and play it in the sim scene

macOS needs `mjpython` for the viewer (Linux: plain `python`). The viewer replays the optimiser's
poses exactly (kinematics only, no physics), at the planned times, until the window is closed.
Without the optimiser group or MICROBAN_URDF, pytest skips this file.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ca = pytest.importorskip("casadi")
if not os.environ.get("MICROBAN_URDF"):
    pytest.skip("MICROBAN_URDF is not set", allow_module_level=True)

from rcr import io  # noqa: E402
from rcr.optimiser import SERVO_KP, optimise, solve_plan, validate_plan  # noqa: E402
from rcr.optimiser import collision as col  # noqa: E402
from rcr.optimiser import robot_model as rm  # noqa: E402
from rcr.optimiser import transition as tr  # noqa: E402
from rcr.types import LIMBS, MICROBAN_JOINTS, ContactPlan, Hold, Stance, WallModel  # noqa: E402

EXAMPLES = Path(__file__).resolve().parent.parent / "data" / "examples"

# The example wall's rows are 0.15 m apart: its feet can move up a row, but no hand can within
# 70 % of the servo torque (it needs 100 %). This ladder can be climbed: rows 75 mm apart, the
# hands starting 300 mm above the feet.
ROW_STEP, HANDS_ABOVE_FEET = 0.075, 4
DENSE = col.Clearance(step=0.0005)  # independent re-checks: 4x denser than the solver's


@pytest.fixture(scope="module")
def model():
    return rm.load_robot_model()


def test_ladder() -> WallModel:
    holds = [
        Hold(f"{side}{j}", (x, 0.0, 0.16 + ROW_STEP * j), "unknown")
        for j in range(10)
        for side, x in (("L", 0.3175), ("R", 0.4825))
    ]
    start = {"right_hand": "R4", "left_hand": "L4", "right_foot": "R0", "left_foot": "L0"}
    return WallModel("test_ladder", 1, holds, Stance(start))


test_ladder.__test__ = False  # a helper, not a test


@pytest.fixture(scope="module")
def wall():
    return test_ladder()


def climb_one_row(wall: WallModel, rows: int = 1) -> ContactPlan:
    """Right foot, left foot, right hand, left hand: each up one row, `rows` times."""
    contacts, stances = dict(wall.start.contacts), [Stance(dict(wall.start.contacts))]
    for _ in range(rows):
        for limb in ("right_foot", "left_foot", "right_hand", "left_hand"):
            contacts[limb] = f"{contacts[limb][0]}{int(contacts[limb][1:]) + 1}"
            stances.append(Stance(dict(contacts)))
    return ContactPlan(wall.id, stances)


@pytest.fixture(scope="module")
def results(wall):
    return solve_plan(wall, climb_one_row(wall))


def random_configuration(model, rng):
    q = np.zeros(rm.NQ)
    q[0:3] = rng.normal(size=3) * 0.2
    quat = rng.normal(size=4)
    q[3:7] = quat / np.linalg.norm(quat)
    q[7:] = rng.uniform(model.lower, model.upper)
    return q


# --- The robot model ----------------------------------------------------------------------------


def test_kinematics_match_pinocchio(model):
    """Our CasADi kinematics (written from the URDF) against Pinocchio on the same URDF, with
    the same hooks added as frames and point masses. Pinocchio's quaternion is (x, y, z, w)."""
    pin = pytest.importorskip("pinocchio")
    pm = pin.buildModelFromUrdf(os.environ["MICROBAN_URDF"], pin.JointModelFreeFlyer())
    balls = {}
    for limb, hook in rm.HOOKS.items():
        link = pm.frames[pm.getFrameId(hook.link, pin.FrameType.BODY)]
        placement = link.placement * pin.SE3(np.eye(3), rm.hook_ball_centre(hook))
        balls[limb] = pm.addFrame(
            pin.Frame(f"{limb}_ball", link.parentJoint, placement, pin.FrameType.OP_FRAME)
        )
        for mass, point in rm.hook_masses(hook):
            pm.inertias[link.parentJoint] += pin.Inertia(
                mass, link.placement * point, np.zeros((3, 3))
            )
    pd = pm.createData()
    qi = [pm.joints[pm.getJointId(n)].idx_q for n in MICROBAN_JOINTS]
    vi = [pm.joints[pm.getJointId(n)].idx_v for n in MICROBAN_JOINTS]

    def to_pinocchio(q):
        qp = pin.neutral(pm)
        qp[0:3], qp[3:7] = q[0:3], [q[4], q[5], q[6], q[3]]
        qp[qi] = q[7:]
        return qp

    rng = np.random.default_rng(0)
    for _ in range(10):
        q = random_configuration(model, rng)
        qp = to_pinocchio(q)
        pin.forwardKinematics(pm, pd, qp)
        pin.updateFramePlacements(pm, pd)
        pin.computeJointJacobians(pm, pd, qp)
        for i, limb in enumerate(LIMBS):
            ball = pd.oMf[balls[limb]].translation
            assert np.allclose(np.array(model.limb_positions(q))[:, i], ball, atol=1e-9)
            jac = pin.getFrameJacobian(pm, pd, balls[limb], pin.LOCAL_WORLD_ALIGNED)[:3, vi]
            assert np.allclose(np.array(model.limb_jacobian(q, limb)), jac, atol=1e-9)
        assert np.allclose(np.array(model.com(q)).ravel(), pin.centerOfMass(pm, pd, qp), atol=1e-9)
        gravity = pin.computeGeneralizedGravity(pm, pd, qp)[vi]
        assert np.allclose(np.array(model.gravity_torques(q)).ravel(), gravity, atol=1e-9)
        dq = rng.normal(size=rm.NV) * 0.3
        ours = np.array(model.integrate(q, dq)).ravel()
        dq_pin = np.zeros(pm.nv)
        dq_pin[:6], dq_pin[vi] = dq[:6], dq[6:]
        theirs = pin.integrate(pm, qp, dq_pin)
        assert np.allclose(ours[0:3], theirs[0:3], atol=1e-12)
        assert np.allclose(ours[3:7], [theirs[6], *theirs[3:6]], atol=1e-12)
    assert model.mass == pytest.approx(sum(i.mass for i in pm.inertias))


def test_hooks_and_contact_point_match_the_sim(model):
    """The hooks are ours (Microban has no climbing limbs yet): the optimiser and the sim must
    put the balls in the same place, and agree on where a ball sits on a hold."""
    pytest.importorskip("mujoco")
    from rcr.sim import ROBOT_PREFIX, Sim
    from rcr.sim.scene import MICROBAN_XML

    if not MICROBAN_XML.exists():
        pytest.skip(f"no Microban MJCF at {MICROBAN_XML}")
    wall = io.load_wall(EXAMPLES / "wall_l1.json")
    sim = Sim(wall, out=None)
    rng = np.random.default_rng(1)
    for _ in range(5):
        q = random_configuration(model, rng)
        sim.place(dict(zip(MICROBAN_JOINTS, q[7:], strict=True)), q[0:3], q[3:7])
        ours = np.array(model.limb_positions(q))
        for i, limb in enumerate(LIMBS):
            assert np.allclose(ours[:, i], sim.site_pos(f"{ROBOT_PREFIX}{limb}_tip"), atol=1e-5)
    cfg = tr.TransitionConfig()
    for limb, point in sim.grasp_targets(wall.start).items():
        assert np.allclose(point, tr.contact_point(wall, wall.start.contacts[limb], cfg))


def test_collision_points_match_microbans_mjcf(model):
    mujoco = pytest.importorskip("mujoco")
    mm = mujoco.MjModel.from_xml_path(str(rm.mjcf_path()))
    md = mujoco.MjData(mm)
    shapes = [g for g in range(mm.ngeom) if mm.geom_group[g] == 3]  # Microban's collision group
    q = random_configuration(model, np.random.default_rng(2))
    md.qpos[:7] = q[:7]  # MuJoCo's free joint is also (position, w, x, y, z)
    for i, name in enumerate(MICROBAN_JOINTS):
        md.qpos[mm.jnt_qposadr[mm.joint(name).id]] = q[7 + i]
    mujoco.mj_kinematics(mm, md)
    ours = np.array(model.body_points(q)).T
    assert len(ours) == 8 * len(shapes)
    for g in shapes:  # every box's farthest corner along each axis is one of our points
        if mm.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            rotation = md.geom_xmat[g].reshape(3, 3)
            reach = md.geom_xpos[g] + np.abs(rotation) @ mm.geom_size[g]
            for axis in range(3):
                assert np.min(np.abs(ours[:, axis] - reach[axis])) < 1e-5  # 6-digit files


# --- Collision geometry ------------------------------------------------------------------------


def test_the_hold_mesh_is_closed_and_outward():
    vertices, triangles = col.read_hold_mesh()
    edges = np.sort(
        np.vstack([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), 1
    )
    _, uses = np.unique(edges, axis=0, return_counts=True)
    assert (uses == 2).all()  # watertight: every edge shared by exactly two triangles
    a, b, c = (vertices[triangles[:, i]] for i in range(3))
    assert np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6 > 0  # normals point out


def test_hold_distance_is_exact_outside_and_signed_inside():
    coal = pytest.importorskip("coal")
    hold = col.hold_geometry()
    ball = next(s for s in col.robot_shapes() if s.name == "right_hand_ball")

    def distance(centre):
        pose = coal.Transform3s(np.eye(3), np.array(centre, dtype=float))
        return col.hold_distance(ball, pose, hold, np.zeros(3))

    assert distance([0, 0, 0]).value == pytest.approx(0.00266, abs=5e-5)  # 12.66 - 10 mm
    assert abs(distance([0, 0, -0.0027]).value) < col.TOUCHING  # resting: touching
    assert distance([0, 0, -0.0057]).value == pytest.approx(-0.003, abs=5e-4)  # 3 mm in
    assert distance([0, -0.05, 0]).value == pytest.approx(0.023, abs=1e-6)  # 5 cm out
    rng = np.random.default_rng(0)
    for _ in range(50):  # the direction `away` is the distance's gradient, outside and inside
        centre = rng.uniform([-0.03, -0.045, -0.03], [0.03, 0.03, 0.03])
        d = distance(centre)
        if abs(d.value) < 2e-4:
            continue
        step = rng.normal(size=3) * 1e-6
        change = distance(centre + step).value - d.value
        assert change == pytest.approx(d.away @ step, rel=0.02, abs=1e-10)


def test_point_in_hold_agrees_with_the_pieces():
    coal = pytest.importorskip("coal")
    hold, dot = col.hold_geometry(), coal.Sphere(1e-6)
    rng = np.random.default_rng(1)
    for point in rng.uniform(hold.centre - 0.022, hold.centre + 0.022, size=(300, 3)):
        pose = coal.Transform3s(np.eye(3), point)
        inside = min(col._query(dot, pose, p, coal.Transform3s()).value for p in hold.pieces) < 0
        assert hold.contains(point) == inside


# --- Inputs -------------------------------------------------------------------------------------


def test_two_limbs_changing_is_rejected(wall):
    start = wall.start.contacts
    plan = ContactPlan(
        wall.id, [Stance(start), Stance({**start, "right_foot": "R1", "left_foot": "L1"})]
    )
    with pytest.raises(ValueError, match="transition 0: exactly one limb"):
        validate_plan(wall, plan)


def test_unknown_hold_and_wrong_wall_are_rejected(wall):
    start = wall.start.contacts
    with pytest.raises(ValueError, match="stance 1: right_foot is on 'R99'"):
        validate_plan(
            wall, ContactPlan(wall.id, [Stance(start), Stance({**start, "right_foot": "R99"})])
        )
    with pytest.raises(ValueError, match="for wall 'other'"):
        validate_plan(wall, ContactPlan("other", [Stance(start), Stance(start)]))


# --- Solving ------------------------------------------------------------------------------------


def test_static_pose_holds_the_start_stance(wall, model):
    cfg = tr.TransitionConfig()
    s = tr.static_pose(wall, wall.start, tr.initial_guess(wall, wall.start, cfg), cfg)
    balls = np.array(model.limb_positions(s.q[0]))
    assert np.abs(balls - tr.contact_points(wall, wall.start, cfg)).max() < 1e-3
    assert np.abs(s.tau).max() <= cfg.tau_limit + 1e-6
    assert np.array(model.body_points(s.q[0]))[1].max() <= -cfg.wall_margin + 1e-6
    contacts = [dict(wall.start.contacts)]
    own = {limb: {wall.start.contacts[limb]} for limb in LIMBS}
    holds = {h.id: h for h in wall.holds}
    checks = col.check_trajectory(model, holds, s.q[0], np.zeros((rm.NV, 1)), contacts, own, DENSE)
    assert min(c.gap for c in checks) >= -col.TOUCHING  # clear of every hold and the wall


def test_a_transition_moves_one_limb_and_keeps_the_rules(wall, results, model):
    cfg, result = tr.TransitionConfig(), results[0]  # right foot R0 -> R1
    assert result.moving == "right_foot"
    m = LIMBS.index("right_foot")
    balls = [np.array(model.limb_positions(q)) for q in result.q]
    target = tr.contact_point(wall, "R1", cfg)
    assert np.linalg.norm(balls[-1][:, m] - target) < 1e-3  # arrives
    assert np.linalg.norm(result.forces["right_foot"][cfg.release_sample]) < 1e-6  # let go at K
    assert np.abs(result.tau).max() <= cfg.tau_limit + 1e-6
    holds = tr.contact_points(wall, wall.start, cfg)
    for k, b in enumerate(balls):  # the other three never leave their holds
        others = [i for i in range(4) if i != m]
        assert np.abs(b[:, others] - holds[:, others]).max() < 1e-3, f"sample {k}"
    for k in range(cfg.release_sample + 2, cfg.n_samples - 1):  # clear of the wall in mid-air
        assert balls[k][1, m] <= -cfg.clearance + 1e-6
    assert max(np.array(model.body_points(q))[1].max() for q in result.q) <= -cfg.wall_margin + 1e-6


def test_the_climb_is_clear_of_every_hold_and_the_wall(wall, results, model):
    """Checked again, independently and 4x more densely than the solver did (every 0.5 mm of
    motion): nothing enters a hold or the wall; balls only rest in, and leave and enter, their
    own sockets."""
    cfg, plan, holds = tr.TransitionConfig(), climb_one_row(wall), {h.id: h for h in wall.holds}
    for i, r in enumerate(results):
        a, b = plan.stances[i], plan.stances[i + 1]
        contacts = tr.transition_contacts(a, b, r.moving, cfg)
        own = {limb: {a.contacts[limb], b.contacts[limb]} for limb in LIMBS}
        checks = col.check_trajectory(model, holds, r.q[0], r.dq, contacts, own, DENSE)
        worst = min(checks, key=lambda c: c.gap)
        assert worst.gap >= -col.TOUCHING, f"transition {i}: {worst.label} {worst.gap * 1e3:.2f} mm"


def test_transitions_chain_and_every_hold_change_is_solved(results):
    assert [r.moving for r in results] == ["right_foot", "left_foot", "right_hand", "left_hand"]
    for a, b in zip(results, results[1:], strict=False):
        assert np.allclose(a.q[-1], b.q[0], atol=1e-6)  # dq_0 = 0 to IPOPT's tolerance


def test_optimise_returns_a_trajectory_that_round_trips(wall, tmp_path):
    traj = optimise(wall, ContactPlan(wall.id, climb_one_row(wall).stances[:2]))  # one move
    assert traj.joint_names == list(MICROBAN_JOINTS)
    assert all(a < b for a, b in zip(traj.times, traj.times[1:], strict=False))
    assert len(traj.positions) == len(traj.times)
    io.save_trajectory(traj, tmp_path / "t.json")
    assert io.load_trajectory(tmp_path / "t.json") == traj


def test_targets_are_preloaded_against_sag(results, wall):
    from rcr.optimiser import to_trajectory

    traj, r = to_trajectory(results, tr.TransitionConfig()), results[0]
    assert np.allclose(traj.positions[0], r.q[0][7:] + r.tau[0] / SERVO_KP)


def test_an_impossible_transition_raises_naming_it():
    wall = io.load_wall(EXAMPLES / "wall_l1.json")  # hands to the top row, feet on the bottom one
    start = wall.start.contacts
    plan = ContactPlan(wall.id, [Stance(start), Stance({**start, "right_hand": "R3"})])
    with pytest.raises(tr.OptimisationError, match="transition 0 .*cannot hold stance 1"):
        solve_plan(wall, plan)


# --- Watch it ------------------------------------------------------------------------------------


def pose_at(results, t: float, cfg) -> np.ndarray:
    """The optimiser's configuration (NQ) at time t [s] of the plan: linear between samples,
    the trunk quaternion normalised after interpolating (fine at these small steps)."""
    from rcr.optimiser import sample_time

    period = cfg.phase_a_duration + cfg.phase_b_duration
    i = min(int(t // period), len(results) - 1)
    local = min(t - i * period, period)
    times = [sample_time(k, cfg) for k in range(cfg.n_samples + 1)]
    k = min(int(np.searchsorted(times, local, side="right")) - 1, cfg.n_samples - 1)
    s = (local - times[k]) / (times[k + 1] - times[k])
    q = (1 - s) * results[i].q[k] + s * results[i].q[k + 1]
    q[3:7] /= np.linalg.norm(q[3:7])
    return q


def show_motion(wall: WallModel, results) -> None:
    """Play the optimiser's motion in the sim's scene (same wall, cups and hooks), on a loop."""
    import mujoco
    import mujoco.viewer

    from rcr.sim import Sim

    cfg = tr.TransitionConfig()
    sim = Sim(wall)
    duration = len(results) * (cfg.phase_a_duration + cfg.phase_b_duration)
    for i, r in enumerate(results):
        print(f"transition {i}: {r.moving}, peak |tau| {np.abs(r.tau).max():.3f} N·m")
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        start = time.perf_counter()
        while viewer.is_running():
            q = pose_at(results, (time.perf_counter() - start) % duration, cfg)
            sim.data.qpos[0:7] = q[0:7]  # the optimiser's q is laid out as the sim's qpos
            for i, name in enumerate(sim.joint_names):
                sim.data.qpos[sim._act_qpos[i]] = q[7 + MICROBAN_JOINTS.index(name)]
            mujoco.mj_forward(sim.model, sim.data)
            viewer.sync()
            time.sleep(0.02)


if __name__ == "__main__":
    ladder = test_ladder()
    rows = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    print(f"solving a climb of {rows} row(s) on the test ladder ...")
    show_motion(ladder, solve_plan(ladder, climb_one_row(ladder, rows)))
