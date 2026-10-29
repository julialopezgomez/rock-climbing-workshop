"""Optimiser tests, and a runner that solves the example plan and plays it in the sim.

    git submodule update --init                             # Microban, in third_party/microban
    uv sync --group optimiser --group sim
    uv run pytest tests/test_optimiser.py                   # the checks
    uv run mjpython tests/test_optimiser.py --hard          # solve the plan, play it as it comes
    uv run mjpython tests/test_optimiser.py --hard --save   # the same, and save the trajectory
    uv run mjpython tests/test_optimiser.py --hard --replay # play the saved trajectory, no solving

The example is the one wall and plan in data/examples: wall_hard.json and contact_plan_hard.json.
--hard takes each two consecutive stances, solves that transition, and plays it in the sim (the
same wall, holds and hooks); the robot then stays where it ended until the next one is ready.
Solving runs in a separate process, so the next transition is solved while the current one plays.
--save writes data/examples/trajectory.json (trajectory/2) after every transition, so a solve that
stops part way keeps what it solved: `positions` is what the robot executes, and the planned
angles and trunk poses in it are what --replay plays. The playback is kinematic: the robot follows
the optimiser's poses exactly, at the planned times, no physics. Add --headless for no window.
--send PI also sends each solved move to the robot at PI (rcr.link) as the trajectory's next
segment, so it can climb while the rest is solved (start it: bash scripts/pi.sh PI --start).
Solved moves are kept in the move cache (build/move_cache, rcr.optimiser.move_cache), so a move
solved once is taken from there next time: the whole example then takes about a second a move.
macOS needs `mjpython` for the window (Linux: python). Without the optimiser group or
Microban, pytest skips this file.
"""

import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rcr.robot import microban

ca = pytest.importorskip("casadi")
try:
    microban.urdf_path()
except FileNotFoundError as missing:
    pytest.skip(str(missing), allow_module_level=True)

from rcr import io  # noqa: E402
from rcr.optimiser import (  # noqa: E402
    SERVO_KP,
    feasibility,
    iter_plan,
    optimise,
    sample_time,
    solve_plan,
    start_pose,
    to_trajectory,
    validate_plan,
)
from rcr.optimiser import collision as col  # noqa: E402
from rcr.optimiser import edge_checks as fe  # noqa: E402
from rcr.optimiser import move_cache as mc  # noqa: E402
from rcr.optimiser import robot_model as rm  # noqa: E402
from rcr.optimiser import transition as tr  # noqa: E402
from rcr.robot import hooks  # noqa: E402
from rcr.types import (  # noqa: E402
    GRID_PITCH,
    LIMBS,
    MICROBAN_JOINTS,
    ContactPlan,
    Hold,
    Stance,
    Trajectory,
    WallModel,
)

EXAMPLES = Path(__file__).resolve().parent.parent / "data" / "examples"

# The example (data/examples/wall_hard.json): 34 holds up to 1.27 m. Drifting columns with uneven
# rows to 0.70 m, random holds to 0.90 m, a regular ladder above. Its plan: 38 moves until the hands
# are at 1.27 m. The tests solve its first moves; the runner below plays all of it.
WALL, PLAN = EXAMPLES / "wall_hard.json", EXAMPLES / "contact_plan_hard.json"
TRAJECTORY = EXAMPLES / "trajectory.json"
FIRST_MOVES = 4  # the example plan's first transitions, solved once for the tests below
DENSE = col.Clearance(step=0.0005)  # independent re-checks: 4x denser than the solver's


@pytest.fixture(scope="module", autouse=True)
def solve_for_real():
    """The tests solve every move: no move cache (move_cache.default_cache), unless one sets it."""
    before = os.environ.get("RCR_MOVE_CACHE")
    os.environ["RCR_MOVE_CACHE"] = "off"
    yield
    if before is None:
        os.environ.pop("RCR_MOVE_CACHE")
    else:
        os.environ["RCR_MOVE_CACHE"] = before


@pytest.fixture(scope="module")
def model():
    return rm.load_robot_model()


@pytest.fixture(scope="module")
def wall():
    return io.load_wall(WALL)


def first_row(wall: WallModel) -> ContactPlan:
    """The example plan's first FIRST_MOVES transitions."""
    plan = io.load_contact_plan(PLAN)
    assert plan.wall_id == wall.id
    return ContactPlan(plan.wall_id, plan.stances[: FIRST_MOVES + 1])


@pytest.fixture(scope="module")
def results(wall):
    return solve_plan(wall, first_row(wall))


def random_configuration(model, rng):
    q = np.zeros(rm.NQ)
    q[0:3] = rng.normal(size=3) * 0.2
    quat = rng.normal(size=4)
    q[3:7] = quat / np.linalg.norm(quat)
    q[7:] = rng.uniform(model.lower, model.upper)
    return q


# --- The robot model ----------------------------------------------------------------------------


def test_the_balls_are_the_models():
    """rcr.robot.hooks against the climbing model: each ball on its link, at its place and size
    (URDF: the fixed frames <limb>_ball_frame; MJCF: the geoms <limb>_ball_coll)."""
    import xml.etree.ElementTree as ET

    urdf = ET.parse(rm.urdf_path()).getroot()
    frames = {j.get("name"): j for j in urdf.iter("joint")}
    mjcf = ET.parse(rm.mjcf_path()).getroot()
    geoms = {
        g.get("name"): (b.get("name"), g) for b in mjcf.iter("body") for g in b.findall("geom")
    }
    for limb, hook in hooks.HOOKS.items():
        joint = frames[f"{limb}_ball_frame"]
        assert joint.find("parent").get("link") == hook.link, limb
        xyz = [float(v) for v in joint.find("origin").get("xyz").split()]
        assert np.allclose(xyz, hook.ball, atol=1e-6), limb
        body, geom = geoms[f"{limb}_ball_coll"]
        assert body == hook.link, limb
        assert np.allclose([float(v) for v in geom.get("pos").split()], hook.ball, atol=1e-6)
        assert float(geom.get("size")) == pytest.approx(hooks.BALL_RADIUS), limb


def test_kinematics_match_pinocchio(model):
    """Our CasADi kinematics (written from the URDF) against Pinocchio on the same URDF, with
    the balls (rcr.robot.hooks) added as frames. Pinocchio's quaternion is (x, y, z, w)."""
    pin = pytest.importorskip("pinocchio")
    pm = pin.buildModelFromUrdf(str(rm.urdf_path()), pin.JointModelFreeFlyer())
    balls = {}
    for limb, hook in hooks.HOOKS.items():
        link = pm.frames[pm.getFrameId(hook.link, pin.FrameType.BODY)]
        placement = link.placement * pin.SE3(np.eye(3), hooks.ball_centre(hook))
        balls[limb] = pm.addFrame(
            pin.Frame(f"{limb}_ball", link.parentJoint, placement, pin.FrameType.OP_FRAME)
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


def test_difference_inverts_integrate(model):
    rng = np.random.default_rng(4)
    for scale in (1e-9, 0.3, 2.0):  # tiny, ordinary and large trunk turns (below pi)
        q0 = random_configuration(model, rng)
        dq = rng.uniform(-1, 1, rm.NV) * scale
        q1 = np.array(model.integrate(q0, dq)).ravel()
        assert np.allclose(rm.difference(q0, q1), dq, atol=1e-9)


def test_hooks_and_contact_point_match_the_sim(model):
    """The hooks are ours (Microban has no climbing limbs yet): the optimiser and the sim must
    put the balls in the same place, and agree on where a ball sits on a hold."""
    pytest.importorskip("mujoco")
    from rcr.sim import ROBOT_PREFIX, Sim
    from rcr.sim.scene import MICROBAN_XML

    if not MICROBAN_XML.exists():
        pytest.skip(f"no Microban MJCF at {MICROBAN_XML}")
    wall = io.load_wall(WALL)
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
    # the model's boxes and cylinders: its mesh pieces are not points (collision.robot_shapes)
    kinds = (int(mujoco.mjtGeom.mjGEOM_BOX), int(mujoco.mjtGeom.mjGEOM_CYLINDER))
    shapes = [g for g in range(mm.ngeom) if int(mm.geom_type[g]) in kinds]
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


def test_what_the_sim_draws_is_inside_the_checked_hulls(model):
    """Every vertex of the model's visual meshes lies in a shape checked on its link: one of its
    hulls, or on the climbing forearms and feet one of the model's pieces or the ball. The
    pieces are a decomposition, a little inside the drawn mesh: 0.21 mm at most [Microban]."""
    mujoco = pytest.importorskip("mujoco")
    mm = mujoco.MjModel.from_xml_path(str(rm.mjcf_path()))
    md = mujoco.MjData(mm)
    q = random_configuration(model, np.random.default_rng(3))
    md.qpos[:7] = q[:7]
    for i, name in enumerate(MICROBAN_JOINTS):
        md.qpos[mm.jnt_qposadr[mm.joint(name).id]] = q[7 + i]
    mujoco.mj_kinematics(mm, md)
    positions, rotations = (np.array(m) for m in model.link_frames(q))
    index = {name: i for i, name in enumerate(model.link_names)}
    hulls: dict[str, list] = {}
    balls: dict[str, np.ndarray] = {}
    for shape in col.robot_shapes():
        i = index[shape.link]
        r = rotations[:, 3 * i : 3 * i + 3]
        centre = positions[:, i] + r @ shape.offset
        if shape.limb is not None:  # a ball, a sphere
            balls[shape.link] = centre
            continue
        normals = np.asarray(shape.geometry.normals()) @ r.T  # world
        offsets = np.asarray(shape.geometry.offsets()).ravel()
        hulls.setdefault(shape.link, []).append((normals, offsets - normals @ centre))
    visual = [g for g in range(mm.ngeom) if mm.geom_group[g] == 2 and mm.geom_dataid[g] >= 0]
    for g in visual:
        mesh = mm.geom_dataid[g]
        start, n = mm.mesh_vertadr[mesh], mm.mesh_vertnum[mesh]
        world = mm.mesh_vert[start : start + n] @ md.geom_xmat[g].reshape(3, 3).T + md.geom_xpos[g]
        link = mujoco.mj_id2name(mm, mujoco.mjtObj.mjOBJ_BODY, mm.geom_bodyid[g])
        outside = np.min([(world @ nrm.T + off).max(1) for nrm, off in hulls[link]], axis=0)
        if link in balls:  # a climbing forearm or foot: its pieces, or its ball
            ball = np.linalg.norm(world - balls[link], axis=1) - hooks.BALL_RADIUS
            assert np.minimum(outside, ball).max() < 3e-4, link
        else:
            assert outside.max() < 1e-5, link  # inside one of its link's hulls (6-digit files)


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

    assert distance([0, 0, 0]).value == pytest.approx(0.00016, abs=5e-5)  # 12.66 - 12.5 mm
    assert abs(distance([0, 0, -0.00016]).value) < col.TOUCHING  # resting: touching
    assert distance([0, 0, -0.0032]).value == pytest.approx(-0.003, abs=5e-4)  # 3 mm in
    assert distance([0, -0.05, 0]).value == pytest.approx(0.0205, abs=1e-6)  # 5 cm out
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


def self_checks(model, dq):
    """The robot's clearances to itself in one pose (trunk at the origin, no holds)."""
    q0 = np.zeros(rm.NQ)
    q0[3] = 1.0
    checks = col._check_moments(model, {}, q0, dq[:, None], [{}], {}, col.Clearance(), [(0, 0.0)])
    return [c for c in checks if c.label.endswith("(self)")]


def test_the_robot_is_kept_clear_of_itself(model):
    shapes, kept = col.robot_shapes(), {tuple(p) for p in col.self_pairs()}
    parent = {j.child: j.parent for j in rm.read_urdf(rm.urdf_path()).joints}
    left_out = {
        (shapes[i].name, shapes[j].name)
        for i in range(len(shapes))
        for j in range(i + 1, len(shapes))
        if (i, j) not in kept
        and shapes[i].link != shapes[j].link
        and parent.get(shapes[i].link) != shapes[j].link
        and parent.get(shapes[j].link) != shapes[i].link
    }
    assert left_out == {  # beyond one link or one joint: only the shapes that always overlap
        ("hip_21", "femur__configuration_right_29"),
        ("hip_2_43", "femur__configuration_left_51"),
    }
    crossed = np.zeros(rm.NV)  # legs swung in, across each other
    crossed[6 + MICROBAN_JOINTS.index("right_hip_roll")] = 0.3
    crossed[6 + MICROBAN_JOINTS.index("left_hip_roll")] = -0.3
    worst = min(self_checks(model, crossed), key=lambda c: c.gap)
    assert worst.value < -0.02 and "tibia" in worst.label
    rng = np.random.default_rng(1)
    tested = 0
    while tested < 5:  # the gradient is the distance's, both shapes moving
        dq = np.zeros(rm.NV)
        dq[6:] = rng.uniform(model.lower, model.upper) * 0.6
        near = [c for c in self_checks(model, dq) if 0.003 < c.value < 0.015]
        if not near:
            continue
        step = rng.normal(size=rm.NV) * 1e-6
        moved = next(c for c in self_checks(model, dq + step) if c.label == near[0].label)
        assert moved.value - near[0].value == pytest.approx(near[0].grad_a @ step, rel=0.02)
        tested += 1


def test_limb_reach_bounds_every_pose(model):
    reach, rng = rm.limb_reach(), np.random.default_rng(2)
    for _ in range(2000):
        q = random_configuration(model, rng)
        balls, frames = np.array(model.limb_positions(q)), model.link_frames(q)
        trunk = model.link_names.index(rm.ROOT_LINK)
        position = np.array(frames[0])[:, trunk]
        rotation = np.array(frames[1])[:, 3 * trunk : 3 * trunk + 3]
        for i, limb in enumerate(LIMBS):
            root, far = reach[limb]
            assert np.linalg.norm(balls[:, i] - (position + rotation @ root)) <= far


# --- Inputs -------------------------------------------------------------------------------------


def test_two_limbs_changing_is_rejected(wall):
    start = wall.start.contacts
    plan = ContactPlan(
        wall.id, [Stance(start), Stance({**start, "right_foot": "R1", "left_foot": "L1"})]
    )
    with pytest.raises(ValueError, match="transition 0: exactly one limb"):
        validate_plan(wall, plan)


def test_a_repeated_hold_id_is_rejected(wall):
    twice = replace(wall, holds=[*wall.holds, Hold(wall.holds[0].id, (0.6, 0.0, 1.0))])
    with pytest.raises(ValueError, match="repeated hold ids"):
        validate_plan(twice, ContactPlan(wall.id, first_row(wall).stances))


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


def test_edge_poses_unload_and_lift_the_moving_limb(wall, model):
    cfg, start = tr.TransitionConfig(), wall.start
    guess = tr.initial_guess(wall, start, cfg)
    release = tr.static_pose(wall, start, guess, cfg, unloaded="right_hand")
    assert np.abs(release.forces["right_hand"]).max() < 1e-6  # the other three carry it
    assert np.abs(release.tau).max() <= cfg.tau_limit + 1e-6
    up = (0.0, 0.0, tr.LIFT)
    lift = tr.static_pose(wall, start, release.q[0], cfg, unloaded="right_hand", away=up)
    ball = np.array(model.limb_positions(lift.q[0]))[:, LIMBS.index("right_hand")]
    socket = tr.contact_point(wall, start.contacts["right_hand"], cfg)
    assert ball == pytest.approx(socket + np.array([0.0, 0.0, tr.LIFT]), abs=1e-4)


def test_a_coarse_path_lands_on_the_full_samples():
    cfg = tr.TransitionConfig()
    rough = tr.coarse(cfg)
    assert (rough.n_samples, rough.release_sample, rough.midpoints) == (6, 2, False)
    dq = np.random.default_rng(5).normal(size=(rm.NV, rough.n_samples + 1))
    full = tr.upsample(dq)
    assert full.shape == (rm.NV, cfg.n_samples + 1)
    assert np.allclose(full[:, ::2], dq)  # coarse sample c is full sample 2c, release included
    assert np.allclose(full[:, 1], (dq[:, 0] + dq[:, 1]) / 2)


def test_feasibility_passes_the_example_moves_and_says_why_not(wall):
    stances = first_row(wall).stances
    for a, b in zip(stances[:2], stances[1:3], strict=True):  # a hand and a foot move
        verdict = feasibility(wall, a, b)
        assert verdict.feasible, verdict.reason
        assert 0 < verdict.cost <= 1  # peak torque over the edge poses, as a fraction of the limit
    a = stances[0]
    both = Stance({**a.contacts, "right_hand": "R4", "left_hand": "L4"})
    assert "exactly one limb" in feasibility(wall, a, both).reason
    too_high = Stance({**a.contacts, "right_hand": "R6"})  # a hand up 0.225 m: past the torque
    verdict = feasibility(wall, a, too_high)
    assert not verdict.feasible and verdict.reason.split(":")[0] in fe.CHECKS
    high = Hold("HIGH", (0.5, 0.0, 1.0))  # a hand 0.8 m above the feet: no solve needed
    tall = replace(wall, id="tall", holds=[*wall.holds, high])
    verdict, ran = fe.trace(tall, a, Stance({**a.contacts, "right_hand": "HIGH"}))
    assert verdict.reason.startswith("reach: ") and [n for n, _ in ran] == ["rules", "reach"]


def test_a_transition_moves_one_limb_and_keeps_the_rules(wall, results, model):
    cfg, result = tr.TransitionConfig(), results[2]  # right foot R0 -> R1, after the hands
    assert result.moving == "right_foot"
    m = LIMBS.index("right_foot")
    balls = [np.array(model.limb_positions(q)) for q in result.q]
    target = tr.contact_point(wall, "R1", cfg)
    assert np.linalg.norm(balls[-1][:, m] - target) < 1e-3  # arrives
    assert np.linalg.norm(result.forces["right_foot"][cfg.release_sample]) < 1e-6  # let go at K
    assert np.abs(result.tau).max() <= cfg.tau_limit + 1e-6
    holds = tr.contact_points(wall, first_row(wall).stances[2], cfg)
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
    cfg, plan, holds = tr.TransitionConfig(), first_row(wall), {h.id: h for h in wall.holds}
    for i, r in enumerate(results):
        a, b = plan.stances[i], plan.stances[i + 1]
        contacts = tr.transition_contacts(a, b, r.moving, cfg)
        contacts += [dict(b.contacts)] * (len(r.q) - len(contacts))  # the settle (cfg.settle)
        own = {limb: {a.contacts[limb], b.contacts[limb]} for limb in LIMBS}
        checks = col.check_trajectory(model, holds, r.q[0], r.dq, contacts, own, DENSE)
        worst = min(checks, key=lambda c: c.gap)
        assert worst.gap >= -col.TOUCHING, f"transition {i}: {worst.label} {worst.gap * 1e3:.2f} mm"


def test_the_limits_hold_between_samples_too(wall, results, model):
    """At every interval midpoint (the joints interpolated, as the controller will): some set of
    balanced, friction-respecting hold forces keeps every motor within the limit, and the held
    balls stay within midpoint_play of their holds. Checked independently of the optimiser's own
    midpoint forces."""
    cfg, plan = tr.TransitionConfig(), first_row(wall)
    for i, r in enumerate(results):
        points = tr.contact_points(wall, plan.stances[i], cfg)
        for k in range(cfg.n_samples):
            q = np.array(model.integrate(r.q[0], (r.dq[:, k] + r.dq[:, k + 1]) / 2)).ravel()
            held = [limb for limb in LIMBS if limb != r.moving or k + 1 <= cfg.release_sample]
            balls = np.array(model.limb_positions(q))
            for limb in held:
                j = LIMBS.index(limb)
                assert np.linalg.norm(balls[:, j] - points[:, j]) <= cfg.midpoint_play + 1e-4
            opti = ca.Opti()
            forces = {limb: opti.variable(3) for limb in held}
            wrench, tau = tr.static_balance(model, q, forces)
            opti.subject_to(wrench == 0)
            for limb in held:
                tr.friction_pyramid(opti, forces[limb], cfg.normals[limb], cfg.mu[limb])
            opti.minimize(ca.sumsqr(tau))
            opti.solver("ipopt", {"print_time": False}, {"print_level": 0, "sb": "yes"})
            assert np.abs(opti.solve().value(tau)).max() <= cfg.tau_limit + 1e-4, f"{i}, {k}"


def test_transitions_chain_and_every_hold_change_is_solved(wall, results):
    stances = first_row(wall).stances
    pairs = zip(stances, stances[1:], strict=False)
    assert [r.moving for r in results] == [tr.moving_limb(a, b, 0) for a, b in pairs]
    for a, b in zip(results, results[1:], strict=False):
        assert np.allclose(a.q[-1], b.q[0], atol=1e-6)  # dq_0 = 0 to IPOPT's tolerance


def test_optimise_returns_a_trajectory_that_round_trips(wall, tmp_path):
    traj = optimise(wall, ContactPlan(wall.id, first_row(wall).stances[2:4]))  # a foot move
    assert traj.joint_names == list(MICROBAN_JOINTS)
    assert all(a < b for a, b in zip(traj.times, traj.times[1:], strict=False))
    assert len(traj.positions) == len(traj.times)
    io.save_trajectory(traj, tmp_path / "t.json")
    assert io.load_trajectory(tmp_path / "t.json") == traj


def shifted(wall: WallModel, holes_x: int, holes_z: int) -> WallModel:
    """The same wall with every hold moved by whole grid holes."""
    dx, dz = holes_x * GRID_PITCH, holes_z * GRID_PITCH
    moved = [Hold(h.id, (h.position[0] + dx, 0.0, h.position[2] + dz), h.kind) for h in wall.holds]
    return replace(wall, holds=moved)


def test_the_move_cache_keys_on_the_layout_not_on_the_wall(wall, results, tmp_path):
    a, b = first_row(wall).stances[:2]
    key, ref = mc.MoveCache.key(wall, a, b)
    there = shifted(wall, 3, 7)
    assert mc.MoveCache.key(there, a, b)[0] == key  # the same holds, 3 holes right and 7 up
    assert mc.MoveCache.key(wall, *first_row(wall).stances[1:3])[0] != key  # another move
    nudged = a.contacts["right_hand"]  # 0.3 mm off the grid: not cached rather than misread
    off_grid = replace(wall, holds=[
        Hold(h.id, (h.position[0] + 0.0003, 0.0, h.position[2]), h.kind) if h.id == nudged else h
        for h in wall.holds
    ])  # fmt: skip
    assert mc.MoveCache.key(off_grid, a, b) is None

    cfg, cache = replace(tr.TransitionConfig(), settle=True), mc.MoveCache(tmp_path)
    cache.put(wall, a, b, cfg, results[0], solve_time=12.0)
    entry = cache.get(there, a, b, cfg)  # stored on one wall, read on the shifted one
    assert entry.feasible and entry.solve_time == 12.0
    lift = np.array([3 * GRID_PITCH, 0.0, 7 * GRID_PITCH])
    assert np.allclose(entry.result.q[:, :3], results[0].q[:, :3] + lift)
    assert np.allclose(entry.result.q[:, 3:], results[0].q[:, 3:])  # the angles stay
    assert cache.get(wall, a, b, replace(cfg, mu=dict.fromkeys(LIMBS, 0.5))) is None  # new version
    cache.put(wall, *first_row(wall).stances[1:3], cfg, reason="stance: infeasible")
    assert not cache.get(there, *first_row(wall).stances[1:3], cfg).feasible
    assert cache.size(cfg) == (1, 1)


def test_feasibility_answers_from_the_cache_and_stores_its_rejections(wall, tmp_path):
    a = first_row(wall).stances[0]
    far = Stance({**a.contacts, "right_hand": "R6"})  # rejected by the checks (a 0.225 m reach)
    cfg = tr.TransitionConfig()
    os.environ["RCR_MOVE_CACHE"] = str(tmp_path)
    try:
        first, ran = fe.trace(wall, a, far)
        assert not first.feasible and "cache" in [n for n, _ in ran]
        again, ran = fe.trace(wall, a, far)  # now from the cache: no check runs
        assert again.reason == f"{first.reason} (cached)" and ran[-1][0] == "cache"
        assert mc.MoveCache(tmp_path).size(cfg) == (0, 1)
    finally:
        os.environ["RCR_MOVE_CACHE"] = "off"


def test_targets_are_preloaded_against_sag(results, wall):
    from rcr.optimiser import to_trajectory

    traj, r = to_trajectory(results, tr.TransitionConfig()), results[0]
    assert np.allclose(traj.positions[0], r.q[0][7:] + r.tau[0] / SERVO_KP)
    last = results[-1].q[-1]  # the planned motion goes along, for an exact replay
    assert np.allclose(traj.planned_positions[-1], last[7:])
    assert np.allclose(traj.trunk_poses[-1], last[:7])
    assert len(traj.trunk_poses) == len(traj.planned_positions) == len(traj.times)
    assert np.allclose(planned_pose(traj, traj.times[-1]), last)


def test_an_impossible_transition_raises_naming_it():
    wall = io.load_wall(WALL)  # a hand up 0.225 m, the feet on the bottom row
    start = wall.start.contacts
    plan = ContactPlan(wall.id, [Stance(start), Stance({**start, "right_hand": "R6"})])
    with pytest.raises(tr.OptimisationError, match="transition 0 .*cannot hold stance 1"):
        solve_plan(wall, plan)


# --- Run a contact plan in the sim ---------------------------------------------------------------


def solve_in_background(wall: WallModel, plan: ContactPlan, out) -> None:
    """Worker process: the plan's start pose, then each transition in order, each starting where
    the previous one ended. Each is sent on `out` (one end of a Pipe) as soon as it is solved:
    ("start", q), ("transition", i, TransitionResult, seconds), then ("done",) or ("error", text).
    Exits as soon as the runner is gone, however it ended (its collision workers follow).
    """
    import threading

    parent = os.getppid()

    def watch() -> None:
        while os.getppid() == parent:
            time.sleep(0.5)
        os._exit(0)

    threading.Thread(target=watch, daemon=True).start()
    try:
        cfg = tr.TransitionConfig()
        q = start_pose(wall, plan, cfg)
        out.send(("start", q))
        start = time.perf_counter()
        for i, result in enumerate(iter_plan(wall, plan, q, cfg)):  # with the move cache
            out.send(("transition", i, result, time.perf_counter() - start))
            start = time.perf_counter()
        out.send(("done",))
    except Exception as e:  # report it to the player; the robot stays where it is
        out.send(("error", f"{type(e).__name__}: {e}"))


class PlanPlayer:
    """What the robot shows at each moment: the next solved transition as soon as the previous
    one has finished playing, and the last pose reached while waiting."""

    def __init__(self, cfg: tr.TransitionConfig, transitions: int):
        self.cfg, self.transitions = cfg, transitions
        self.ready: list = []  # solved, not yet played
        self.playing = None  # (index, result, start time)
        self.pose = None  # the robot's configuration (NQ), None until the start pose is known
        self.finished = False  # the worker is done (or failed)
        self.played = 0

    def receive(self, message) -> None:
        kind = message[0]
        if kind == "start":
            self.pose = message[1]
            print("start pose solved; solving transition 0 ...", flush=True)
        elif kind == "transition":
            _, i, result, seconds = message
            self.ready.append((i, result))
            waiting = "" if self.playing else ", playing it now"
            print(
                f"transition {i} ({result.moving}) solved in {seconds:.0f} s{waiting}", flush=True
            )
        elif kind == "done":
            self.finished = True
        else:
            self.finished = True
            print(f"stopped: {message[1]}", flush=True)

    def update(self, now: float) -> None:
        """Advance to `now` [s]: finish the playing transition, start the next ready one."""
        if self.playing is not None:
            i, result, start = self.playing
            elapsed = now - start
            if elapsed < sample_time(len(result.q) - 1, self.cfg):
                self.pose = pose_at(result, elapsed, self.cfg)
                return
            self.pose, self.playing = result.q[-1], None
            self.played += 1
            more = self.played < self.transitions and not self.finished
            print(
                f"transition {i} played"
                + (f"; waiting for transition {i + 1} ..." if more and not self.ready else ""),
                flush=True,
            )
        if self.ready:
            i, result = self.ready.pop(0)
            self.playing = (i, result, now)
            self.pose = result.q[0]

    @property
    def done(self) -> bool:
        return self.finished and self.playing is None and not self.ready


def pose_at(result, t: float, cfg) -> np.ndarray:
    """The optimiser's configuration (NQ) at time t [s] of one transition: linear between
    samples, the trunk quaternion normalised after interpolating (fine at these small steps)."""
    times = [sample_time(k, cfg) for k in range(len(result.q))]
    t = min(t, times[-1])
    k = min(int(np.searchsorted(times, t, side="right")) - 1, len(times) - 2)
    s = (t - times[k]) / (times[k + 1] - times[k])
    q = (1 - s) * result.q[k] + s * result.q[k + 1]
    q[3:7] /= np.linalg.norm(q[3:7])
    return q


def planned_pose(traj: Trajectory, t: float) -> np.ndarray:
    """The planned configuration (NQ) at time t [s] of a trajectory with its trunk poses and
    planned angles: linear between samples, the quaternion normalised after interpolating."""
    times = np.asarray(traj.times)
    k = int(np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(times) - 2))
    s = float(np.clip((t - times[k]) / (times[k + 1] - times[k]), 0.0, 1.0))
    rows = [np.r_[traj.trunk_poses[i], traj.planned_positions[i]] for i in (k, k + 1)]
    q = (1 - s) * rows[0] + s * rows[1]
    q[3:7] /= np.linalg.norm(q[3:7])
    return q


def run_plan(
    wall: WallModel, plan: ContactPlan, view: bool = True, save: bool = False, send: str = ""
):
    """Solve the plan in a worker process and play each transition in the sim as it arrives.
    save: write TRAJECTORY after every solved transition (what is solved so far).
    send: the robot's address; each solved transition goes to it as the trajectory's next
    segment (rcr.link), so it can climb while the rest is solved. It is not started here."""
    import multiprocessing

    from rcr import link

    context = multiprocessing.get_context("spawn")
    messages, worker_end = context.Pipe(duplex=False)
    # not a daemon: the collision check runs its own worker processes; stopped in `finally`
    worker = context.Process(target=solve_in_background, args=(wall, plan, worker_end))
    worker.start()
    worker_end.close()  # this process only reads
    count = len(plan.stances) - 1
    print(f"plan for wall {wall.id!r}: {count} transitions; solving the start pose ...")
    cfg, results = tr.TransitionConfig(), []
    player = PlanPlayer(cfg, count)
    robot, sent = (link.LaptopLink(send) if send else None), 0

    def to_robot(message) -> None:
        """The new samples to the robot: a segment per transition, then the end of it."""
        nonlocal robot, sent
        try:
            if message[0] == "transition":
                full = to_trajectory(results, cfg)
                if sent == 0:
                    robot.send_trajectory(full, final=False)
                else:
                    robot.append_segment(link.segment(full, sent))
                print(f"  sent to {send}: up to {full.times[-1]:.1f} s", flush=True)
                if sent == 0:
                    print(f"  start it: bash scripts/pi.sh {send} --start", flush=True)
                sent = len(full.times)
            elif sent:
                robot.finish()  # done, or stopped: nothing more will come
        except link.LinkError as e:
            print(f"  not sent, and no more will be: {e}", flush=True)
            robot.close()
            robot = None

    def pose(now: float):
        while not player.finished and messages.poll():
            message = messages.recv()
            player.receive(message)
            if message[0] == "transition":
                results.append(message[2])
            if message[0] == "transition" and save:
                io.save_trajectory(to_trajectory(results, cfg), TRAJECTORY)
                print(f"  saved {len(results)} of {count} transitions in {TRAJECTORY}", flush=True)
            if robot is not None and message[0] != "start":
                to_robot(message)
        player.update(now)
        return player.pose

    try:
        play(wall, pose, lambda: player.done, view)
    finally:
        worker.terminate()
        worker.join()
        if robot is not None:
            robot.close()
    print(f"{player.played} of {player.transitions} transitions played")


def replay(wall: WallModel, view: bool = True) -> None:
    """Play TRAJECTORY (saved by --save) in the sim, without solving anything."""
    traj = io.load_trajectory(TRAJECTORY)
    if traj.trunk_poses is None or traj.planned_positions is None:
        sys.exit(f"{TRAJECTORY} has no planned motion to replay: solve it with --hard --save")
    print(f"replaying {TRAJECTORY}: {traj.times[-1]:.0f} s, {len(traj.times)} samples")
    start = None

    def pose(now: float):
        nonlocal start
        start = now if start is None else start
        return planned_pose(traj, now - start)

    play(
        wall, pose, lambda: start is not None and time.perf_counter() - start > traj.times[-1], view
    )
    print("replayed")


def play(wall: WallModel, pose, done, view: bool) -> None:
    """Show pose(now) in the sim, kinematically (None: nothing to show yet), until the window
    closes or, without one, until done()."""
    import mujoco

    from rcr.sim import Sim

    sim = Sim(wall, out=None)

    def frame() -> None:
        q = pose(time.perf_counter())
        if q is not None:
            sim.data.qpos[0:7] = q[0:7]  # the optimiser's q is laid out as the sim's qpos
            for i, name in enumerate(sim.joint_names):
                sim.data.qpos[sim._act_qpos[i]] = q[7 + MICROBAN_JOINTS.index(name)]
            mujoco.mj_forward(sim.model, sim.data)

    if view:
        import mujoco.viewer

        with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
            while viewer.is_running():  # keeps showing the last pose after the plan ends
                frame()
                viewer.sync()
                time.sleep(0.02)
    else:
        while not done():
            frame()
            time.sleep(0.02)


USAGE = """usage: tests/test_optimiser.py --hard [--save | --replay] [--send PI] [--headless]
    --hard            solve the example plan (contact_plan_hard.json) and play it as it comes
    --hard --save     the same, and save it in data/examples/trajectory.json
    --hard --replay   play data/examples/trajectory.json, without solving
    --send PI         also send each solved move to the robot at PI (rcr.link), move by move"""

if __name__ == "__main__":
    args = sys.argv[1:]
    send = ""
    if "--send" in args:
        i = args.index("--send")
        if i + 1 >= len(args) or args[i + 1].startswith("--"):
            sys.exit("--send needs the robot's address\n" + USAGE)
        send = args.pop(i + 1).split("@")[-1]  # pi@172.20.10.8 as for scripts/pi.sh: the address
        args.pop(i)
    flags = set(args)
    if "--hard" not in flags or not flags <= {"--hard", "--save", "--replay", "--headless"}:
        sys.exit(USAGE)
    if send and "--replay" in flags:
        sys.exit("--send sends what is solved; to send a saved one: scripts/link_pi.py stream")
    if {"--save", "--replay"} <= flags:
        sys.exit("--save solves and saves, --replay plays what was saved: pick one\n" + USAGE)
    wall, view = io.load_wall(WALL), "--headless" not in flags
    if "--replay" in flags:
        replay(wall, view)
    else:
        run_plan(wall, io.load_contact_plan(PLAN), view, save="--save" in flags, send=send)
