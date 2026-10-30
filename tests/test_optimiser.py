"""Optimiser tests, and a runner that solves a contact plan and plays it in the organisers' sim.

    git submodule update --init                                  # Microban and their simulator
    uv sync --group optimiser --group sim
    uv run pytest tests/test_optimiser.py                        # the checks
    uv run mjpython tests/test_optimiser.py                   # solve build/plan_N1.json, play it
    uv run mjpython tests/test_optimiser.py --save            # also save build/trajectory_N1.json
    uv run mjpython tests/test_optimiser.py --replay          # play the saved one, no solving
    uv run mjpython tests/test_optimiser.py PLAN ...          # another plan file
    uv run mjpython tests/test_optimiser.py --help            # every option

The wall is the organisers' N1 (data/examples/N1.json, from their simulator by
scripts/wall_generator.py). A plan's wall is data/examples/<its wall_id>.json, else it is rebuilt
from the plan saved by scripts/plan_sim_wall.py --save (plan_walls), and it is played on their wall.
There is no example contact plan yet: the tests that solve one skip until our planner's plan on N1
is saved as data/examples/contact_plan_N1.json.
Run directly (not by pytest), it takes each two consecutive stances, solves that transition, and
plays it in the organisers' sim (the same wall, holds and balls); the robot then stays where it
ended until the next one is ready.
Solving runs in a separate process, so the next transition is solved while the current one plays.
--save writes build/trajectory_<wall id>.json (trajectory/2) after every transition, so a solve that
stops part way keeps what it solved: `positions` is what the robot executes, and the planned
angles and trunk poses in it are what --replay plays. The playback is kinematic: the robot follows
the optimiser's poses exactly, at the planned times, no physics. Add --headless for no window.
--send PI also sends each solved move to the robot at PI (rcr.link) as the trajectory's next
segment, so it can climb while the rest is solved (start it: bash scripts/pi.sh PI --start).
Collisions with the holds, the wall and the robot itself are not modelled by the optimiser.
macOS needs `mjpython` for the window (Linux: python). Without the optimiser group or the robot
model, pytest skips this file; without the sim group, the test against their sim.
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
    iter_plan,
    optimise,
    sample_time,
    solve_plan,
    start_pose,
    to_trajectory,
    validate_plan,
)
from rcr.optimiser import robot_model as rm  # noqa: E402
from rcr.optimiser import transition as tr  # noqa: E402
from rcr.robot import hooks  # noqa: E402
from rcr.types import (  # noqa: E402
    LIMBS,
    MICROBAN_JOINTS,
    ContactPlan,
    Hold,
    Stance,
    Trajectory,
    WallModel,
)

EXAMPLES = Path(__file__).resolve().parent.parent / "data" / "examples"

# The organisers' N1: a ladder, rungs every 5 rows (75 mm), holds in columns 18, 21 (left) and 26,
# 29 (right); start: feet c21_r2, c26_r2, hands c18_r22, c29_r22. The tests solve the first moves
# of its plan, once there is one (PLAN).
WALL, PLAN = EXAMPLES / "N1.json", EXAMPLES / "contact_plan_N1.json"
# The runner's files: the plan our planner saved (scripts/plan_sim_wall.py --save), and the solved
# trajectory, build/trajectory_<wall id>.json [Ours]
BUILD = EXAMPLES.parent.parent / "build"
RUN_PLAN = BUILD / "plan_N1.json"


def trajectory_file(wall_id: str) -> Path:
    return BUILD / f"trajectory_{wall_id}.json"


# The organisers' wall each loaded plan is on, by wall id, for the playback (plan_walls)
_SIM_WALLS: dict = {}


def plan_walls(plan: ContactPlan) -> WallModel:
    """The wall a plan is on: data/examples/<wall_id>.json if there is one; else rebuilt from
    scripts/plan_sim_wall.py's output, whose hold ids are their grid holes, "r<row>_c<col>", so each
    position is their Hold(col, row).bolt_axis, as that script places it. For the playback, their
    whole wall: l<level>_s<seed> regenerated, else the bundled wall (N1, N2, N3) with all the plan's
    holds (--wall N2 still names the wall l<level>_s<seed>), else only the plan's holds."""
    import re

    from climbing_sim import Wall
    from climbing_sim.routes import generate_wall
    from climbing_sim.wall import Hold as TheirHold
    from climbing_sim.wall import bundled_walls

    example = EXAMPLES / f"{plan.wall_id}.json"
    if example.exists():
        _SIM_WALLS[plan.wall_id] = Wall.load(plan.wall_id)
        return io.load_wall(example)
    used = {hold for stance in plan.stances for hold in stance.contacts.values()}
    grid = {}
    for hold in used:
        match = re.fullmatch(r"r(\d+)_c(\d+)", hold)
        if match is None:
            sys.exit(
                f"no wall {plan.wall_id}: not in {EXAMPLES}, and hold {hold} is not r<row>_c<col>"
            )
        grid[hold] = TheirHold(col=int(match[2]), row=int(match[1]))
    candidates = [Wall.load(name) for name in bundled_walls()]
    seeded = re.fullmatch(r"l(\d)_s(\d+)", plan.wall_id)
    if seeded:
        candidates.insert(0, generate_wall(int(seeded[1]), int(seeded[2])))
    sim_wall = next(
        (w for w in candidates if set(grid.values()) <= set(w.holds)),
        Wall(name=plan.wall_id, holds=list(grid.values())),
    )
    print(f"wall {plan.wall_id}: rebuilt from the plan's holds; played on {sim_wall.name}")
    _SIM_WALLS[plan.wall_id] = sim_wall
    holds = [
        Hold(f"r{h.row}_c{h.col}", tuple(round(float(v), 6) for v in h.bolt_axis))
        for h in sim_wall.holds
    ]
    return WallModel(id=plan.wall_id, level=0, holds=holds, start=plan.stances[0])


def organisers_wall(wall_id: str):
    """Their wall for the sim: the one plan_walls found, else their wall of that name."""
    from climbing_sim import Wall

    return _SIM_WALLS.get(wall_id) or Wall.load(wall_id)


FIRST_MOVES = 4  # the plan's first transitions, solved once for the tests below
CFG = tr.TransitionConfig()


@pytest.fixture(scope="module")
def model():
    return rm.load_robot_model()


@pytest.fixture(scope="module")
def wall():
    return io.load_wall(WALL)


def first_row(wall: WallModel) -> ContactPlan:
    """The plan's first FIRST_MOVES transitions (skips the test if there is no plan yet)."""
    if not PLAN.exists():
        pytest.skip(f"no contact plan on the organisers' wall yet ({PLAN.name})")
    plan = io.load_contact_plan(PLAN)
    assert plan.wall_id == wall.id
    return ContactPlan(plan.wall_id, plan.stances[: FIRST_MOVES + 1])


@pytest.fixture(scope="module")
def results(wall):
    return solve_plan(wall, first_row(wall), CFG)


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


def test_balls_and_contact_point_match_the_organisers_sim(model):
    """The optimiser and the organisers' sim (climbing_sim) put the balls in the same place, and
    a ball resting in a socket sits on our contact point."""
    climbing_sim = pytest.importorskip("climbing_sim")

    wall = io.load_wall(WALL)
    sim = climbing_sim.ClimbingSim(climbing_sim.Wall.load(wall.id), cameras=[])
    rng = np.random.default_rng(1)
    for _ in range(5):
        q = random_configuration(model, rng)
        by_name = dict(zip(MICROBAN_JOINTS, q[7:], strict=True))
        theirs = sim.ball_positions_at([by_name[n] for n in climbing_sim.JOINTS], q[0:3], q[3:7])
        ours = np.array(model.limb_positions(q))
        for i, limb in enumerate(LIMBS):
            assert np.allclose(ours[:, i], theirs[limb], atol=1e-6)
    cfg = CFG
    sag = np.array([0.0, 0.0, tr.SOCKET_RADIUS - hooks.BALL_RADIUS])  # the ball on the bottom
    for limb, hold in sim.wall.start.items():
        point = tr.contact_point(wall, wall.start.contacts[limb], cfg)
        assert np.allclose(point, hold.socket - sag, atol=1e-9)


# --- Inputs -------------------------------------------------------------------------------------


def test_two_limbs_changing_is_rejected(wall):
    start = wall.start.contacts
    plan = ContactPlan(
        wall.id, [Stance(start), Stance({**start, "right_foot": "c26_r7", "left_foot": "c21_r7"})]
    )
    with pytest.raises(ValueError, match="transition 0: exactly one limb"):
        validate_plan(wall, plan)


def test_a_repeated_hold_id_is_rejected(wall):
    twice = replace(wall, holds=[*wall.holds, Hold(wall.holds[0].id, (0.6, 0.0, 1.0))])
    with pytest.raises(ValueError, match="repeated hold ids"):
        validate_plan(twice, ContactPlan(wall.id, [wall.start]))


def test_unknown_hold_and_wrong_wall_are_rejected(wall):
    start = wall.start.contacts
    with pytest.raises(ValueError, match="stance 1: right_foot is on 'c99_r99'"):
        validate_plan(
            wall, ContactPlan(wall.id, [Stance(start), Stance({**start, "right_foot": "c99_r99"})])
        )
    with pytest.raises(ValueError, match="for wall 'other'"):
        validate_plan(wall, ContactPlan("other", [Stance(start), Stance(start)]))


# --- Solving ------------------------------------------------------------------------------------


def test_static_pose_holds_the_start_stance(wall, model):
    cfg = CFG
    s = tr.static_pose(wall, wall.start, tr.initial_guess(wall, wall.start, cfg), cfg)
    balls = np.array(model.limb_positions(s.q[0]))
    assert np.abs(balls - tr.contact_points(wall, wall.start, cfg)).max() < 1e-3
    assert np.abs(s.tau).max() <= cfg.tau_limit + 1e-6


def test_a_coarse_path_lands_on_the_full_samples():
    cfg = CFG
    rough = tr.coarse(cfg)
    assert (rough.n_samples, rough.release_sample, rough.midpoints) == (6, 2, False)
    dq = np.random.default_rng(5).normal(size=(rm.NV, rough.n_samples + 1))
    full = tr.upsample(dq)
    assert full.shape == (rm.NV, cfg.n_samples + 1)
    assert np.allclose(full[:, ::2], dq)  # coarse sample c is full sample 2c, release included
    assert np.allclose(full[:, 1], (dq[:, 0] + dq[:, 1]) / 2)


def test_the_keep_out_points_box_each_part_but_the_hook_ends(model):
    points = rm.keep_out_points()
    parts = {link for link, _ in points}
    assert len(points) == 8 * len(parts) and {"head", "trunk", "foot_climb"} <= parts
    assert tr.CUP_FRONT == pytest.approx(-0.043, abs=1e-3)  # the cups' front, from their sim
    for limb, hook in hooks.HOOKS.items():  # a forearm's or foot's box leaves out its ball's end
        corners = np.array([p for link, p in points if link == hook.link])
        centre = hooks.ball_centre(hook)
        assert not np.all((corners.min(axis=0) <= centre) & (centre <= corners.max(axis=0))), limb
    assert np.array(model.body_points(np.zeros(rm.NQ))).shape == (3, len(points))


def test_the_hooks_point_along_their_shafts(model):
    axes = rm.hook_axes()  # from the meshes: the hands' shafts along the forearm, the feet's +y
    assert axes["right_hand"][0] < -0.9 and axes["left_hand"][0] > 0.9
    assert axes["right_foot"][1] > 0.9 and axes["left_foot"][1] > 0.9
    u = np.array(model.hook_directions(np.r_[0, 0, 0, 1, 0, 0, 0, np.zeros(rm.NJ)]))
    assert np.allclose(np.linalg.norm(u, axis=0), 1)  # unit vectors in the wall frame
    up = replace(CFG, hook_up=-1.0)  # anything counts as up: no cost
    assert float(tr.hook_tilt(model, np.r_[0, 0, 0, 1, 0, 0, 0, np.zeros(rm.NJ)], up)) == 0


def test_the_swing_path_goes_up_out_across_in_and_down():
    cfg = replace(CFG, swing_lift=0.025, swing_out=0.04, swing_vertical=0.25)
    start, target = np.array([0.4, -0.03, 0.5]), np.array([0.5, -0.03, 0.62])
    up, out = np.array([0.0, 0.0, 0.025]), np.array([0.0, -0.04, 0.0])
    path = lambda s: tr.swing_point(start, target, s, cfg)  # noqa: E731
    corners = [start, start + up, start + up + out, target + up + out, target + up, target]
    for s, corner in zip([0.0, 0.25, 0.375, 0.625, 0.75, 1.0], corners, strict=True):
        assert np.allclose(path(s), corner)
    assert np.allclose(path(0.125), start + up / 2)  # straight up out of the old cup
    assert np.allclose(path(0.875), target + up / 2)  # straight down into the new one


def test_a_transition_moves_one_limb_and_keeps_the_rules(wall, results, model):
    plan, cfg = first_row(wall), CFG
    i = next(i for i, r in enumerate(results) if r.moving.endswith("foot"))  # the first foot move
    result, a, b = results[i], plan.stances[i], plan.stances[i + 1]
    m = LIMBS.index(result.moving)
    balls = [np.array(model.limb_positions(q)) for q in result.q]
    target = tr.contact_point(wall, b.contacts[result.moving], cfg)
    assert np.linalg.norm(balls[-1][:, m] - target) < 1e-3  # arrives
    assert np.linalg.norm(result.forces[result.moving][cfg.release_sample]) < 1e-6  # let go at K
    assert np.abs(result.tau).max() <= cfg.tau_limit + 1e-6
    holds = tr.contact_points(wall, a, cfg)
    for k, b in enumerate(balls):  # the other three never leave their holds
        others = [i for i in range(4) if i != m]
        assert np.abs(b[:, others] - holds[:, others]).max() < 1e-3, f"sample {k}"


def test_the_limits_hold_between_samples_too(wall, results, model):
    """At every interval midpoint (the joints interpolated, as the controller will): some set of
    balanced, friction-respecting hold forces keeps every motor within the limit, and the held
    balls stay within midpoint_play of their holds. Checked independently of the optimiser's own
    midpoint forces."""
    cfg, plan = CFG, first_row(wall)
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
    traj = optimise(wall, ContactPlan(wall.id, first_row(wall).stances[:2]))  # the first move
    assert traj.joint_names == list(MICROBAN_JOINTS)
    assert all(a < b for a, b in zip(traj.times, traj.times[1:], strict=False))
    assert len(traj.positions) == len(traj.times)
    io.save_trajectory(traj, tmp_path / "t.json")
    assert io.load_trajectory(tmp_path / "t.json") == traj


def test_targets_are_preloaded_against_sag(results, wall):
    from rcr.optimiser import to_trajectory

    traj, r = to_trajectory(results, CFG), results[0]
    assert np.allclose(traj.positions[0], r.q[0][7:] + r.tau[0] / SERVO_KP)
    last = results[-1].q[-1]  # the planned motion goes along, for an exact replay
    assert np.allclose(traj.planned_positions[-1], last[7:])
    assert np.allclose(traj.trunk_poses[-1], last[:7])
    assert len(traj.trunk_poses) == len(traj.planned_positions) == len(traj.times)
    assert np.allclose(planned_pose(traj, traj.times[-1]), last)


def test_an_impossible_transition_raises_naming_it():
    wall = io.load_wall(WALL)  # a hand up 0.225 m, the feet on the bottom row
    start = wall.start.contacts
    plan = ContactPlan(wall.id, [Stance(start), Stance({**start, "right_hand": "c29_r37"})])
    with pytest.raises(tr.OptimisationError, match="transition 0 .*cannot hold stance 1"):
        solve_plan(wall, plan)


# --- Run a contact plan in the sim ---------------------------------------------------------------


def solve_in_background(wall: WallModel, plan: ContactPlan, out, cfg=None) -> None:
    """Worker process: the plan's start pose, then each transition in order, each starting where
    the previous one ended. Each is sent on `out` (one end of a Pipe) as soon as it is solved:
    ("start", q), ("transition", i, TransitionResult, seconds), then ("done",) or ("error", text).
    Exits as soon as the runner is gone, however it ended.
    """
    import threading

    parent = os.getppid()

    def watch() -> None:
        while os.getppid() == parent:
            time.sleep(0.5)
        os._exit(0)

    threading.Thread(target=watch, daemon=True).start()
    try:
        cfg = cfg or tr.TransitionConfig()
        q = start_pose(wall, plan, cfg)
        out.send(("start", q))
        start = time.perf_counter()
        for i, result in enumerate(iter_plan(wall, plan, q, cfg)):
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
    wall: WallModel,
    plan: ContactPlan,
    view: bool = True,
    save: bool = False,
    send: str = "",
    cfg: tr.TransitionConfig | None = None,
):
    """Solve the plan in a worker process and play each transition in the sim as it arrives.
    save: write trajectory_file(wall.id) after every solved transition (what is solved so far).
    send: the robot's address; each solved transition goes to it as the trajectory's next
    segment (rcr.link), so it can climb while the rest is solved. It is not started here.
    cfg: the optimiser's settings (default TransitionConfig())."""
    import multiprocessing

    from rcr import link

    context = multiprocessing.get_context("spawn")
    messages, worker_end = context.Pipe(duplex=False)
    # stopped in `finally`
    cfg, results = cfg or tr.TransitionConfig(), []
    worker = context.Process(target=solve_in_background, args=(wall, plan, worker_end, cfg))
    worker.start()
    worker_end.close()  # this process only reads
    count = len(plan.stances) - 1
    print(f"plan for wall {wall.id!r}: {count} transitions; solving the start pose ...")
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
                out = trajectory_file(wall.id)
                out.parent.mkdir(parents=True, exist_ok=True)
                io.save_trajectory(to_trajectory(results, cfg), out)
                print(f"  saved {len(results)} of {count} transitions in {out}", flush=True)
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
    """Play trajectory_file(wall.id) (saved by --save) in the sim, without solving anything."""
    path = trajectory_file(wall.id)
    traj = io.load_trajectory(path)
    if traj.trunk_poses is None or traj.planned_positions is None:
        sys.exit(f"{path} has no planned motion to replay: solve it with --save")
    print(f"replaying {path}: {traj.times[-1]:.0f} s, {len(traj.times)} samples")
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
    """Show pose(now) in the organisers' sim, kinematically (None: nothing to show yet), until the
    window closes or, without one, until done()."""
    import mujoco
    from climbing_sim import ClimbingSim

    sim = ClimbingSim(organisers_wall(wall.id), cameras=[])  # their wall of that name
    model, data = sim.model, sim.data
    qpos = [model.jnt_qposadr[model.joint(name).id] for name in MICROBAN_JOINTS]

    def frame() -> None:
        q = pose(time.perf_counter())
        if q is not None:
            data.qpos[0:7] = q[0:7]  # the optimiser's q is laid out as the sim's qpos
            data.qpos[qpos] = q[7:]
            mujoco.mj_forward(model, data)

    if view:
        import mujoco.viewer

        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():  # keeps showing the last pose after the plan ends
                frame()
                viewer.sync()
                time.sleep(0.02)
    else:
        while not done():
            frame()
            time.sleep(0.02)


USAGE = """usage: tests/test_optimiser.py [PLAN] [--save | --replay] [--samples N] [--swing W]
                                [--keep-out W] [--hook W] [--send PI] [--headless]
    PLAN              the contact plan to solve (default build/plan_N1.json): each move is solved
                      and played in the organisers' sim as it comes
    --save            also save it in build/trajectory_<wall id>.json after every move
    --replay          play build/trajectory_<wall id>.json on the plan's wall, without solving
    --samples N       samples per move (TransitionConfig.n_samples, default 12); more take longer
    --swing W         how hard the moving ball is pulled along its path up, out, across, in and
                      down (TransitionConfig.w_swing, default 5000); 0 turns it off
    --keep-out W      how hard the body is pushed out of the cups' layer (TransitionConfig.
                      w_keep_out, default 20000); 0 turns it off
    --hook W          how hard each hook is turned to point up from its ball, so it goes into
                      the cup from above (TransitionConfig.w_hook, default 10); 0 turns it off
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
    cfg = tr.TransitionConfig()
    if "--samples" in args:
        i = args.index("--samples")
        try:
            cfg = replace(cfg, n_samples=int(args[i + 1]))
        except (IndexError, ValueError):
            sys.exit("--samples needs a number\n" + USAGE)
        del args[i : i + 2]
    if "--swing" in args:
        i = args.index("--swing")
        try:
            cfg = replace(cfg, w_swing=float(args[i + 1]))
        except (IndexError, ValueError):
            sys.exit("--swing needs a number (its weight; 0 turns it off)\n" + USAGE)
        del args[i : i + 2]
    if "--hook" in args:
        i = args.index("--hook")
        try:
            cfg = replace(cfg, w_hook=float(args[i + 1]))
        except (IndexError, ValueError):
            sys.exit("--hook needs a number (its weight; 0 turns it off)\n" + USAGE)
        del args[i : i + 2]
    if "--keep-out" in args:
        i = args.index("--keep-out")
        try:
            cfg = replace(cfg, w_keep_out=float(args[i + 1]))
        except (IndexError, ValueError):
            sys.exit("--keep-out needs a number (its weight; 0 turns it off)\n" + USAGE)
        del args[i : i + 2]
    if {"-h", "--help"} & set(args):
        sys.exit(USAGE)
    if "--hard" in args:  # the former way to ask for the runner: still accepted, not needed
        args.remove("--hard")
    plans = [a for a in args if not a.startswith("-")]
    if len(plans) > 1:
        sys.exit(f"one plan at a time, not {plans}\n" + USAGE)
    plan_file = Path(plans[0]) if plans else RUN_PLAN
    if plans:
        args.remove(plans[0])
    if not plan_file.exists():
        sys.exit(f"no plan at {plan_file}: plan a wall first (scripts/plan_sim_wall.py --save)")
    plan = io.load_contact_plan(plan_file)
    print(f"plan {plan_file}: wall {plan.wall_id}, {len(plan.stances) - 1} moves", flush=True)
    flags = set(args)
    if not flags <= {"--save", "--replay", "--headless"}:
        sys.exit(
            f"unknown options {sorted(flags - {'--save', '--replay', '--headless'})}\n" + USAGE
        )
    if send and "--replay" in flags:
        sys.exit("--send sends what is solved; to send a saved one: scripts/link_pi.py stream")
    if {"--save", "--replay"} <= flags:
        sys.exit("--save solves and saves, --replay plays what was saved: pick one\n" + USAGE)
    wall, view = plan_walls(plan), "--headless" not in flags
    if "--replay" in flags:
        replay(wall, view)
    else:
        run_plan(wall, plan, view, save="--save" in flags, send=send, cfg=cfg)
