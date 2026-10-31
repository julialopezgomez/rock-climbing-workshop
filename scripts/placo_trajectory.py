# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["placo>=0.10.2", "numpy", "microban-model", "climbing-sim", "static-balance"]
# [tool.uv.sources]
# climbing-sim = { path = "../third_party/simulator", editable = true }
# microban-sim = { git = "https://github.com/Rhoban/microban_sim", rev = "29234e369e33411ff42276096c6c6af208ad3b65" }  # noqa: E501
# microban-model = { git = "https://github.com/Rhoban/microban_model", rev = "8e42ef70673b5747dbdca3efaf60bc0f8ec7ea6d", lfs = true }  # noqa: E501
# static-balance = { git = "https://github.com/stephane-caron/rock-climbing-robots-tutorial", subdirectory = "static_balance", rev = "913fdede5f8765c9fa93b8942c0f51a13e82f6b8" }  # noqa: E501
# ///
"""A contact plan -> the kinematic climbing motion by placo (rcr.kinematics.placo_ik).

    uv run scripts/placo_trajectory.py build/plan_N3.json --wall N3             # compare, save
    uv run scripts/placo_trajectory.py build/plan_N3.json --wall N3 --generate  # placo+CoM's
    uv run scripts/placo_trajectory.py build/plan_N3.json --wall N3 --play      # ... and replay

placo pins pinocchio 3.8 and the project 4.x, so this script runs in its own environment (the
block above; uv builds it on first use), with the repository's src/ on the path: the project's
environment, the planner and the CasADi optimiser are untouched. Input and output are those of
scripts/plan_sim_wall.py's IK motion: a contact plan (its wall: --wall for a bundled one, else
rebuilt from the plan's id l<level>_s<seed>), and the trajectory saved as
build/trajectory_<plan>_placo[_com].json (rcr.io) and .npz (q: N x 26 at 50 Hz,
kinematics.convert_trajectory_to_npz).

First it compares, move by move, three motions of the same moving-ball path: Julia's IK (ik.py),
placo tracking it, and placo also shaping the body's motion (--generate's: the CoM into the
static-equilibrium polygon of the held balls before the release, and kept there). For each: the
CoM's margin to that polygon while the limb is in the air (Stéphane Caron's static_balance) and a
physics test in the organisers' sim (servo gain KP_FW, gravity): slot-in, miss or fall.
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rcr import io  # noqa: E402
from rcr.kinematics import convert_trajectory_to_npz, ik  # noqa: E402
from rcr.types import ContactPlan, Hold, Stance, Trajectory, WallModel  # noqa: E402


def plan_wall(plan: ContactPlan, name: str | None = None):
    """(our wall, the simulator's) for the plan, holds named r<row>_c<col> as
    scripts/plan_sim_wall.py names them: the simulator's wall `name` (N1, N2, N3 or a JSON file)
    if given, else generated from the plan's id l<level>_s<seed>. plan_sim_wall.py --wall N3 also
    names its plan l<level>_s<seed>, so a plan of a bundled wall needs `name`: rebuilt from the id,
    the wall would be another one (validate_plan then names the holds it lacks)."""
    from climbing_sim import Wall
    from climbing_sim.routes import generate_wall

    seeded = re.fullmatch(r"l(\d)_s(\d+)", plan.wall_id)
    if name is None and seeded is None:
        return io.load_wall(ROOT / "data/examples" / f"{plan.wall_id}.json"), Wall.load(
            plan.wall_id
        )
    level = int(seeded[1]) if seeded else 3
    sim_wall = Wall.load(name) if name else generate_wall(level, int(seeded[2]))
    holds = [
        Hold(f"r{h.row}_c{h.col}", tuple(float(v) for v in h.bolt_axis)) for h in sim_wall.holds
    ]
    start = Stance({limb: f"r{h.row}_c{h.col}" for limb, h in sim_wall.start.items()})
    return WallModel(id=plan.wall_id, level=level, holds=holds, start=start), sim_wall


KP_FW = (
    1000.0  # the servos' gain in the physics test [the IK motion climbed N1 on the robot at 1000]
)
HOOK = 0.006  # [m] a ball this close to a socket sits in it [simulator hooked()]
HOLD_S = 0.5  # [s] the physics test holds the first pose before the move, and the last after
# the physics test's stand-in for the servos' I gain: ki = 2 /s, capped at 10 deg, lifted playback
# of the N1 IK motion at kp_fw 1000 from 47 to 63 slot-ins in 100 (hands 0 -> 18 of 39) [Ours]
KI_CAP = np.radians(10.0)


def signed_margin(point: np.ndarray, polygon) -> float:
    """[m] distance from point (x, y) to the polygon's edge (counterclockwise vertices): positive
    inside, negative outside, NaN without a polygon."""
    if polygon is None:
        return float("nan")
    a, b = polygon, np.roll(polygon, -1, axis=0)
    edge = b - a
    s = np.clip(np.einsum("ij,ij->i", point - a, edge) / np.einsum("ij,ij->i", edge, edge), 0, 1)
    distance = np.linalg.norm(point - (a + s[:, None] * edge), axis=1).min()
    inside = np.all(edge[:, 0] * (point[1] - a[:, 1]) - edge[:, 1] * (point[0] - a[:, 0]) >= 0)
    return float(distance if inside else -distance)


def measures(wall: WallModel, a: Stance, b: Stance, qs: list) -> dict:
    """One move's motion, kinematically: the held balls' worst drift from their sockets [mm], the
    fastest joint against its speed limit, the closest any joint gets to a limit [rad], and the
    CoM's smallest margin [mm] to the support polygon of the three held balls while the moving
    ball is in the air (in neither socket)."""
    import pinocchio as pin

    from rcr.kinematics import placo_ik

    i = ik.LIMBS.index(ik.moving_limb(a, b))
    held = [j for j in range(4) if j != i]
    s_a, s_b = ik.sockets(wall, a), ik.sockets(wall, b)
    polygon = placo_ik.support_polygon(s_a[held])
    data, drift, margins = ik.model.createData(), 0.0, []
    for q in qs:
        balls = ik.balls(q)
        drift = max(drift, np.linalg.norm(balls[held] - s_a[held], axis=1).max())
        if min(np.linalg.norm(balls[i] - s_a[i]), np.linalg.norm(balls[i] - s_b[i])) > HOOK:
            margins.append(signed_margin(pin.centerOfMass(ik.model, data, q)[:2], polygon))
    joints = np.array([q[7:] for q in qs])
    speed = (np.abs(np.diff(joints, axis=0)) / ik.DT / ik.SPEED_LIMIT).max()
    lower, upper = ik.model.lowerPositionLimit[7:], ik.model.upperPositionLimit[7:]
    return {
        "drift": drift * 1000,
        "speed": speed,
        "limit": np.minimum(joints - lower, upper - joints).min(),
        "com": 1000 * min(margins) if margins else float("nan"),
        "seconds": len(qs) * ik.DT,
    }


def physics(sim, wall: WallModel, a: Stance, b: Stance, qs: list, ki: float = 0.0) -> str:
    """The move in the organisers' sim with gravity: placed in its first pose, held there HOLD_S,
    then the joints sent qs at 50 Hz (the robot's servo targets), then the last pose held HOLD_S.
    ki [1/s]: an integral of each joint's error added to its target (capped at KI_CAP), standing
    in for the robot servos' own I gain, which the sim lacks (it models the P gain only).
    "slot-in": every ball in b's socket; "fall": a held ball out of its socket; "miss": the
    moving ball not in its new socket, how far; "no start": the first pose not held in the sim."""
    from climbing_sim import JOINTS as SIM_JOINTS

    sockets = sim.wall.sockets()
    integral = np.zeros(len(SIM_JOINTS))

    def send(target):
        nonlocal integral
        error = target - sim.joint_positions()
        integral = np.clip(integral + ki * ik.DT * error, -KI_CAP, KI_CAP)
        sim.step(target + integral)

    def expected(stance):
        return {limb: int(np.argmin(np.linalg.norm(sockets - p, axis=1)))
                for limb, p in zip(ik.LIMBS, ik.sockets(wall, stance), strict=True)}  # fmt: skip

    order = [ik.model.idx_qs[ik.model.getJointId(name)] for name in SIM_JOINTS]
    q0 = qs[0]
    joints = {name: float(q0[j]) for name, j in zip(SIM_JOINTS, order, strict=True)}
    sim.reset(q0[0:3], q0[[6, 3, 4, 5]], joints)  # pinocchio (x, y, z, w) -> (w, x, y, z)
    for _ in range(round(HOLD_S / ik.DT)):
        send(q0[order])
    before, after = expected(a), expected(b)
    if sim.hooked() != before:
        return "no start"
    for q in [*qs, *[qs[-1]] * round(HOLD_S / ik.DT)]:
        send(q[order])
    hooked = sim.hooked()
    if hooked == after:
        return "slot-in"
    moving = ik.moving_limb(a, b)
    if any(hooked[limb] != before[limb] for limb in ik.LIMBS if limb != moving):
        return "fall"
    off = np.linalg.norm(sim.ball_positions()[moving] - sockets[after[moving]])
    return f"miss {off * 1000:.0f}"


def compare(wall: WallModel, sim_wall, plan: ContactPlan, ki: float = 0.0) -> None:
    """Each move made three ways, each from its own stance pose of a: Julia's IK (ik.py), placo
    tracking the same path, and placo also shaping the motion (generate). Per move and in total:
    whether the motion exists, its kinematic measures, and the physics test."""
    from climbing_sim import ClimbingSim

    from rcr.kinematics import placo_ik, placo_motion

    # ik.py at fe95f66 uses SPEED_MARGIN without defining it; its README gives 0.8: set here so its
    # blend runs, without editing it
    ik.SPEED_MARGIN = getattr(ik, "SPEED_MARGIN", 0.8)
    ways = {
        "ik.py": (ik.track_move, ik.start_pose, {}),
        "placo": (placo_ik.track_move, placo_ik.stance_pose, {}),
        "placo+CoM": (placo_ik.track_move, placo_ik.stance_pose, {"generate": True}),
        "placo route": (
            lambda wall, a, b, q0: placo_motion.generate_move(wall, sim_wall, a, b, q0),
            placo_ik.stance_pose,
            {},
        ),
    }
    sim = ClimbingSim(sim_wall, cameras=[], kp_fw=KP_FW)
    totals = {name: [] for name in ways}
    print(
        f"per move (physics at kp_fw {KP_FW:g}, ki {ki:g}): result [miss: mm from the socket]"
        " | CoM margin in the air | fastest joint vs its limit"
    )
    print(f"{'move':>4}  {'limb':10s}" + "".join(f"  | {name:28s}" for name in ways))
    for n, (a, b) in enumerate(zip(plan.stances, plan.stances[1:], strict=False)):
        row = f"{n:4d}  {ik.moving_limb(a, b):10s}"
        for name, (track, start, options) in ways.items():
            try:
                with np.errstate(over="ignore", invalid="ignore"):
                    q0 = start(wall, a)
                    qs = [q0, *track(wall, a, b, q0, **options)[0]]
                m, result = measures(wall, a, b, qs), physics(sim, wall, a, b, qs, ki)
                totals[name].append((result, m))
                row += f"  | {result:8s} {m['com']:+6.1f} mm  {m['speed']:4.2f}x       "
            except Exception as e:  # noqa: BLE001  no motion: report why and go on
                totals[name].append(("no motion", None))
                row += f"  | no motion ({type(e).__name__})".ljust(31)
        print(row, flush=True)
    print("\ntotal over the plan's moves:")
    for name, runs in totals.items():
        results = [r for r, _ in runs]
        made = [m for _, m in runs if m is not None]
        com = np.array([m["com"] for m in made])
        print(
            f"  {name:10s} motion for {len(made)}/{len(runs)}; physics: "
            + ", ".join(
                f"{k} {sum(r.startswith(k) for r in results)}"
                for k in ("slot-in", "miss", "fall", "no start")
            )
            + (
                f"; CoM margin in the air: median {np.nanmedian(com):+.1f} mm, outside the "
                f"polygon on {int(np.sum(com < 0))} moves"
                if made
                else ""
            )  # fmt: skip
        )


def play(sim_wall, trajectory: Trajectory) -> None:
    """The trajectory in MuJoCo's viewer at real time, kinematically (no physics), as
    scripts/plan_sim_wall.py plays it."""
    import mujoco
    from climbing_sim import ClimbingSim

    sim = ClimbingSim(sim_wall, cameras=[])
    viewer = sim.view()
    joints = [sim.model.joint(name).qposadr[0] for name in trajectory.joint_names]
    clock = time.perf_counter()
    for t, pose, row in zip(
        trajectory.times, trajectory.trunk_poses, trajectory.positions, strict=True
    ):
        sim.data.qpos[0:7] = pose
        sim.data.qpos[joints] = row
        mujoco.mj_forward(sim.model, sim.data)
        viewer.sync()
        time.sleep(max(0.0, clock + t - time.perf_counter()))
    input("done. Enter: close ")
    viewer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("plan", type=Path, help="a contact plan, e.g. build/plan_N3.json")
    parser.add_argument("--wall", help="the bundled wall it was planned on (N1, N2, N3)")
    parser.add_argument("--out", type=Path, help="default: build/trajectory_<plan>_placo.json")
    parser.add_argument("--generate", action="store_true", help="save placo+CoM's motion")
    parser.add_argument("--route", action="store_true", help="save placo route's motion")
    parser.add_argument("--ki", type=float, default=0.0, help="physics test: I gain, 1/s")
    parser.add_argument("--no-compare", action="store_true", help="skip the per-move comparison")
    parser.add_argument("--play", action="store_true", help="replay it in MuJoCo's viewer")
    args = parser.parse_args()
    if args.play and sys.platform == "darwin" and "MJPYTHON_BIN" not in os.environ:
        mjpython = Path(sys.executable).parent / "mjpython"  # macOS: the viewer needs mjpython
        os.execv(mjpython, [str(mjpython), __file__, *sys.argv[1:]])

    from rcr.kinematics import placo_ik

    plan = io.load_contact_plan(args.plan)
    wall, sim_wall = plan_wall(plan, args.wall)
    print(f"{args.plan}: wall {args.wall or wall.id}, {len(plan.stances) - 1} moves")
    if not args.no_compare:
        compare(wall, sim_wall, plan, args.ki)
    clock = time.perf_counter()
    if args.route:
        from rcr.kinematics import placo_motion

        trajectory = placo_motion.optimise(wall, sim_wall, plan)
    else:
        trajectory = placo_ik.optimise(wall, plan, args.generate)
    seconds = time.perf_counter() - clock
    suffix = "_placo_route" if args.route else "_placo_com" if args.generate else "_placo"
    out = args.out or ROOT / "build" / f"trajectory_{args.plan.stem}{suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    io.save_trajectory(trajectory, out)
    convert_trajectory_to_npz(trajectory, str(out.with_suffix(".npz")))
    print(
        f"{suffix[1:]}: {trajectory.times[-1]:.1f} s of motion in {seconds:.1f} s -> {out} and .npz"
    )
    if args.play:
        play(sim_wall, trajectory)


if __name__ == "__main__":
    main()
