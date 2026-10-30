"""Generate a wall with the simulator, plan it, and optionally show each stance in MuJoCo.

uv run scripts/plan_sim_wall.py --level 2                  # a random seed, printed
uv run scripts/plan_sim_wall.py --level 2 --seed 5 --save build/plan.json --show
uv run mjpython scripts/plan_sim_wall.py --level 2 --seed 5 --play   # the optimiser's motion
"""

import argparse
import random
import time
from collections import Counter
from itertools import pairwise
from pathlib import Path

import mujoco
from climbing_sim import ClimbingSim, Wall
from climbing_sim.routes import generate_wall
from numpy.linalg import norm

from rcr import io, optimiser
from rcr.planner import astar, feasible, plan
from rcr.types import LIMBS, ContactPlan, Hold, Stance, Trajectory, WallModel


def hold_id(hold) -> str:
    return f"r{hold.row}_c{hold.col}"


def to_wall_model(sim_wall, level: int, seed: int) -> WallModel:
    """Simulator wall (holds by col, row) -> ours (holds by position: the bolts' midpoint)."""
    holds = [Hold(hold_id(h), tuple(float(v) for v in h.bolt_axis)) for h in sim_wall.holds]
    start = Stance(contacts={limb: hold_id(h) for limb, h in sim_wall.start.items()})
    return WallModel(id=f"l{level}_s{seed}", level=level, holds=holds, start=start)


def why_stuck(wall: WallModel, last: Stance) -> None:
    """The moves from the last stance that would put a hand higher than it has been (a hand to a
    higher free hold), counted by why they fail, with one example of the values."""
    k = astar.key(last)
    z = {h.id: h.position[2] for h in wall.holds}
    highest = max(z[k[0]], z[k[1]])  # the hands, LIMBS order
    reasons, example = Counter(), {}
    for i in (0, 1):
        for h in wall.holds:
            if h.id not in k and z[h.id] > highest:
                reason = feasible(wall, last, astar.stance(k[:i] + (h.id,) + k[i + 1 :])).reason
                reason = reason or "feasible (already visited)"
                kind = reason.split(" (")[0]  # the values are in parentheses
                reasons[kind] += 1
                example.setdefault(kind, reason)
    print(f"the {sum(reasons.values())} moves that would put a hand higher, by reason:")
    for kind, n in reasons.most_common():
        print(f"  {n:4d}  {example[kind]}")


def show(sim_wall, wall: WallModel, contact_plan: ContactPlan) -> None:
    """Each stance's IK configuration in MuJoCo's viewer (no physics); Enter for the next."""
    sim = ClimbingSim(sim_wall, cameras=[])
    viewer = sim.view()
    for n, s in enumerate(contact_plan.stances):
        targets = astar.targets(wall, astar.key(s))
        _, q, _ = astar.ik(wall, astar.key(s))
        sim.data.qpos[:] = astar.qpos(sim.model, q)
        mujoco.mj_forward(sim.model, sim.data)
        viewer.sync()
        balls = sim.ball_positions()
        error = max(norm(balls[limb] - targets[i]) for i, limb in enumerate(LIMBS))
        prompt = f"stance {n}/{len(contact_plan.stances) - 1}: balls within {error * 1000:.1f} mm"
        if input(f"{prompt} of their sockets. Enter: next, q: quit ") == "q":
            break
    viewer.close()


def play(sim_wall, trajectory: Trajectory) -> None:
    """The trajectory in MuJoCo's viewer at real time, kinematically (no physics): the trunk pose
    and the joints by name, as the robot gets them."""
    sim = ClimbingSim(sim_wall, cameras=[])
    viewer = sim.view()
    joints = [sim.model.joint(name).qposadr[0] for name in trajectory.joint_names]
    clock = time.perf_counter()
    for t, pose, row in zip(
        trajectory.times, trajectory.trunk_poses, trajectory.positions, strict=True
    ):
        sim.data.qpos[0:7] = pose  # x, y, z, qw, qx, qy, qz: MuJoCo's order
        sim.data.qpos[joints] = row
        mujoco.mj_forward(sim.model, sim.data)
        viewer.sync()
        time.sleep(max(0.0, clock + t - time.perf_counter()))
    input("done. Enter: close ")
    viewer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--level", type=int, default=1, choices=(1, 2, 3, 4))
    parser.add_argument("--seed", type=int, help="random if not given")
    parser.add_argument("--save", help="save the contact plan to this JSON file")
    parser.add_argument("--show", action="store_true", help="show each stance in MuJoCo")
    parser.add_argument(
        "--play", action="store_true", help="the optimiser's motion of the plan, in MuJoCo"
    )
    parser.add_argument("--save-trajectory", help="save the optimiser's trajectory to this file")
    parser.add_argument(
        "--save-npz", help="save the optimiser's trajectory as a .npz (q: N x 26, every 0.02 s)"
    )
    parser.add_argument(
        "--heuristic-weight", type=float, default=astar.WEIGHT, help="A* heuristic weight"
    )
    parser.add_argument(
        "--wall",
        type=Path,
        help="load a wall from this JSON file instead of generating one. Options: N1, N2, N3",
    )
    args = parser.parse_args()

    seed = random.randrange(10_000) if args.seed is None else args.seed
    if args.wall:
        sim_wall = Wall.load(str(args.wall))  # a bundled name (N1, N2, N3) or a JSON file
        wall = to_wall_model(sim_wall, args.level, seed)
    else:
        sim_wall = generate_wall(args.level, seed)
        wall = to_wall_model(sim_wall, args.level, seed)
    print(f"wall: level {args.level}, seed {seed}, {len(wall.holds)} holds")

    clock = time.perf_counter()
    contact_plan = plan(wall, feasible, HEURISTIC_WEIGHT=args.heuristic_weight)
    seconds = time.perf_counter() - clock
    for a, b in pairwise(contact_plan.stances):
        (limb,) = [limb for limb in LIMBS if a.contacts[limb] != b.contacts[limb]]
        print(f"  {limb:<10} {a.contacts[limb]} -> {b.contacts[limb]}")
    z = {h.id: h.position[2] for h in wall.holds}
    hand = max(z[contact_plan.stances[-1].contacts[h]] for h in ("right_hand", "left_hand"))
    top = max(z.values())
    print(
        f"{len(contact_plan.stances) - 1} moves in {seconds:.1f} s; highest hand {hand:.3f} m, "
        f"top {top:.3f} m: {'reached' if hand == top else 'NOT reached'}"
    )
    if hand != top:
        why_stuck(wall, contact_plan.stances[-1])

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        io.save_contact_plan(contact_plan, args.save)
        print(f"saved {args.save}")
    if args.show:
        show(sim_wall, wall, contact_plan)
    if (args.play or args.save_trajectory or args.save_npz) and len(contact_plan.stances) > 1:
        clock = time.perf_counter()
        trajectory = optimiser.optimise(wall, contact_plan)
        seconds = time.perf_counter() - clock
        print(f"trajectory: {trajectory.times[-1]:.1f} s of motion, solved in {seconds:.1f} s")
        if args.save_trajectory:
            Path(args.save_trajectory).parent.mkdir(parents=True, exist_ok=True)
            io.save_trajectory(trajectory, args.save_trajectory)
            print(f"saved {args.save_trajectory}")
        if args.save_npz:
            Path(args.save_npz).parent.mkdir(parents=True, exist_ok=True)
            optimiser.convert_trajectory_to_npz(trajectory, args.save_npz)
            print(f"saved {args.save_npz}")
        if args.play:
            play(sim_wall, trajectory)


if __name__ == "__main__":
    main()
