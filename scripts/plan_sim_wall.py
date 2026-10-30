"""Generate a wall with the simulator, plan it, and optionally show each stance in MuJoCo.

uv run scripts/plan_sim_wall.py --level 2                  # a random seed, printed
uv run scripts/plan_sim_wall.py --level 2 --seed 5 --save build/plan.json --show
"""

import argparse
import random
import time
from collections import Counter
from itertools import pairwise
from pathlib import Path

import mujoco
from climbing_sim import ClimbingSim
from climbing_sim.routes import generate_wall
from numpy.linalg import norm

from rcr import io
from rcr.planner import astar, feasible, plan
from rcr.types import LIMBS, ContactPlan, Hold, Stance, WallModel


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
        _, q = astar.ik(wall, astar.key(s))
        sim.data.qpos[:] = astar.qpos(sim.model, q)
        mujoco.mj_forward(sim.model, sim.data)
        viewer.sync()
        balls = sim.ball_positions()
        error = max(norm(balls[limb] - targets[i]) for i, limb in enumerate(LIMBS))
        prompt = f"stance {n}/{len(contact_plan.stances) - 1}: balls within {error * 1000:.1f} mm"
        if input(f"{prompt} of their sockets. Enter: next, q: quit ") == "q":
            break
    viewer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--level", type=int, default=1, choices=(1, 2, 3, 4))
    parser.add_argument("--seed", type=int, help="random if not given")
    parser.add_argument("--save", help="save the contact plan to this JSON file")
    parser.add_argument("--show", action="store_true", help="show each stance in MuJoCo")
    parser.add_argument("--heuristic-weight", type=float, default=astar.WEIGHT, help="A* heuristic weight")
    args = parser.parse_args()

    seed = random.randrange(10_000) if args.seed is None else args.seed
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


if __name__ == "__main__":
    main()
