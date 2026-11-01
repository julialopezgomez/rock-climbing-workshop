"""Extend a contact plan from its last stance to the top of its wall, and check each new move.

    uv run --group optimiser scripts/extend_plan.py data/results/to_n2/plan_N2.json \
        data/results/to_n2/plan_N2_full.json --rest data/results/to_n2/plan_N2_rest.json

The new moves are the planner's (rcr.planner.astar: reach, IK on each stance, collisions), with
two of its limits a little looser [Ours]: --max-level (one hand above the other, one foot above
the other; 0.12 m, as before fd3da59 set 0.10) and --max-step (a limb's move; 0.105 m, the
simulator routes' longest step, astar.STEP, against 0.09). With the planner's own limits, N2
(seed 66102) stops at 0.985 m. Big steps failed on the real robot, so the moves are not
astar.solve's (the fewest moves) but the smallest [Ours]: the fewest steps longer than --big
(0.09 m, the longest in plan_N2.json's 34 moves), then the fewest moves, then the least sum of
squared steps (small_steps). On N2 (seed 66102) a few 105 mm steps stay: rows 72 and 79 are
105 mm apart with one hold between them (r75_c24).

Each new move a -> b is then checked statically, without solving the motion [Ours]: the
optimiser's balanced pose of a and of b (tr.stance_pose), and from each a balanced pose with the
moving limb unloaded, at its release (on a's holds) and at its landing (on b's), with the three
other balls at most --max-force each (Caron's tutorial: 5 N, as its contacts bound it,
|fx| + |fy| + fz, so the CoM is then in its polygon) and every motor torque at most
--torque-share of run.sh's limit (0.9 of the servo's). A move that fails is never taken again and
the planner plans again. Also printed: the CoM's margin to Caron's static-equilibrium polygon of the
three held balls (scripts/compare_com.py) in those poses.
"""

import argparse
import heapq
import sys
from dataclasses import replace
from itertools import count
from pathlib import Path

import casadi as ca
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_com import FRICTION_DEG, polygon, signed_margin  # noqa: E402
from test_optimiser import plan_walls  # noqa: E402  the wall the plan is on

from rcr import io  # noqa: E402
from rcr.optimiser import robot_model as rm  # noqa: E402
from rcr.optimiser import transition as tr  # noqa: E402
from rcr.planner import astar  # noqa: E402
from rcr.types import LIMBS, ContactPlan  # noqa: E402

TAU_LIMIT = 0.9  # run.sh's, as a fraction of the servo's limit
W_HOOK = 100.0  # run.sh's (the other weights it sets are the defaults)
MAX_ATTEMPTS = 40  # plans tried, each without the moves that failed before


def unloaded(wall, stance, limb, q_ref, cfg, model, max_force):
    """A balanced pose on the stance's holds, near q_ref, with `limb` carrying nothing and each
    other ball at most max_force: (largest torque / cfg.tau_limit, CoM margin [m]).
    tr.OptimisationError if there is none."""
    opti = ca.Opti()
    dq = opti.variable(rm.NV, 1)
    forces = {other: opti.variable(3, 1) for other in LIMBS}
    q, tau = tr.balanced_pose(opti, model, dq[:, 0], q_ref, forces, cfg)
    opti.subject_to(model.limb_positions(q) == tr.contact_points(wall, stance, cfg))
    for other in LIMBS:
        if other == limb:
            opti.subject_to(forces[other] == 0)
            continue
        tr.friction_pyramid(opti, forces[other], cfg.normals[other], cfg.mu[other])
        for sx in (-1, 1):  # |fx| + |fy| + fz <= max_force, as Caron's contacts [static_balance]
            for sy in (-1, 1):
                f = forces[other]
                opti.subject_to(sx * f[0] + sy * f[1] + f[2] <= max_force)
        opti.set_initial(forces[other], tr.weight_share(len(LIMBS) - 1))
    cost = cfg.w_tau * ca.sumsqr(tau) + cfg.w_stay * ca.sumsqr(dq) + tr.posture(q, cfg)
    cost += cfg.w_keep_out * tr.intrusion(model, q, cfg) + cfg.w_hook * tr.hook_tilt(model, q, cfg)
    opti.minimize(cost)
    options = {"print_level": 0, "sb": "yes", "max_iter": 1000}
    opti.solver("ipopt", {"print_time": False, "expand": True}, options)
    sol, _ = tr.solve_opti(opti, f"{limb} unloaded", patience=3)
    q_value, tau_value = np.array(sol.value(q)).ravel(), np.array(sol.value(tau)).ravel()
    balls = np.array(model.limb_positions(q_value)).T
    held = [i for i, other in enumerate(LIMBS) if other != limb]
    com = np.array(model.com(q_value)).ravel()
    friction = float(np.tan(np.radians(FRICTION_DEG)))
    margin = signed_margin(com[:2], polygon(balls[held], model.mass, friction, max_force))
    return np.abs(tau_value).max() / cfg.tau_limit, margin


def small_steps(wall, start, failed, big):
    """The planner's moves from `start` to a hand on a top hold (each stance within reach, each
    step at most astar.MAX_TRANSITION_DISTANCE, IK on each stance), never a move in `failed`, at
    the least cost: (steps longer than `big`, moves, sum of squared steps [m^2]), compared in that
    order. A ContactPlan, or None if the top cannot be reached."""
    position = {h.id: np.array(h.position) for h in wall.holds}
    top = max(h.position[2] for h in wall.holds)
    on_top = {h.id for h in wall.holds if h.position[2] == top}
    k0 = astar.key(start)
    best, parent, tie = {k0: (0, 0, 0.0)}, {}, count()
    heap = [((0, 0, 0.0), next(tie), k0)]
    while heap:
        cost, _, k = heapq.heappop(heap)
        if cost > best[k]:
            continue
        if on_top & set(k[:2]):  # the hands, LIMBS order
            path = [k]
            while path[-1] in parent:
                path.append(parent[path[-1]])
            return ContactPlan(wall.id, [astar.stance(s) for s in reversed(path)])
        for i in range(len(LIMBS)):
            for hold in position:
                if hold in k:
                    continue
                child = k[:i] + (hold,) + k[i + 1 :]
                step = float(np.linalg.norm(position[hold] - position[k[i]]))
                if step > astar.MAX_TRANSITION_DISTANCE or (k, child) in failed:
                    continue
                new = (cost[0] + (step > big), cost[1] + 1, cost[2] + step**2)
                if new >= best.get(child, (np.inf,)):
                    continue
                if astar.reach(astar.targets(wall, child)) is not None:
                    continue
                if not astar.feasible(wall, astar.stance(k), astar.stance(child)).feasible:
                    continue
                best[child], parent[child] = new, k
                heapq.heappush(heap, (new, next(tie), child))
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("plan", type=Path, help="the plan to extend")
    parser.add_argument("out", type=Path, help="the whole plan, extended")
    parser.add_argument("--rest", type=Path, help="also the new moves alone, from the last stance")
    parser.add_argument("--max-level", type=float, default=0.12, help="[m] astar.MAX_LEVEL")
    parser.add_argument("--max-step", type=float, default=0.105, help="[m] a limb's move")
    parser.add_argument("--big", type=float, default=0.09, help="[m] a step longer is big")
    parser.add_argument("--max-force", type=float, default=5.0, help="[N] a held ball's most")
    parser.add_argument("--torque-share", type=float, default=0.8, help="of run.sh's limit")
    args = parser.parse_args()

    plan = io.load_contact_plan(args.plan)
    wall = plan_walls(plan)
    astar.MAX_LEVEL, astar.MAX_TRANSITION_DISTANCE = args.max_level, args.max_step
    cfg = replace(tr.TransitionConfig(), tau_limit=TAU_LIMIT * tr.SERVO_TORQUE_LIMIT, w_hook=W_HOOK)
    model = rm.load_robot_model()
    position = {h.id: np.array(h.position) for h in wall.holds}
    z = {h.id: h.position[2] for h in wall.holds}
    first = len(plan.stances) - 1  # the first new move's index in the whole plan

    checked, failed = {}, set()

    def check(i, a, b):
        """(passed, what was found) for move i, a -> b; cached."""
        k = (astar.key(a), astar.key(b))
        if k not in checked:
            limb = tr.moving_limb(a, b, i)
            try:
                qa, qb = tr.stance_pose(wall, a, cfg, i), tr.stance_pose(wall, b, cfg, i + 1)
                ta, ma = unloaded(wall, a, limb, qa, cfg, model, args.max_force)
                tb, mb = unloaded(wall, b, limb, qb, cfg, model, args.max_force)
                text = (
                    f"release: torque {ta:4.2f}x, CoM {ma * 1000:+5.1f} mm | "
                    f"landing: torque {tb:4.2f}x, CoM {mb * 1000:+5.1f} mm"
                )
                checked[k] = (max(ta, tb) <= args.torque_share, text)
            except tr.OptimisationError as e:
                checked[k] = (False, f"no balanced pose: {e}")
        return checked[k]

    for attempt in range(MAX_ATTEMPTS):
        ext = small_steps(wall, plan.stances[-1], failed, args.big)
        if ext is None:
            sys.exit(f"no plan to the top ({max(z.values()):.3f} m) without the moves that failed")
        lines, bad = [], None
        for n, (a, b) in enumerate(zip(ext.stances, ext.stances[1:], strict=False)):
            limb = tr.moving_limb(a, b, first + n)
            step = np.linalg.norm(position[b.contacts[limb]] - position[a.contacts[limb]])
            passed, text = check(first + n, a, b)
            lines.append(
                f"move {first + n:2d} {limb:10s} {a.contacts[limb]:>7s} -> "
                f"{b.contacts[limb]:<7s} step {step * 1000:3.0f} mm | {text}"
            )
            if not passed:
                bad = (a, b)
                break
        last = ext.stances[-1].contacts
        hand = max(z[last["right_hand"]], z[last["left_hand"]])
        print(
            f"plan {attempt}: {len(ext.stances) - 1} new moves, a hand at {hand:.3f} m", flush=True
        )
        if bad is None:
            break
        print(f"  failed: {lines[-1]}", flush=True)
        failed.add((astar.key(bad[0]), astar.key(bad[1])))
    print(*lines, sep="\n")
    if bad is not None or hand != max(z.values()):
        sys.exit(f"no checked plan to the top ({max(z.values()):.3f} m)")
    whole = ContactPlan(plan.wall_id, plan.stances + ext.stances[1:])
    io.save_contact_plan(whole, args.out)
    print(f"saved {args.out}: {len(whole.stances) - 1} moves, a hand on the top hold")
    if args.rest:
        io.save_contact_plan(ext, args.rest)
        print(f"saved {args.rest}: moves {first}-{len(whole.stances) - 2}, from stance {first}")


if __name__ == "__main__":
    main()
