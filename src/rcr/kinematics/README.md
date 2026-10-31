# optimiser

**Purpose:** turn the contact plan into the joint motion the robot actually executes, and say which transitions
are possible at all.

| | |
|---|---|
| Input | `WallModel`, the full `ContactPlan` |
| Output | `optimise(wall, plan) -> Trajectory`: the **full** joint trajectory. `feasibility(wall, a, b) -> Feasibility`: is stance a → b doable, at what cost |
| Runs on | laptop (never on the Pi) |

The optimiser owns `feasibility` because "can the robot actually do this transition?" is a control question. The
planner receives `rcr.optimiser.feasibility` as an argument and never imports this module.

## Method (D3): a kinematic motion, IK tracking the moving ball [Ours]
Started over from v0 (below) to keep it minimal. No torques, no servo-sag pre-load, no collisions and no cache yet:
first a kinematic motion, then we see. All in `ik.py`.
- **The ball's path, per move:** only the moving limb's ball has one, from its socket on hold a to its socket on
  hold b (`ball_path`). It's a corridor clear of every hold: five straight pieces, each a degree-1 Bézier
  (`bezier.py`), at one constant speed (the corridor's length / `MOVE_TIME`). Each piece lasts its share of the
  length. The speed is the same everywhere, but the direction turns at each corner, and the ball starts and stops at
  full speed. The pieces are straight, so the path can't cut a corner into a hold or rise above where it's going:
  - **up** `ABOVE` out of socket a. A ball leaves a socket only within ~10° of vertical [simulator `wall.py`,
    `EXIT_BLOCK`];
  - **out** `OUT` from the wall;
  - **across** to in front of socket b;
  - **in** to above socket b;
  - **down** into socket b.
- **The IK step is a QP** (`ik_step_qp`, solved with `quadprog` through the ediadvancedrobotics tutorials' helper
  `quadprog_solve_qp`). Each step chooses the change of configuration v that:
  - **meets the four balls exactly:** equality constraints `J v = displacement`. The held balls stay in their
    sockets and the moving ball is on its path. If no step can, the move is infeasible;
  - **minimises** the joints' distance to a reference, `‖S v − reference‖²`. During a move the reference goes from
    a's stance pose to b's at constant joint speed; in `start_pose` it's a step up the manipulability gradient. A
    `1e-6 ‖v‖²` term keeps the QP strictly convex;
  - **must stay within** the joint limits (one Euler step) and, during a move, the URDF's speed limits from the
    sample before (6.6 rad/s over `DT`);
  - **tries to stay** `JOINT_MARGIN` inside the joint limits and under `SPEED_MARGIN` of the speed limits, through
    slack variables ≥ 0 penalised by `SLACK_WEIGHT`. In `start_pose` the joint margin is hard (`hard_margin`).
- **`ik_step_pinv`, the previous IK:** the whole body follows by the planner's hierarchical IK (`astar.ik`, copied here, not imported). The held
  balls' tasks come first, in `TASK_ORDER`, then the moving ball's, each in the null space of the ones before, then
  the trunk upright and facing the wall. At each sample, `ITERATIONS` steps each close `GAIN` of every ball's error
  to its target. The moving ball's target is on the path; the held balls' targets are their sockets.
- **Out of reach** [Ours]: the held balls are solved exactly (their sockets are always in reach). The moving ball
  and the trunk use a damped least-squares inverse (`DAMPING`), so where the path is out of the limb's reach the
  moving ball only gets as close as it can, while the held balls stay in their sockets. With plain `pinv` for all
  and a fixed order, long hand moves with a big `OUT` / `ABOVE` diverged (NaN) or pulled a held hand out.
- **Joint limits:** a joint the step would push past a limit is locked there and the step solved again without it,
  then the joints are clipped (`projecttojointlimits`, as in the planner). With the clipping alone, the ankles and
  the shoulder roll saturate and a held ball drifted up to 10 mm off its socket on a level-2 climb.
- **Stance pose** (`start_pose`, cached per stance):
  1. the hang pose with its trunk fitted to the stance's sockets;
  2. IK until the balls are in their sockets;
  3. with the balls held exactly, steps up the gradient of the four limbs' sum of `log w` (manipulability,
     `MANIPULABILITY_STEP`), until a step is under `SETTLED` or `POSE_ITERATIONS`.

  **Every move starts exactly at a's pose and ends exactly at b's:** the joint reference arrives at b's pose and
  the balls at b's sockets. At the last sample the QP steps until it settles, so the next move starts where this
  one ended. There is no blend. If a move ends more than `ARRIVAL_TOL` from b's pose, it's infeasible.
- **Output:** `optimise(wall, plan) -> Trajectory`: the start pose, then every move's samples every `DT`, each move
  from where the previous one ended. `positions` = `planned_positions` = the IK's joint angles (kinematic);
  `trunk_poses` for replaying it.
- **`feasible_ik_traj(wall, a, b)`**: `track_move` from a's stance pose. That is the move `optimise` makes, so the
  verdict is cached per (a, b). `scripts/plan_sim_wall.py --feasibility ik_traj` runs the planner's
  `feasible_ik_contact` (reach, IK, collision of stance b) first, then this.

| Parameter | Value | |
|---|---|---|
| `ABOVE` | 0.025 m | up out of socket a before moving out; above socket b before coming down |
| `OUT` | 0.04 m | in front of the sockets while going across (66 mm from the wall) |
| `MOVE_TIME`, `DT` | 2.0 s, 0.02 s | per move; between samples (the robot's 50 Hz loop) |
| `MANIPULABILITY_STEP`, `POSE_ITERATIONS`, `SETTLED` | 0.05, 500, 1e-5 | the stance pose's ascent |
| `ARRIVAL_TOL` | 1e-4 rad | a move ending further from b's pose is infeasible (measured: 0) |
| `JOINT_MARGIN`, `SPEED_MARGIN`, `SLACK_WEIGHT` | 0.1 rad, 0.8, 10 | the QP: how far inside its limits it tries to stay, and at what cost |
| `GAIN`, `ITERATIONS` | 0.5, 3 | fraction of each error (and of the trunk's tilt) closed per IK step; steps per sample |
| `DAMPING` | 0.03 | the moving ball's and the trunk's damped inverse |
| `MAX_ERROR` | 0.01 m | a held ball further from its socket: `ValueError` naming the move |

Measured on the simulator's collision shapes (ball radius 12.5 mm, holds 43 mm out of the wall), on the 440 hold → hold
paths of the 1113 moves A* asks about on L1 s1, L2 s5226, s7 and s66102, and L3 s1 and s2:
- **The corridor:** clears every hold by at least 6.3 mm, both as planned and as the IK tracks it. 1110 of the 1113
  moves are doable, and it never rises above `ABOVE` over the higher socket.
- **`ABOVE`:**
  - at 25 mm, moving out clears the ball's own hold by 7.9 mm (at 20 mm, only 2.9);
  - it also clears a hold 60 mm above and 15 mm beside (the routes allow one there) by 8.5 mm (at 30 mm, only 3.8).
- **`OUT`:** at 40 mm, the ball goes across 10.5 mm in front of every hold (at 35 mm, only 5.5).
- **Any single curve over** (up, one Bézier over, down) cut its corner into the next hold, by up to 14 mm. That held
  for every `ABOVE` from 30 to 45 mm, `OUT` 40 or 60 mm, and junction speed.

**The QP IK with the stance poses**, on 36 moves (12 from each of L2 s66102, L2 s5226 and L3 s1):
- **Moves:** all 36 doable, each starting exactly at a's pose and ending exactly at b's. The gap before snapping to
  b's pose is 0.
- **Tracking and speed:** the moving ball stays within 2.2 mm of its path. The fastest joint uses 0.36× its speed
  limit, and no joint gets closer than 0.039 rad to a limit.
- **Stance poses:** the ascent raises the four limbs' sum of `log w` by about 0.45 (their product ×1.6).
- **Joint velocity:** the largest change between samples is 1.85 rad/s, at the corners and at the start and end of
  a move: the ball goes at constant speed and turns sharply at the corridor's corners.
- **Time:** 0.32 s a move.

```bash
uv run mjpython scripts/plan_sim_wall.py --level 2 --seed 5226 --heuristic-weight 1.5 --play  # see it
uv run scripts/plan_sim_wall.py --level 2 --seed 5226 --save-trajectory build/trajectory.json
uv run scripts/plan_sim_wall.py --level 2 --seed 5 --save-npz seed5  # npz/<timestamp>_seed5.npz: q N x 26, 50 Hz
```

## The CasADi optimiser
Quasi-static direct transcription, one NLP per transition, CasADi + IPOPT, on Microban's URDF: in
[`rcr.optimiser`](../optimiser/), described in [its README](../optimiser/README.md).
```bash
uv sync --group optimiser --group sim
uv run pytest tests/test_optimiser.py
uv run mjpython tests/test_optimiser.py PLAN    # solve a contact plan, play it in the sim
```

## Open decisions
- Collisions with the holds, the wall and itself (none checked yet; the path's `OUT`/`ABOVE` are the only clearance).
- Torques: whether the motion can be held (0.605 N·m limit), and the sag pre-load of `positions`.
- What `feasibility` checks.
- Time per move (time is the tie-breaker in scoring).
- Whether `Trajectory` needs velocities, per-joint kp or the stance index (marked CHANGE in INTERFACES.md).
