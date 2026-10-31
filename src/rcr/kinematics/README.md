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
  hold b (`ball_path`). It's a corridor clear of every hold, five Bézier curves in time (`bezier.py`) through its
  corners, each at rest at both ends. Their control points come from the states at both ends [C-CROC, Fernbach et
  al. 2018, eq. 4: `bezier.connecting`], so the path is continuous in position, velocity and acceleration. Each
  curve is a straight line, so it can't cut a corner into a hold or rise above where it's going:
  - **up** `ABOVE` out of socket a. A ball leaves a socket only within ~10° of vertical [simulator `wall.py`,
    `EXIT_BLOCK`];
  - **out** `OUT` from the wall;
  - **across** to in front of socket b;
  - **in** to above socket b;
  - **down** into socket b.
- **The IK step is a QP** (`ik_step_qp`, solved with `quadprog` through the ediadvancedrobotics tutorials' helper
  `quadprog_solve_qp`). Each step chooses the change of configuration v that:
  - **minimises** the four balls' errors (`‖J v − displacement‖²`; a held ball weighs `HELD_WEIGHT`, the moving
    one 1), plus `REGULARISATION ‖v‖²` (as `DAMPING` did), minus a gradient step up the moving limb's log
    manipulability (Yoshikawa's `w = √det(J Jᵀ)` over its own joints, forward differences, `MANIPULABILITY_GAIN`);
  - **must stay within** the joint limits (one Euler step) and the URDF's speed limits (6.6 rad/s, per IK step: over
    `DT / ITERATIONS`);
  - **tries to stay** `JOINT_MARGIN` inside the joint limits and under `SPEED_MARGIN` of the speed limits, through
    slack variables ≥ 0 penalised by `SLACK_WEIGHT`.
  For now it has the balls only: no trunk or posture task, so `height` and `posture` are ignored.
  `ik_step_pinv` (below) is the previous IK, kept for comparison.
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
- **Stance pose** (`start_pose`, cached per stance): the hang pose with its trunk fitted to the stance's sockets,
  then IK to 1 mm, the joints pulled toward the hang pose's (`POSTURE_GAIN`, last and lowest), until q stops moving.
  The same posture task runs during each move, and **every move ends exactly in its stance pose**:
  - within `JOINT_TOL`, it jumps there;
  - further, it blends there over `BLEND_TIME`, with the balls checked.
  So each move starts from its stance's pose, whatever came before.
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
| `CORNER_TIME` | 0.3 s | each of up, out, in and down; across takes the rest of `MOVE_TIME` |
| `MOVE_TIME`, `DT` | 2.0 s, 0.02 s | per move; between samples (the robot's 50 Hz loop) |
| `POSTURE_GAIN`, `JOINT_TOL`, `BLEND_TIME` | 0.1, 0.01 rad, 0.3 s | the stance pose (above); the blend lasts long enough to stay under `SPEED_MARGIN` |
| `HELD_WEIGHT`, `REGULARISATION` | 1e3, `DAMPING`² | the QP: a held ball's weight (the moving one's is 1); on ‖v‖² |
| `JOINT_MARGIN`, `SPEED_MARGIN`, `SLACK_WEIGHT` | 0.1 rad, 0.8, 1e2 | the QP: how far inside its limits it tries to stay, and at what cost |
| `MANIPULABILITY_GAIN` | 1e-5 | the QP: on the gradient of the moving limb's log manipulability |
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

**QP vs pinv IK**, on 279 of those moves (every fourth):

| | moves doable | fastest joint vs its speed limit | closest to a joint limit | moving limb's mean log w |
|---|---|---|---|---|
| pinv | 279 | 3.72× (84 moves over) | 0 rad (277 moves within 0.1) | −7.46 |
| QP, `MANIPULABILITY_GAIN` 0 | 276 | 0.78× | 0.100 rad | −7.27 |
| **QP, 1e-5** | **279** | **0.77×** | **0.100 rad** | **−7.17** |
| QP, 1e-4 | 267 | 0.80× | 0.100 rad | −7.06 |
| QP, 1e-3 | 200 | 0.80× | 0.100 rad | −7.28 |

- **Cost:** with no posture task, every QP move ends away from b's stance pose and needs the blend. The L2 s66102 climb
  (35 moves) lasts 80.5 s instead of 74.5 s.
- **Speed:** the QP takes 48 ms a move against 139, and planning that wall with `ik_traj` takes 18 s against 40.

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
