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
  hold b (`ball_path`). It's three Bézier curves in time (`bezier.py`), continuous in position, velocity and
  acceleration where they meet. The control points come from the states at both ends [C-CROC, Fernbach et al.
  2018, eq. 4: `bezier.connecting`]:
  - **up:** straight up out of socket a, from rest, to `ABOVE` it, in `VERTICAL_TIME`. A ball leaves a socket only
    within ~10° of vertical [simulator `wall.py`, `EXIT_BLOCK`].
  - **over:** out from the wall and over to `ABOVE` socket b. Its two middle control points are `OUT` in front of
    its two ends.
  - **down:** straight down into socket b, to rest, in `VERTICAL_TIME`.
  - Where they meet, the ball moves straight up (down) at `JUNCTION_SPEED`, with no acceleration.
- **The whole body follows** by the planner's hierarchical IK (`astar.ik`, copied here, not imported). The held
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
| `ABOVE`, `VERTICAL_TIME` | 0.03 m, 0.4 s | straight up out of socket a, and down into socket b |
| `OUT` | 0.06 m | the way over's middle control points, away from the wall (the ball at most 33 mm off) |
| `JUNCTION_SPEED` | `ABOVE / VERTICAL_TIME` | straight up (down) where the curves meet |
| `MOVE_TIME`, `DT` | 2.0 s, 0.02 s | per move; between samples (the robot's 50 Hz loop) |
| `POSTURE_GAIN`, `JOINT_TOL`, `BLEND_TIME` | 0.1, 0.01 rad, 0.3 s | the stance pose (above) |
| `GAIN`, `ITERATIONS` | 0.5, 3 | fraction of each error (and of the trunk's tilt) closed per IK step; steps per sample |
| `DAMPING` | 0.03 | the moving ball's and the trunk's damped inverse |
| `MAX_ERROR` | 0.01 m | a held ball further from its socket: `ValueError` naming the move |

Measured on the 442 moves that A* asks about and that pass the contact check, on L2 s5226, L2 s7 and L3 s1:
- **The three curves:** 440 doable, the moving ball within 9.8 mm of its path.
- **With `JUNCTION_SPEED` twice as fast:** 416 doable. The way over overshoots 51 mm above socket b, and hands go out
  of reach.
- **With a single middle control point:** only 19 mm off the wall, and 406 doable.
- **The old single Bézier** (`OUT` = `ABOVE` = 0.06, min-jerk; it left socket a sideways): 442 doable.

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
