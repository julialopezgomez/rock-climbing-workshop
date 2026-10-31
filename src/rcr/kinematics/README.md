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
  hold b. It's a cubic Bézier through `P0 = socket a`, `P1 = P0 + OUT` away from the wall, `P2 = socket b + OUT`
  away and `ABOVE` up, and `P3 = socket b`. The ball leaves socket a straight off the wall, arcs up and away, and
  goes into socket b from above and out. It is timed by min-jerk (`10τ³ − 15τ⁴ + 6τ⁵`), so it is at rest at both
  ends.
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
- **Start pose:** the planner's: the hang pose with its trunk fitted to stance 0's sockets, then IK to 1 mm.
- **Output:** `optimise(wall, plan) -> Trajectory`: the start pose, then every move's samples every `DT`, each move
  from where the previous one ended. `positions` = `planned_positions` = the IK's joint angles (kinematic);
  `trunk_poses` for replaying it.
- **`feasibility`** is a stub (`NotImplementedError`) for now. `scripts/run_pipeline.py --check` shows the planner
  as `stub` as well, since it calls it; `scripts/plan_sim_wall.py` uses the planner's own `astar.feasible`.

| Parameter | Value | |
|---|---|---|
| `OUT` | see `ik.py` | the ball away from the wall at both ends of its path |
| `ABOVE` | see `ik.py` | above socket b, before going in (arrival direction `(0, OUT, -ABOVE)`) |
| `MOVE_TIME`, `DT` | 2.0 s, 0.02 s | per move; between samples (the robot's 50 Hz loop) |
| `GAIN`, `ITERATIONS` | 0.5, 3 | fraction of each error (and of the trunk's tilt) closed per IK step; steps per sample |
| `DAMPING` | 0.03 | the moving ball's and the trunk's damped inverse |
| `MAX_ERROR` | 0.01 m | a held ball further from its socket: `ValueError` naming the move |

Measured on level-2 climbs, seeds 1, 2, 4, 5, 6, 7, 3352 and 5226 (about 40 moves, 80 s of motion each, solved in
about 2 s), every move ending with all balls in their sockets:
- `OUT` 0.02 / `ABOVE` 0.04: the moving ball within 1.1 mm of its path, the held balls exact.
- `OUT` 0.05 / `ABOVE` 0.10: the moving ball up to 10 mm short of the top of its path on long hand moves (out of
  reach), within 2 mm elsewhere; the held balls within 5 mm of their sockets (under 1 mm on most walls).

```bash
uv run mjpython scripts/plan_sim_wall.py --level 2 --seed 5226 --heuristic-weight 1.5 --play  # see it
uv run scripts/plan_sim_wall.py --level 2 --seed 5226 --save-trajectory build/trajectory.json
uv run scripts/plan_sim_wall.py --level 2 --seed 5 --save-npz build/trajectory.npz  # q: N x 26, every 0.02 s
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
