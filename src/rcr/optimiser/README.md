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
- **The whole body follows** by the planner's hierarchical IK (`astar.ik`, copied here, not imported): each ball's
  task in `TASK_ORDER`, each in the null space of the ones before, then the trunk upright and facing the wall. One
  step per sample, with a feed-forward and an error correction (CLIK): `Δx = Δp_ref + GAIN · (p_ref − ball)`. The
  held balls' `p_ref` is their socket.
- **Joint limits:** a joint the step would push past a limit is locked there and the step solved again without it,
  then the joints are clipped (`projecttojointlimits`, as in the planner). With the clipping alone, the ankles and
  the shoulder roll saturate and a held ball drifted up to 10 mm off its socket on a level-2 climb. With the
  locking, every ball stays within 1 mm.
- **Start pose:** the planner's: the hang pose with its trunk fitted to stance 0's sockets, then IK to 1 mm.
- **Output:** `optimise(wall, plan) -> Trajectory`: the start pose, then every move's samples every `DT`, each move
  from where the previous one ended. `positions` = `planned_positions` = the IK's joint angles (kinematic);
  `trunk_poses` for replaying it.
- **`feasibility`** is a stub (`NotImplementedError`) for now. `scripts/run_pipeline.py --check` shows the planner
  as `stub` as well, since it calls it; `scripts/plan_sim_wall.py` uses the planner's own `astar.feasible`.

| Parameter | Value | |
|---|---|---|
| `OUT` | 0.055 m | the ball away from the wall at both ends of its path |
| `ABOVE` | 0.10 m | above socket b, before going in |
| `MOVE_TIME`, `DT` | 2.0 s, 0.02 s | per move; between samples (the robot's 50 Hz loop) |
| `GAIN` | 0.5 | fraction of a tracking error (and of the trunk's tilt) corrected per sample |
| `MAX_ERROR` | 0.01 m | a ball further from its target: `ValueError` naming the move (the path out of reach) |

Measured, level-2 climbs (about 40 moves, 80 s of motion, solved in 0.5 s):
- With `OUT` 0.03 m and `ABOVE` 0.02 m, every ball stayed within 1 mm of its target on seeds 1, 2, 4, 6, 7, 3352 and
  5226, and every move ended with all balls in their sockets.
- With the current 0.055 m and 0.10 m, the path goes out of some limbs' reach. Seed 5 tracks; seeds 1 and 5226 each stop
  at a hand move (seed 1: `left_hand r37_c21 -> r44_c18: a ball 17 mm from its target at 1.32 s`).

```bash
uv run mjpython scripts/plan_sim_wall.py --level 2 --seed 5226 --heuristic-weight 1.5 --play  # see it
uv run scripts/plan_sim_wall.py --level 2 --seed 5226 --save-trajectory build/trajectory.json
uv run scripts/plan_sim_wall.py --level 2 --seed 5 --save-npz build/trajectory.npz  # q: N x 26, every 0.02 s
```

## v0 (reference)
Quasi-static direct transcription, one NLP per transition, CasADi + IPOPT, on Microban's URDF: in [`v0/`](v0/),
described in [`v0/README.md`](v0/README.md). It still runs as it did:
```bash
uv sync --group optimiser --group sim
uv run pytest tests/test_optimiser_v0.py
uv run mjpython tests/test_optimiser_v0.py PLAN    # solve a contact plan, play it in the sim
```

## Open decisions
- Collisions with the holds, the wall and itself (none checked yet; the path's `OUT`/`ABOVE` are the only clearance).
- Torques: whether the motion can be held (0.605 N·m limit), and the sag pre-load of `positions`.
- What `feasibility` checks.
- Time per move (time is the tie-breaker in scoring).
- Whether `Trajectory` needs velocities, per-joint kp or the stance index (marked CHANGE in INTERFACES.md).
