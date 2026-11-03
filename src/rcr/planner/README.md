# planner

**Purpose:** decide which limb goes on which hold and in what order, for the whole climb.

| | |
|---|---|
| Input | `WallModel` (holds + `start` stance); a `FeasibilityFn` passed in as an argument |
| Output | the **full** `ContactPlan`: `stances[0] == wall.start`, ending with a hand on one of the highest holds |
| Runs on | laptop |

- **Rule:** the planner imports `rcr.types` only. It **never** imports `rcr.optimiser` or `rcr.controller`, and a test enforces this.
  When the planner needs to know "can the robot do this transition?", it calls `feasible(wall, a, b)`, which it receives as an argument.
- **Scoring** [Rules §8]: height comes first (the highest hold *touched*), then time. A plan that reliably reaches high beats a
  fast plan that might fail.

## Method (v1, 29 Oct) [Ours]
A\* over stances (`astar.py`), started over from v0 to keep it minimal:
- **Node:** a stance, every limb on its own hold. **Move:** one limb to a free hold, cost 1.
- **Heuristic:** sum over the four limbs of (top hold z − limb z) / `STEP`, `STEP` = 0.105 m, the longest step the
  simulator's routes leave a limb. Not admissible: plans are not always the fewest moves.
- **Goal:** a hand on a highest hold [Rules §8]. Otherwise, the plan goes to the highest hand reachable.
- **`feasible(wall, a, b)`**, depends on `b` only, so it is cached:
  - reach: no two balls further apart than their reaches plus the distance between their shoulders/hips;
  - IK: hierarchical, like the ARO course reference, on the four ball positions, joints clipped to their limits;
  - collision: the simulator's MuJoCo scene of the wall; any robot contact counts except a ball in its own hold and
    self pairs already touching in the hanging posture (our SRDF).

**Replanning** (`replan(wall, feasible, start, failed_moves, failed_stances)`) [Ours]: the same A\* from `start`
(where the robot is) instead of `wall.start`. It never takes a move `(a, b)` in `failed_moves` and never enters a
stance in `failed_stances`, whatever `feasible` says. A limb on `None` (off any hold) is placed first; IK and reach
then leave it out. `ik()` returns, besides the verdict, which balls reached their sockets.

The robot comes only from files Microban and the simulator use:
| What | Source |
|---|---|
| URDF, ball frames `<limb>_ball` | `microban_model.urdf(climb=True)` [Microban] |
| socket = `Hold.position` + (0, −0.026, −0.014) m | `climbing_sim/wall.py` [simulator] |
| IK start: hanging posture, trunk fitted to the sockets | `climbing_sim/assets/poses/hang.json`, `sim.py` `reset_hanging` [simulator] |

## Open decisions
- Ball rotations; the IK's task order.
- Dangling limbs at the goal and the 3 s hold.
- How to handle L2 gaps and L3 lateral moves.

## v0 (removed)
The first planner (reachability A\* with LazySP over the optimiser's feasibility, its reach sets and cache) is in git
history: `src/rcr/planner/` on `main` at commit `b52d553`, with its scripts `plan_wall.py`, `plan_eval.py` and
`make_reach.py`.
