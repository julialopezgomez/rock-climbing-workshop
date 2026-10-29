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

## v0 could be
For L1: a ladder heuristic that moves one limb at a time to the next hold up, hands first and then feet, and keeps
each step only if `feasible` says yes.

## Candidates later (not decided)
- A graph search over stances (BFS / A*), with `feasible` as the edge test and `cost` as the edge weight.
- Hand-authored plans per wall as a fallback.

## Open decisions
- Method (D2, decide by Tue 20 Oct).
- How to handle L2 gaps and L3 lateral moves.
- Whether one or more limbs may move per stance change.
- Whether the plan should reach for the top fast or play safe.

## v0 (reference)
The first planner (reachability A\* with LazySP over the optimiser's feasibility, its reach sets and cache) is in
[v0/](v0/), with its own [README](v0/README.md), [GUIDE](v0/GUIDE.md) and playground. It still runs:
`rcr.planner.v0.plan`, `scripts/plan_wall.py`, `scripts/plan_eval.py`, `scripts/make_reach.py` and
`tests/test_planner_v0.py` use it. `rcr.planner.plan` above starts over from the stub [Ours].
