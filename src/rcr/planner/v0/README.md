# planner v0 (reference)

> Kept for reference: `rcr.planner.plan` starts over from the stub (see [../README.md](../README.md)).
> This is `rcr.planner.v0`; the paths below are as they were before the move into `v0/`.

**Purpose:** decide which limb goes on which hold and in what order, for the whole climb.

| | |
|---|---|
| Input | `WallModel` (holds + `start` stance); a `FeasibilityFn` passed in as an argument |
| Output | the **full** `ContactPlan`: `stances[0] == wall.start`, ending with a hand on one of the highest holds (or as high as it can get) |
| Runs on | laptop |

- **Rule:** the planner imports `rcr.types` (and `rcr.walls` for the grid) only. It **never** imports `rcr.optimiser` or
  `rcr.controller`, and a test enforces this. It asks `feasible(wall, a, b)`, which it receives as an argument.
- **Scoring** [Rules §8]: height comes first (the highest hold *touched*), then time. A plan that reliably reaches high beats a
  fast plan that might fail.

## Method (D2): reachability A\* over stances, with the optimiser's feasibility checked at expansion
Decided on 27 Oct, on this branch; on 29 Oct the check moved from the best path's moves (LazySP) to each stance A\*
expands (Lazy Weighted A\*) [Ours]. **Reason:**
- Holds are points, so what stays continuous is where the trunk can be. A stance's set of trunk positions is convex,
  so one small LP tells whether the stance is possible, in about 1 ms.
- That cheap test generates the candidate moves. The optimiser's 1–17 s `feasible()` is asked about the move into a
  stance when A\* pops it to expand, so children that are never expanded cost nothing, and the search never has to
  start again.
- A\* keeps plans short (time is the tie-break).

It draws on (see Julia's notes):
- **CASSR** (Wang & Tonneau 2026): A\* whose nodes carry continuous reachable sets; a node is expanded through the
  Minkowski sum of a region and a reach set (Eq. 2).
- **SL1M** (Tonneau et al. 2020): reach polytopes for each effector, sampled offline in that effector's frame
  (§V-A.2). With point contacts the polytopes don't depend on the surface (§V-C).
- **FARO** (Ciebielski et al. 2026): checks run from cheap to expensive, a failure prunes everything above it, and
  results are cached.
- **NAS** (Wang et al. 2024): the backward pass over all solutions. An open item, not used yet.

### Terms
- **Trunk:** Microban's torso, the URDF root link `trunk` (`q[0:3]` in the optimiser and the sim).
- **Trunk region of a stance:** every trunk position, orientation held fixed, from which all four balls can sit on
  their holds at once.
  - Each limb on its hold allows the trunk only inside that hold minus the limb's reach set.
  - The region is where these four sets overlap, so it is convex.

### How it works
1. **Reach sets** (`data/robot/reach.json`, written by `scripts/make_reach.py`):
   - Sample each limb's joints uniformly within the URDF's limits, with the trunk orientations the optimiser used on
     the example climb.
   - Keep the ball − trunk offsets, as support values along 206 directions shared by all limbs. That is an outer
     approximation of their convex hull, plus 5 mm.
   - Because the directions are shared, a stance's trunk region is `{t : N t ≤ u}`, with `u` the elementwise minimum
     over the limbs.
   - Kinematics only [Ours].
2. **Reach graph** (`astar.Graph`):
   - A node is a stance (limb → hold); a move changes one limb's hold.
   - A move is in the graph if:
     - the limb doesn't move down (`MAX_DROP`);
     - the new stance is inside the posture envelope (`ENVELOPE`);
     - its trunk region is not empty.
   - Candidate holds are tested all at once, with array masks and then a necessary test (`u_k + u_-k ≥ 0`, right for
     95 % of what it passes).
   - The exact LP runs only for the stances the search expands.
3. **A\*** (`astar.solve`):
   - Cost is 1 per move. The goal is a hand on a highest hold.
   - The heuristic is a lower bound on the moves left; a test checks it never overestimates on a small wall.
   - It is weighted (`WEIGHT` = 3): plans have at most 3× the fewest moves of the graph.
4. **Feasibility at expansion** (`astar.solve`, Lazy Weighted A\*, Cohen, Phillips & Likhachev 2014):
   - A stance is queued once per parent. When it's popped, its trunk-region LP runs, then `feasible(parent, stance)`.
   - A rejected move is banned (5) and the next stance is popped; the stance can still be expanded from another parent.
   - So every stance expanded, and every move of the plan, has been checked. Each move is asked at most once.
   - If the top can't be reached (or after `MAX_REJECTIONS`), the plan goes to the highest stance expanded.
   - **Cost:** one call per stance expanded. With every move accepted that's 158–3,564 per test wall (L1–L3, seeds
     0–2), with as many distinct layouts, so the cache (6) barely helps. Not yet run with the real optimiser.
5. **A rejection reaches as far as its check** (`astar.Banned`). The optimiser's reason names the check that failed
   (`edge_checks.CHECKS`), and each check only looks at part of the move:

   | Check that failed | What it looks at | What is banned |
   |---|---|---|
   | `stance` | the new stance | every move into it |
   | `start` | the old stance | every move out of it |
   | `release`, `lift` | the moving limb leaving its hold | every move of that limb out of the old stance |
   | `arrival` | the moving limb settling on its new hold | every move of that limb into the new stance |
   | anything else | the move | that move |

   A test checks these names against the optimiser's.
6. **Layout cache** (`cache.Cache`):
   - The verdicts are keyed by the move's layout: the moving limb, the grid offsets of the five holds involved, and
     those of every other hold within 0.25 m, all relative to the moving limb's old hold.
   - Holds are identical, on a grid, under fixed gravity, so the same move anywhere on any wall is asked once.
   - `solve` always uses it in memory; the scripts keep it in `build/feasibility_cache.json`.
   - Its version is a hash of the optimiser's code, the hooks and the hold mesh.

## Files
New to the planner? Start with [GUIDE.md](GUIDE.md) and its interactive companion `playground.html` (open it in a browser).

| File | What |
|---|---|
| `GUIDE.md`, `playground.html` | starter guide; the planner rebuilt in JavaScript on a 2D wall, one tab per step (still the best-path version) |
| `__init__.py` | `plan(wall, feasible)` → `astar.solve(...).plan` |
| `reach.py` | loads the reach sets; trunk regions (`region`, `maybe_nonempty`, `centre`: the Chebyshev LP) |
| `astar.py` | `posture`, `Graph` (the reach graph), `Banned`, `solve` (A\* with feasibility at expansion) |
| `cache.py` | `layout_key`, `Cache` |
| `scripts/make_reach.py` | writes `data/robot/reach.json` (optimiser group) |
| `scripts/plan_wall.py` | plans one wall (or `--warm N` walls), with the reach graph only or the optimiser |
| `scripts/plan_eval.py` | `moves`: what the optimiser accepts, by rise; `plans`: plan walls, optionally `--solve` them |

## Parameters
| Parameter | Value | Tag |
|---|---|---|
| reach sets | 20000 samples a limb, 206 directions, +5 mm, trunk 0.06–0.20 m from the wall | [Ours]; the trunk range widens the example climb's 0.083–0.174 m |
| `ENVELOPE` | the ranges of the 36 solved stances of `contact_plan_hard.json`, + `ENVELOPE_MARGIN` 0.03 m; lean ±0.10 m | [Ours], a proxy for torque and collisions |
| `MAX_DROP` | 0 m | [Ours] |
| `WEIGHT` | 3 | [Ours] |
| `MAX_REJECTIONS` | 200 | [Ours] |
| `CACHE_RADIUS` | 0.25 m | [Ours] |
| goal | a hand on a highest hold | [Rules §8] |

## How to run
```bash
uv run scripts/plan_wall.py --level 1 --seed 0                    # reach graph only, < 1 s
uv sync --group optimiser --group sim     # both: syncing one group uninstalls the other
uv run scripts/plan_wall.py --level 1 --seed 0 --feasibility optimiser --save build/plan.json
uv run scripts/plan_eval.py moves --per-bin 6                     # what the optimiser accepts
uv run scripts/plan_eval.py plans --seeds 3 --solve               # plan walls, then optimise the plans
uv run scripts/make_reach.py                                      # rebuild the reach sets (3 s)
```

## Measured
With the reach graph alone (every move accepted), it plans the generated walls (L1–L3, seeds 0–9) and `wall_hard.json`
to the top in 0.1–4 s per wall, expanding (and so asking `feasible()` about) 114–3,564 stances. In tests: 9 walls plus a small ladder, where weight 1 matches a breadth-first search.

With the real optimiser (`scripts/plan_eval.py`, MacBook, 27 Oct), on moves sampled at random from the reach graph:

| Reach graph | Accepted | Rejected at | Mean check |
|---|---|---|---|
| kinematics only | 2 of 72 | `stance` 58, `start` 6, `release` 5, `path` 1 | 2.6 s |
| + `ENVELOPE` | 18 of 63 (29 %) | `stance` 17, `path` 16, `start` 8, `release` 3, `arrival` 1 | 17 s |

**Open problem: too many optimiser calls.**
- With the previous best-path checks (before 29 Oct), on `gen_l1_s0` with the real optimiser, there was still no plan
  after 40 min and about 80 checks.
- With checks at expansion there's no loop to converge, but every stance expanded costs a call. On the reach graph
  alone, that's 114–3,564 per wall, nearly all distinct layouts. Not yet measured with the real optimiser, where
  rejections both ban families of moves (fewer expansions) and push A\* off the bold path (more).
- The cause, reproduced with the playground's toy optimiser (tab 5):
  - the fewest-moves path picks the boldest moves (stances at the envelope's edge), which a stricter check rejects;
  - a `stance` ban removes one stance but not its near-twins, and a `path` ban removes one move while the same hold blocks
    many others.
- On the toy, a CASSR-style safety cost (β = 3) cuts `stance` rejections from 18–148 per wall to 0–16, but `path`
  rejections remain.
- GUIDE.md §5 lists the fixes to decide on.

## Open items
- **A stance model learned from the optimiser's static poses**, to replace `ENVELOPE`. This is the main one, see
  Measured.
- The friction-based static-equilibrium region per stance (Del Prete, Tonneau & Mansard 2016, SL1M ref [19]): an LP,
  convex per stance.
- Trunk lean as discrete branches (CASSR's yaw), rather than one set over all the example's orientations.
- A stage-2 trunk point per stance, deep in its region, to seed the optimiser (CASSR §VI).
- A NAS-style backward pass: cost-to-go for every stance, as an informed heuristic and a replanning policy.
