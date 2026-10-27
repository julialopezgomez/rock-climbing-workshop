# Planner: starter guide

For teammates who know robotics but not planning.
- Read §1 first (five minutes).
- Then follow §3 with the playground open: `open src/rcr/planner/playground.html`. It is one offline file, and each step below
  says which tab shows it.
- For parameters and files, see [README.md](README.md).

---

## 1. What the planner does, in one page

**The job.**
- **In:** the wall (`WallModel`: the holds and the start stance), and a function `feasible(wall, a, b)` that says whether the
  robot can go from stance `a` to stance `b`.
- **Out:** a `ContactPlan`, the list of stances from the start to a hand on a highest hold, one limb moving at a time.
- The optimiser then turns that plan into joint trajectories.

**The difficulty.**
- The only reliable judge of "can the robot do this move?" is the optimiser's `feasible()`. It solves small non-linear
  programs for torque, balance and collisions, and takes **2–17 s per move**.
- A climb is 20–35 moves.
- From any stance there are 20–50 possible next moves, and the number of possible sequences explodes.
- So we can't ask the optimiser about everything.

**The idea, in three parts:**
1. **A cheap stand-in graph** (the *reach graph*). Each limb can reach only so far from the trunk, and the robot only holds
   postures like ones the optimiser has solved. This prunes most moves in **milliseconds**, with no optimiser call.
2. **A\* search** on that graph finds the plan with the fewest moves.
3. **Lazy checks.**
   - Only the moves of the best plan are sent to the optimiser.
   - A "no" bans that move, or a whole family of moves when the failed check implies it, and the search runs again.
   - Verdicts are remembered by the *shape* of the move, so the same move elsewhere on the wall isn't asked twice.

```
   wall ──► reach graph (ms per stance) ──► A* ──► best plan ──► check its moves with feasible() (seconds each)
                 ▲                                                         │
                 └──────────── ban what the "no" rules out ◄───────────────┘ (until every move passes)
```

> [!IMPORTANT]
> **Status (27 Oct).**
> - With the reach graph alone, it plans every test wall (L1–L3) in under 2 s.
> - With the real optimiser, the "search → reject → search again" loop **does not converge fast enough yet**. See §5:
>   it's the main open problem.

> [!TIP]
> **See it:** playground tab 0 runs the same wall two ways. *Eager* asks the optimiser about every move the search touches;
> *ours* asks about the best path only. Compare the optimiser-call counters.

---

## 2. Words you need

| Term | What it means here |
|---|---|
| **stance** | which hold each limb is on: `(right hand, left hand, right foot, left foot)`. The planner's unit of state. |
| **move** | one limb changes hold; the other three stay. |
| **graph, node, edge** | nodes are stances; an edge is a move between two of them. Planning means finding a path in this graph. |
| **search** | exploring the graph from the start, one stance at a time, until reaching a goal stance. |
| **expanding a stance** | listing its *children*: the stances one move away. |
| **A\*** | a search that always expands the most promising stance first, the one with the lowest `f = g + h`. |
| **g** | the cost so far: moves from the start to this stance. |
| **h (heuristic)** | an estimate of the moves still needed to reach the top. |
| **admissible** | `h` never overestimates. Then A\* returns the fewest moves. Ours is (a test checks it). |
| **weighted A\*** | using `f = g + w·h` with `w > 1`. It trusts `h` more, so it expands fewer stances, and plans have at most `w` times the fewest moves. We use `w = 3`. |
| **trunk** | Microban's torso, the root link the four limbs hang off. |
| **reach set** | everywhere a limb's ball can be, relative to the trunk. Sampled offline from the URDF. |
| **trunk region** | every trunk position from which all four balls reach their holds at once. Empty means the stance is impossible, whatever the joint angles. |
| **LP** | a linear program: a fast, exact optimisation with linear constraints. One LP tells whether a trunk region is empty, in about 1 ms. |
| **posture envelope** | ranges (hands apart, hand above the feet, lean, …) taken from the 36 stances the optimiser solved on the example wall. Our stand-in for torque limits. |
| **lazy evaluation / LazySP** | ask the expensive check only about the edges of the current best path, not about every edge the search sees. |
| **ban** | a move, or a family of moves, the optimiser rejected. The search skips it. |
| **layout cache** | verdicts stored under the move's layout: the hold positions relative to each other, on the grid. The same move anywhere hits the same entry. |

---

## 3. Walkthrough, in order of execution

### Step 0 (offline, once): build the reach sets
- **Command:** `uv run scripts/make_reach.py` (needs the optimiser group and `MICROBAN_URDF`). It takes about 3 s.
- **What it does:**
  - Samples 20,000 random joint configurations per limb within the URDF's limits, with the trunk tilted as the optimiser
    tilted it on the example climb.
  - Records where the ball ends up relative to the trunk.
  - Stores each limb's set as 206 numbers: how far it reaches along 206 fixed directions.
- **Output:** `data/robot/reach.json`, committed.
- **Why fixed directions:** every limb uses the same 206 directions, so intersecting the four limbs' sets is just an
  elementwise minimum (§3, step 4c).

> [!TIP]
> **See it:** tab 2, *forward view*: drag the trunk and see each limb's reach set around it.

### Step 1: you type a command
```bash
uv run scripts/plan_wall.py --level 1 --seed 0                          # reach graph only, < 1 s
uv run scripts/plan_wall.py --level 1 --seed 0 --feasibility optimiser  # with the real optimiser (slow, see §5)
```
- In the pipeline, the same thing is one call: `planner.plan(wall, optimiser.feasibility)` in
  [__init__.py](__init__.py), which returns `astar.solve(...).plan`.

### Step 2: `plan_wall.main()` loads the wall and wraps the optimiser
- **The wall:** `walls.generate(level, seed)`, or `io.load_wall(file)`.
- **The optimiser:** `optimiser_feasibility()` wraps `rcr.optimiser.feasibility` in `Cache(...)`
  ([cache.py](cache.py)).
- The planner never imports the optimiser; it only receives this function (a rule in CLAUDE.md, enforced by a test).

### Step 3: `astar.solve()` sets up
- `Graph(wall, reach.load())`:
  - the holds' contact points (where a resting ball's centre sits);
  - the reach sets;
  - empty caches for children and trunk-region verdicts.
- `Banned()`: empty.
- The start key: the wall's start stance.

### Step 4: the loop (`solve`, [astar.py](astar.py))
```
repeat:
    path = search(...)                 # 4a: A* on the reach graph, skipping banned moves (no optimiser call)
    for each move (a → b) of path, from the start:
        verdict = feasible(a, b)       # 4e: the optimiser (through the cache)
        if no: ban what it implies     # 4f
               and search again
    if every move passed: return the plan
```

> [!TIP]
> **See it:** tab 1 walks through this whole loop on a real run, one step at a time, highlighting the box in the diagram
> and the move on the wall.

#### 4a. `search()`: A\* on the reach graph
1. Put the start in a *priority queue*, ordered by `f = g + w·h`.
2. Pop the stance with the lowest `f`.
3. Skip it if it was already expanded, or if its trunk region turns out empty (4c).
4. If a hand is on a highest hold, stop: follow the parent links back to get the path.
5. Otherwise expand it (4b). Each child gets `g + 1` if that's better than any earlier route to it, and goes into the queue.
6. If the queue empties without reaching the top, return the path to the highest stance reached (height first [Rules §8]).

> [!TIP]
> **See it:** tab 4 steps through `search()` line by line on a small ladder.
> - The queue (with f, g and h) and the heuristic's breakdown update as you go.
> - Tick **quiz me** to be asked, at each line, which stance pops, whether it's skipped, how many children are queued, and
>   what a child's f is.
> - Compare `w = 1` (fewest moves) with `w = 3` (fewer stances expanded).

#### 4b. `Graph.children()`: which moves exist
For each limb, every hold on the wall goes through these filters, cheapest first:

| Filter | Throws out |
|---|---|
| occupied | a hold another limb is on |
| `MAX_DROP` | a hold lower than the limb's current one (we never move down) |
| posture envelope (`posture`) | a stance outside the postures already solved (too wide, too stretched, leaning too far) |
| trunk region, cheap test | a stance whose trunk region is certainly empty (`maybe_nonempty`) |

- Everything is done with arrays, all holds at once. A stance's children take milliseconds.

> [!TIP]
> **See it:** tab 3 colours every hold by the filter that rejects it, for the limb you choose. The blue shape is where that
> limb can reach while the other three stay put.

#### 4c. `Graph.valid()` and [reach.py](reach.py): is the stance possible?
- Each limb on its hold allows the trunk only inside one shape: the hold minus that limb's reach set.
- The stance is possible where all four shapes overlap: the **trunk region**.
- With the 206 shared directions, the region is `{t : N t ≤ u}`, where `u` is the elementwise **minimum** of the four
  limbs' bounds.
- **Cheap test** (`maybe_nonempty`):
  - In each direction, the bound must not be below minus the opposite direction's bound; otherwise the region is certainly
    empty.
  - It's right for 95 % of the stances it passes.
- **Exact test** (`centre`):
  - One LP finds the point deepest inside the region.
  - It runs only for stances the search actually pops (in step 4a), the same "lazy" idea one level down.

> [!TIP]
> **See it:** tab 2.
> - Drag the four holds and watch the green overlap appear or vanish.
> - The table prints the four bounds along one direction and their minimum.
> - The "hands too high" preset is empty.

#### 4d. `Graph.h()`: the heuristic
- At the goal, one hand is at the top, the other hand is near it, and the feet are within reach below.
- In one move, a hand can rise at most 0.35 m (the envelope's limit above the feet), and a foot at most twice that.
- So `h` = moves the hands still need + moves the feet still need. That's a lower bound, so it's admissible. It is also
  loose, which is why we weight it (`w = 3`).

#### 4e. Checking the path: `Cache` → `feasible()`
- Moves are checked in order from the start. A move already verified isn't asked again.
- **The cache** ([cache.py](cache.py)):
  - The key is the moving limb, plus the grid offsets of the five holds involved, plus any other hold within 0.25 m, all
    relative to the moving limb's old hold.
  - This works because holds are identical, on a 15 mm grid, under fixed gravity.
  - The scripts keep the cache on disk (`build/feasibility_cache.json`). Its version is a hash of the optimiser's code,
    the hooks and the hold mesh.
- **`feasible()`** (the optimiser, [edge_checks.py](../optimiser/edge_checks.py)):
  - It runs its own checks cheapest first: `rules → reach → stance → start → release → arrival → lift → path`.
  - It stops at the first one that fails, and its `reason` names that check.

> [!TIP]
> **See it:** tab 6. Shifting the whole wall keeps the key. Removing a nearby hold changes it; removing a far one doesn't.

#### 4f. `Banned.add()`: one "no", many moves removed
Each of the optimiser's checks looks at only part of the move, so its "no" reaches exactly that far:

| Failed check | It looked at | Banned |
|---|---|---|
| `stance` | the new stance alone | every move into it |
| `start` | the old stance alone | every move out of it |
| `release`, `lift` | the moving limb leaving the old stance | every move of that limb out of it |
| `arrival` | the moving limb settling into the new stance | every move of that limb into it |
| `path`, others | the motion itself | that move only |

> [!TIP]
> **See it:** tab 5.
> - Step through search → check → ban.
> - Switch "a 'no' bans" between *by its check* and *only that move*, and the cache on and off, and compare the totals.

### Step 5: the result
- **`Result`:** the plan, whether it reached the top, and counters (searches, stances expanded, optimiser calls, rejections,
  time).
- **The plan:** `plan_wall.py --save` writes it as `contact_plan/1`.
- **Next:** `optimiser.optimise(wall, plan)` turns it into the trajectory the robot executes.

---

## 4. Eager vs ours (the same wall)

|  | Eager: ask about every move | Ours |
|---|---|---|
| Who proposes moves | every hold in reach | the reach graph (posture rules + trunk region) |
| When the optimiser is asked | every move the search touches | only the best path's moves |
| What a "no" removes | one move | everything the failed check implies |
| Same move elsewhere | asked again | from the cache |
| Optimiser calls, toy, `gen_l1_s0` | 14,155 (≈ 13 h) | 346 (≈ 25 min) |
| Reaches the top (toy, 5 walls) | always | 2 of 5 within 200 rejections (§5) |

The bottom two rows come from the playground's toy optimiser, so they show the pattern, not real timings.

---

## 5. Where it stands, and the open problem

**Measured with the real optimiser** (`scripts/plan_eval.py`, 27 Oct). It asked about moves sampled at random from the
reach graph:

| Reach graph | Accepted | Mostly rejected at | Mean check |
|---|---|---|---|
| kinematics only | 2 of 72 | `stance` (58): the new stance can't be held at all | 2.6 s |
| + posture envelope | 18 of 63 (29 %) | `stance` 17, `path` 16, `start` 8, `release` 3 | 17 s |

On `gen_l1_s0`, a full plan with the real optimiser had still not converged after 40 minutes and about 80 checks.

**Why the loop doesn't converge** (the toy in tab 5 shows the same thing):
- **The fewest-moves path is bold.** It picks the biggest rises and stances at the edge of the envelope. Those are exactly
  the moves a stricter check rejects, so the rejection rate on the path is far higher than on random moves.
- **Bans are narrow.** A `stance` ban removes one stance, but its near-identical neighbours remain. A `path` ban (a shin
  hitting a hold) removes one move, while the same hold blocks dozens of others.

**What would fix it** (not in the code yet; to decide):

| Fix | Effect in the toy |
|---|---|
| **Safety cost** (as CASSR): each move costs `1 + β·(how far the new stance is outside the postures already solved)` | `stance` rejections drop from 18–148 per wall to 0–16. L2, L3 and the example wall reach the top. On the two plain ladders, though, `path` rejections then use up all 200 (on the small ladder too, which reached the top without it). Try β = 3 in tabs 0 and 5. |
| **A cheap sweep test** in the reach graph, for a foot's shin passing a hold (as SL1M handles collisions) | needed together with the safety cost: most of the remaining rejections are `path` |
| **A warm cache:** plan many generated walls before the event | repeated layouts answered instantly |
| **A learned stance model** from the optimiser's static poses, to replace the hand-set envelope | the real fix for "which stances can be held" |

---

## 6. Can we assume quasi-static motion?

The optimiser is quasi-static, and so is everything the planner relies on. The worry is that "gravity is always there".

> [!NOTE]
> Quasi-static doesn't remove gravity; it removes the robot's **own** acceleration. Newton for the centre of mass:
>
> $$m\,\ddot c = m\,\mathbf g + \sum_i f_i \qquad \dot L = \sum_i (x_i - c)\times f_i$$
>
> - Quasi-static sets $m\ddot c \approx 0$ and $\dot L \approx 0$. What remains is static equilibrium: the holds' forces
>   cancel gravity exactly, and their moments cancel too.
> - Gravity is a force on the robot; 9.81 m/s² is only what you'd see in free fall. In equilibrium the holds push back,
>   and the net acceleration is zero.

**When it holds:** when $|\ddot c| \ll g$. A smooth move of 0.15 m peaks at about $5.8\,d/T^2$:

| Time per move | Peak acceleration | Share of g |
|---|---|---|
| 2 s (the optimiser's phase B) | 0.22 m/s² | 2 % |
| 1 s | 0.87 m/s² | 9 % |
| 0.5 s | 3.5 m/s² | 35 %: no longer quasi-static |

**It breaks:**
- when we speed up for the time tie-break;
- at the instant a limb lets go;
- at impacts.

**How to keep it safe:**
- Keep the 70 % torque margin.
- Choose move durations so that $|\ddot c| \le 0.05$–$0.1\,g$.
- Check the timed trajectory with full dynamics ($\tau = M\ddot q + h - J^\top f$) or in MuJoCo.
- For faster climbing, a centroidal-dynamics feasibility test such as C-CROC.

---

## Tools in this guide

| Concept | Playground tab | Code |
|---|---|---|
| eager vs lazy, same wall | 0 | `astar.solve` |
| what runs, in order | 1 | `scripts/plan_wall.py` → `astar.solve` |
| trunk region, reach sets | 2 | `reach.py`, `Graph.valid` |
| expanding a stance | 3 | `Graph.children`, `posture` |
| A\*, with a quiz | 4 | `search`, `Graph.h` |
| lazy checks and bans | 5 | `solve`, `Banned` |
| layout cache | 6 | `cache.layout_key` |

**About the playground:**
- **It is 2D.** The wall plane only, with the robot's real reach sets sliced at the trunk's typical distance from the wall
  (0.127 m). The real code also lets the trunk move in and out from the wall.
- **Shapes and regions.** Reach sets are polygons along 48 directions, and the exact test clips polygons instead of solving
  an LP.
- **The optimiser is a toy rule**, with the same kinds of failure (`stance`, `release`, `path`) and simulated seconds per
  call. It shows patterns, not real numbers.
- **The safety cost β** is a proposal: the Python code today is β = 0.
- **The playground stops a search** at 30,000 expansions; the Python search has no limit.
