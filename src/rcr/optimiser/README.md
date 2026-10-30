# optimiser

Turns a contact plan into the joint motion the robot executes. Runs on the laptop, never on the Pi.

| | |
|---|---|
| Input | the wall (`WallModel`) and the whole contact plan (`ContactPlan`) |
| Output | `optimise(wall, plan) -> Trajectory`: every joint's angle over time |

## Files
| File | What is in it |
|---|---|
| [`transition.py`](transition.py) | **the solver**: the settings (`TransitionConfig`), the constraints of one pose, the two problems (a static pose and a transition), and how they are solved |
| [`robot_model.py`](robot_model.py) | the robot's kinematics in CasADi, read from Microban's URDF: ball positions, Jacobians, centre of mass, gravity torques |
| [`__init__.py`](__init__.py) | `optimise`: solves the plan move by move, times the samples, and pre-loads the motor targets against sag |

## How a plan is solved
1. **Start pose.** A static pose of the plan's first stance: balanced, four balls on their holds, least torque
   (`stance_pose`).
2. **One move per pair of stances.** Each move changes one limb's hold. It starts where the previous move ended and
   ends when the moving ball is on its new hold; the rest of the body ends wherever is cheapest (`solve_transition`).
3. **Trajectory.** The samples of all moves, one after another, timed and pre-loaded (`to_trajectory`).

## One move: the problem (`transition_problem`)
Quasi-static: no inertia, every sample is a pose the robot can hold still. Samples k = 0..N (N = 12). The same list is
the docstring of `transition_problem` in [`transition.py`](transition.py), and each line is labelled in its code.

**Variables**, per sample: the pose `dq_k` (trunk and 19 joints, relative to the start pose `q0`) and the force each
hold pushes its ball with. The torques follow from these through the static balance
(`tau = g(q) − Σ Jᵀ f`); they are not variables.

**Phases:**
- **A**, k = 0..K (K = 4): all four balls held; the moving limb's force reaches 0 at K.
- **B**, k = K..N: the other three held; the moving ball travels, and is on its new hold at N.

**Subject to:**

| Constraint | In the code |
|---|---|
| sample 0 is the start pose | `dq_0 = 0` |
| every sample: the trunk balanced (net force and moment zero) | `balanced_pose` → `static_balance(...) == 0` |
| every sample: joint limits, \|torque\| ≤ `tau_limit` (70 % of the servo limit) | `balanced_pose` → `joint_and_torque_limits` |
| every sample: the trunk within 0.3 m / 1.2 rad of the start pose | `balanced_pose` → `trunk_bounds` |
| a held ball on its hold (the moving one up to K) | `balls == holds` |
| a held ball pushed by its hold within the friction pyramid (the moving one up to K−1) | `friction_pyramid` |
| the moving ball carries nothing from K on | `force == 0` |
| the moving ball on its new hold at N | `balls[N] == target` |
| the same rules at every midpoint between samples, held balls within 1 mm there | `midpoint_poses` |

**Cost** (minimised):
```
  w_tau    · Σ ||tau||²                 over the samples and midpoints    (low torque)
+ w_smooth · Σ ||dq_{k+1} − dq_k||²     between samples                   (smooth)
+ Σ posture(q)                          over the samples and midpoints    (legs near a chosen pose, spine at 25°)
+ w_swing  · Σ ||ball(q) − r(s)||²      phase B's samples and midpoints   (the moving ball near its swing curve; off by default)
```
`posture(q) = w_nominal · Σ (leg joint − its nominal angle)² + w_spine · (spine_z − cos 25°)²`.

**The swing curve** `r(s)` (`swing_point`) is a reference path for the moving ball, with no collision constraints:
`s = (k − K)/(N − K)` runs from 0 (the ball leaves its hold) to 1 (on its new hold); midpoints are at half steps.
`r` is a quartic Bézier curve from the old contact point to the new one. It leaves the cup straight up, is `swing_lift`
higher and `swing_out` further from the wall (−y) halfway, and comes straight down into the new cup. `ball(q)` is the
moving ball's centre (forward kinematics). It's a cost, so the ball is pulled towards the curve, not held on it:
`w_swing` sets how hard. It only steers the moving ball; the rest of the body can still touch.

**The static pose** (`static_pose_problem`) is the same with one sample and all four balls held, and the cost
`w_tau ||tau||² + w_stay ||dq||² + posture(q)` (`w_stay` keeps it near its guess). Besides the start pose, it gives each
move its starting guess: phase A stays put, then a straight line to the static pose of the next stance.

If a move fails, it is solved once with half the samples and no midpoints, and the full solve starts again from that
path (`_rescue`).

Each problem is built once and re-solved with new numbers (start, holds, target): building costs more than solving.

## Output
Per sample, the 19 motor targets pre-loaded against servo sag: `q_cmd = q + tau / kp`, with
`kp = 400 × 0.0022 = 0.88 N·m/rad` [Microban constants.py]. Times are set after solving: phase A takes 1 s and phase B
2 s, so a move takes 3 s.

## How to run
```bash
git submodule update --init                       # Microban and the organisers' simulator (once)
uv sync --group optimiser --group sim             # casadi (IPOPT), pin (tests), their sim (playback)
uv run pytest tests/test_optimiser.py             # the checks
uv run mjpython tests/test_optimiser.py PLAN          # solve the plan, play it in their sim
uv run mjpython tests/test_optimiser.py PLAN --save   # the same, and save build/trajectory_<wall id>.json
uv run mjpython tests/test_optimiser.py PLAN --replay # play the saved one, without solving
```
- `PLAN` defaults to `build/plan_N1.json`. `--samples N` changes the number of samples; `--swing [W]` turns the swing
  curve on (`w_swing = W`, default 1000); `--headless` runs without a window; `--send PI` also sends each solved move to
  the robot.
- The playback shows the planned poses exactly (no physics). The next move is solved while the current one plays.
- The saved file's `positions` are what the robot executes; `planned_positions` and `trunk_poses` are the planned
  motion, which `--replay` plays.

In code:
```python
from rcr.optimiser import optimise, solve_plan, to_trajectory
from rcr.optimiser.transition import TransitionConfig

traj = optimise(wall, plan)  # defaults
cfg = TransitionConfig(tau_limit=0.5)  # or change settings
results = solve_plan(wall, plan, cfg)  # q, tau, forces per move
traj = to_trajectory(results, cfg)
```
A bad plan raises `ValueError` naming the stance or move; a move IPOPT cannot solve raises `OptimisationError` naming
the move, IPOPT's status and the largest constraint violation.

## Settings (`TransitionConfig`)
| Setting | Default | Why |
|---|---|---|
| `n_samples` (N), `release_sample` (K) | 12, 4 | [Ours] |
| `tau_limit` | 0.424 N·m | 70 % [Ours] of 0.605 N·m, the servo limit (1.75 A × 0.346 N·m/A) as the organisers' sim clips it [Microban] |
| `normals`, `mu` | +z for every limb, 0.8 | the socket opens upwards and holds a ball up [Ours] |
| `contact_offset` | (0, −0.026, −0.0142) m from a hold's bolt midpoint | where a ball rests in the socket, from the organisers' sim and the model's ball |
| `max_trunk_shift`, `max_trunk_turn` | 0.3 m, 1.2 rad | [Ours] |
| `w_tau`, `w_smooth`, `w_stay` | 1, 1, 0.001 | [Ours] |
| `nominal`, `w_nominal` | a hand-designed leg pose (knees bent 20°, legs mirrored); 0.3 | without it the least-torque pose crouches [Ours] |
| `spine_target`, `w_spine` | 25° from vertical; 2 | [Ours] |
| `midpoints`, `midpoint_play` | on, 1 mm | without them, +8 % torque and 4 mm ball drift between samples [Ours] |
| `w_swing`, `swing_lift`, `swing_out` | 0 (off; `--swing` gives 1000), 30 mm, 30 mm | at 1000, a 1 cm miss over phase B's ~16 poses costs about as much as the torques; not yet tuned [Ours] |
| `phase_a_duration`, `phase_b_duration` | 1 s, 2 s | [Ours] |
| `ipopt` | `tol` 1e-6, `max_iter` 150 (continued up to ~1000), `mu_init` 1e-3 | [Ours] |

The kinematics are written in CasADi from the URDF, not with `pinocchio.casadi` (the PyPI `pin` wheel has none); the
tests check them against Pinocchio to machine precision.

## Known limitations
- **No collisions.** A motion can pass a limb through a hold, the wall or the robot itself. The playback shows it; the
  real robot would hit. The swing curve (`--swing`) keeps the moving ball away from them, but not the other links.
- **Quasi-static.** No inertia: slow motion is assumed, and the phase durations are chosen, not optimised.
- **Our contact model.** The holds push only upwards (+z), within friction 0.8; the real socket may hold more.
- **Not yet measured on the climbing model.** Earlier numbers (e.g. a hand could not reach up from 300 mm above the
  feet at 70 % torque, so "both hands up, then both feet up") were with the old limbs.

## Open decisions
- Whether and how to add collision avoidance back (see below), or whether the swing curve is enough; its weight,
  lift and distance from the wall, and whether to have it on by default.
- Time per move (time is the tie-breaker in scoring).
- Whether `Trajectory` needs velocities, per-joint kp or the stance index (marked CHANGE in INTERFACES.md).

## Removed on 30 Oct
To keep the solver simple, these were deleted (they were all off by then). They are in commit `6156612` (branch
`optimiser/cut-feet-and-approach`) if wanted back: collision avoidance, the move cache, the optimiser's feasibility
checks, the settle into each stance's own pose, and reusing a solved move as a starting guess.
