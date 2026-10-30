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
| [`robot_model.py`](robot_model.py) | the robot's kinematics in CasADi, read from Microban's URDF: ball positions, Jacobians, centre of mass, gravity torques; the body's keep-out points, from its MJCF meshes |
| [`reference.py`](reference.py) | a reference motion to follow (e.g. an IK trajectory): its wall, plan and timing read from it, cut onto each move's samples |
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
| every sample: joint limits, \|torque\| ≤ `tau_limit` (85 % of the servo limit) | `balanced_pose` → `joint_and_torque_limits` |
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
+ w_swing  · Σ ||ball(q) − r(s)||²      phase B's samples and midpoints   (the moving ball near its swing path)
+ w_keep_out · Σ intrusion(q)           over the samples and midpoints    (the body out of the cups' layer)
+ w_hook     · Σ hook_tilt(q)           over the samples and midpoints    (each hook pointing up from its ball)
+ w_track    · Σ ||S (dq − dq_ref)||²   over the samples                  (near a reference motion; only with --track)
```
`posture(q) = w_nominal · Σ (leg joint − its nominal angle)² + w_spine · (spine_z − cos 25°)²`.

**The swing path** `r(s)` (`swing_point`) is a reference for the moving ball, so no collision constraints are needed.
`s = (k − K)/(N − K)` runs from 0 (the ball leaves its hold) to 1 (on its new hold); midpoints are at half steps.
`r` is straight lines through six points (`a` the old contact point, `b` the new one, `up = swing_lift·ẑ`,
`out = −swing_out·ŷ`, away from the wall; `v = swing_vertical`):

| s | point | the ball |
|---|---|---|
| 0 | `a` | in the old cup |
| v | `a + up` | straight up, clear of the rim |
| 1.5 v | `a + up + out` | out from the wall |
| 1 − 1.5 v | `b + up + out` | across, as far out |
| 1 − v | `b + up` | in, above the new cup |
| 1 | `b` | straight down into it |

With v = 0.25 and N − K = 8, that is 2 sample intervals up, 1 out, 2 across, 1 in and 2 down. `ball(q)` is the moving
ball's centre (forward kinematics). It's a cost, so the ball is pulled towards the path, not held on it: `w_swing` sets
how hard. The ball clears a cup's rim about 17 mm up; with the defaults, the path stays at least 7.7 mm from every cup
of N1 for every move to one of a hold's 8 nearest holds. It only steers the moving ball; the rest of the body can still
touch.

**Keeping the body off the cups** (`intrusion`). Every cup is in a thin layer in front of the wall: from the wall
face (y = 0) to `CUP_FRONT` = −43 mm, the front of the organisers' sim's cup. The body is pushed out of it at every
sample, midpoint and static pose, through the 8 corners of a box around each part (`robot_model.keep_out_points`: 20
parts, 160 corners, from the visual meshes of Microban's MJCF):

```
intrusion(q) = Σ over the corners  max(0, y(corner) − (CUP_FRONT − cup_clearance))²
```

A corner `cup_clearance` (5 mm) or more in front of the cups costs nothing; one further in costs the square of how far.
A box contains its part, so with its corners out, the whole part is out: no collision checks, wherever the cups are.
A forearm's or foot's box leaves out its hook and ball end (within `BALL_ZONE` = 50 mm of the ball), which must go into
the cup. The boxes are 2–5 mm larger than the head and arms along y, up to 24 mm for the trunk. It is a cost, not a
constraint: as 4,000 constraints (1,916 → 5,916 per move), the first move of N2 did not solve in 5 minutes. As a cost,
the solver may still accept a small intrusion where that saves a lot of torque.

**Hooks from above** (`hook_tilt`). The ball must go into its cup from above, so the hook above it must point up:
the hands' hooks are straight shafts along the forearm, the feet's a short stub above the ball (`robot_model.hook_axes`,
measured on the meshes between the ball's surface and 30 mm from its centre). With `u` a hook's direction in the wall
frame:

```
hook_tilt(q) = Σ over the four limbs  max(0, hook_up − u_z)²        hook_up = 0.5: at most 60° from up
```

The organisers' own hanging pose (`reset_hanging`, N2) costs nothing (hands 51° from up, feet 11°). Without it, the
solved hands lay flat into the cups from the front (82° from up) or came from below them.

**Following a reference motion** (`--track FILE`, `reference.py`). A whole-body motion made elsewhere, e.g. by IK
(`q`, T × 26 at 50 Hz, our layout), which places the balls but knows nothing of balance or torques, is made
feasible: each move stays as close to it as the constraints allow. From the file:
- its **wall** (N1–N3, the one its balls are on) and its **plan**: a stance wherever all four balls are within 3 mm
  of holds, a move wherever one changes;
- each move cut where the moving ball leaves its hold: the held part onto phase A's samples, the part in the air onto
  phase B's; and its **timing** (the medians over its moves become the phase durations).

The cost gets `w_track · Σ_k ||S (dq_k − dq_ref,k)||²` (`w_track` = 10; `S` counts the trunk's position 1 cm as
0.1 rad, `track_position_scale`), the reference is the starting point, and the start pose stays near its first pose.
The costs that shape the motion themselves (swing, keep-out, hooks, posture) are off: the reference decides it. After
each move, the runner prints how far it ended from the reference.

**The static pose** (`static_pose_problem`) is the same with one sample and all four balls held, and the cost
`w_tau ||tau||² + w_stay ||dq||² + posture(q) + w_keep_out · intrusion(q) + w_hook · hook_tilt(q)` (`w_stay` keeps it near its guess). Besides the start pose, it gives each
move its starting guess: phase A stays put, then a straight line to the static pose of the next stance.

If a move fails, it is solved once with half the samples and no midpoints, and the full solve starts again from that
path (`_rescue`).

Each problem is built once and re-solved with new numbers (start, holds, target): building costs more than solving.

## Output
Per sample, the 19 motor targets pre-loaded against servo sag: `q_cmd = q + tau / kp`, with
`kp = 400 × 0.0022 = 0.88 N·m/rad` [Microban constants.py]. Times are set after solving: phase A takes 1 s and phase B
2 s, so a move takes 3 s.

## How to run
**Simplest:** set the weights and what to solve at the top of [`run.sh`](run.sh), then `bash src/rcr/optimiser/run.sh`.
It prints the full command it runs; the options below are what it uses. `--set NAME=VALUE` sets any number setting of
`TransitionConfig`, last, so it wins.

```bash
git submodule update --init                       # Microban and the organisers' simulator (once)
uv sync --group optimiser --group sim             # casadi (IPOPT), pin (tests), their sim (playback)
uv run pytest tests/test_optimiser.py             # the checks
uv run mjpython tests/test_optimiser.py PLAN          # solve the plan, play it in their sim
uv run mjpython tests/test_optimiser.py PLAN --save   # the same, and save build/trajectory_<wall id>.json
uv run mjpython tests/test_optimiser.py PLAN --replay # play the saved one, without solving
```
- `PLAN` defaults to `build/plan_N1.json`. `--tau-limit F` sets the torque limit as a fraction of the servo's (default 0.85). `--track FILE` follows a reference motion instead of a plan (above):
  `uv run mjpython tests/test_optimiser.py --track iktrajv0.npz --save`. `--samples N` changes the number of samples; `--swing W` sets the swing
  path's weight (`w_swing`, default 5000; 0 turns it off); `--keep-out W` sets the keep-out's weight (`w_keep_out`, default 20000; 0 turns it off); `--hook W` sets the hooks' (`w_hook`, default 10; 0 turns it off); `--headless` runs without a window; `--send PI` also sends each solved move to
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
| `tau_limit` | 0.514 N·m | 85 % [Ours] of 0.605 N·m (`--tau-limit F` for another fraction; it was 70 %, at which the IK climb of N2 put a shoulder at the limit for whole moves), the servo limit (1.75 A × 0.346 N·m/A) as the organisers' sim clips it [Microban] |
| `normals`, `mu` | +z for every limb, 0.8 | the socket opens upwards and holds a ball up [Ours] |
| `contact_offset` | (0, −0.026, −0.0142) m from a hold's bolt midpoint | where a ball rests in the socket, from the organisers' sim and the model's ball |
| `max_trunk_shift`, `max_trunk_turn` | 0.3 m, 1.2 rad | [Ours] |
| `w_tau`, `w_smooth`, `w_stay` | 1, 1, 0.001 | [Ours] |
| `nominal`, `w_nominal` | a hand-designed leg pose (knees bent 20°, legs mirrored); 0.3 | without it the least-torque pose crouches [Ours] |
| `spine_target`, `w_spine` | 25° from vertical; 2 | [Ours] |
| `midpoints`, `midpoint_play` | on, 1 mm | without them, +8 % torque and 4 mm ball drift between samples [Ours] |
| `w_swing` | 5000 | not yet tuned [Ours] |
| `swing_lift`, `swing_out`, `swing_vertical` | 25 mm, 40 mm, 0.25 | clear of every cup on N1 by ≥ 7.7 mm (above) [Ours] |
| `w_track`, `track_position_scale` | 0 (10 with `--track`), 10 rad/m | at 10, a 3° miss on every joint costs ~0.5 per pose against ~0.075 of torque [Ours] |
| `w_hook`, `hook_up` | 10, 0.5 (60° from up) | on N2, a too flat hook cost ~1.7 per pose against ~0.075 of torque; not tuned [Ours] |
| `w_keep_out`, `cup_clearance` | 20000, 5 mm | a corner 1 cm past the limit costs about a whole move's torque (~2 on N2); not tuned further [Ours]; `CUP_FRONT` from the organisers' sim |
| `phase_a_duration`, `phase_b_duration` | 1 s, 2 s | [Ours] |
| `ipopt` | `tol` 1e-6, `max_iter` 150 (continued up to ~1000), `mu_init` 1e-3 | [Ours] |

The kinematics are written in CasADi from the URDF, not with `pinocchio.casadi` (the PyPI `pin` wheel has none); the
tests check them against Pinocchio to machine precision.

## Known limitations
- **No collision model.** The body is pushed out of the cups' layer and the swing path steers the moving ball, both
  as costs (not guaranteed), and the hooks next to a held ball (within 50 mm of it) may still touch a cup, and the robot is not checked against itself.
  The playback shows it; the real robot would hit.
- **Quasi-static.** No inertia: slow motion is assumed, and the phase durations are chosen, not optimised.
- **Our contact model.** The holds push only upwards (+z), within friction 0.8; the real socket may hold more.
- **Not yet measured on the climbing model.** Earlier numbers (e.g. a hand could not reach up from 300 mm above the
  feet at 70 % torque (the limit then), so "both hands up, then both feet up") were with the old limbs.

## Open decisions
- Whether and how to add collision avoidance back (see below), or whether the swing path is enough; its weight.
- Time per move (time is the tie-breaker in scoring).
- Whether `Trajectory` needs velocities, per-joint kp or the stance index (marked CHANGE in INTERFACES.md).

## Removed on 30 Oct
To keep the solver simple, these were deleted (they were all off by then). They are in commit `6156612` (branch
`optimiser/cut-feet-and-approach`) if wanted back: collision avoidance, the move cache, the optimiser's feasibility
checks, the settle into each stance's own pose, and reusing a solved move as a starting guess.
