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

## Method (D3)
**Quasi-static direct transcription, one NLP per transition, CasADi + IPOPT, on Microban's URDF.** Reason: each
transition becomes a small, well-posed NLP (~500 variables) whose every sample is a pose the robot can hold, so
torque limits and friction are hard constraints, and it solves in about a second. [Ours]

**Feasibility (D3, "what feasibility checks") [Ours, after FARO]:** `feasibility(wall, a, b)` checks the poses at the
move's edges, cheapest first, not the whole motion: the rules, a reach bound (no two holds of b further apart than
their limbs can span, from the URDF: microseconds, never rejects a reachable stance), stance b held, the release pose (the moving ball in its
old socket carrying nothing), the arrival pose (the same at the new hold) and a lift pose (the ball out of its old
socket, at one of three exits), and finally the move itself, coarsely (6 samples, no midpoints). Each pose is one
static-pose NLP made clear by the exact collision loop; the whole check takes 1–17 s, cached per wall and pose. Failing means the move cannot be done; passing means the full solve is worth trying (the path
between the edges is checked only coarsely). `cost` is the peak |τ| over those poses as a fraction of `tau_limit`.
`edge_checks.trace` also returns the time each check took, for finding what to speed up.

The kinematics are written in CasADi straight from Microban's URDF (`robot_model.py`), not with
`pinocchio.casadi`: the PyPI `pin` wheel has no CasADi bindings. The tests check them against Pinocchio on the same
URDF to machine precision (positions, Jacobians, CoM, gravity torques, `integrate`).

**Each move from stance pose to stance pose, and the move cache [Ours, after FARO]:** every stance has its own pose
(`stance_pose`): balanced, clear, solved from a fresh guess, so it depends on the stance and the holds only. A move
(`solve_move`, on by default with `TransitionConfig.settle`) starts at stance a's pose, ends freely, then a *settle* with
all four balls held brings the robot to stance b's pose. So a move depends on its two stances and the holds around them,
never on the moves before it.

- **Why:** before, each move started wherever the previous one ended. A hold changed far away made move 2 come out
  14° different, and the full re-solve of the example plan then failed at move 9, which had solved before.
- **Measured on the example plan (35 moves):**
  - 34 solve this way; move 32 (right foot to R10) does not;
  - two runs are identical to the last digit;
  - consecutive moves join exactly (2e-7);
  - about 21 minutes in all.

  Pinning the move's own end to stance b's pose, without a settle, solved only 21.
- **The move cache** (`move_cache.py`) stores every move's outcome under its layout:
  - the key is the moving limb, stance a's holds and the new hold as grid offsets from where the limb starts, and
    every other hold in the box around those holds, widened by 0.12 m (0.15 m above);
  - a version covers the hooks, the robot model, the hold mesh and the settings.

  Moving all holds by whole grid holes moves the problem with them, so a move solved on one wall serves any wall where
  its holds are laid out alike. A stored motion is checked again for collisions on the actual wall before it is used.
  A new version (another model, other settings) starts with an empty cache.
- **Filling it:** `scripts/fill_cache.py` climbs generated walls (levels 1–3) and stores every move it asks for,
  feasible or not. It resumes where it stopped. `--no-store` measures how many moves the cache already answers.
- **Every run uses it, with no setting:**
  - `optimise`, `solve_plan` and the runner (`tests/test_optimiser.py --hard PLAN`) take stored moves from
    `build/move_cache/` and add each new one;
  - `feasibility()` answers a stored move at once;
  - rejections (`feasibility()`'s and failed solves) are neither stored nor trusted, for now
    (`move_cache.REJECTIONS = False` [Ours]): a coarse rejection stored by planning used to block the full
    solve of the same move. A move that failed once is checked or solved again;
  - `RCR_MOVE_CACHE=<folder>` uses another cache, and `RCR_MOVE_CACHE=off` none (the tests set it, so they always
    solve).

  The example plan then takes about 25 s through `optimise()`, against over an hour to solve it.
- **Measured:**
  - *Example plan:* move 32 (right foot to R10) cannot be done this way, so it was replaced by right foot to R11. All
    35 moves then solved, in 24 minutes the first time. Re-solving from the cache takes about a second per move and
    gives a byte-identical `trajectory.json`.
  - *After tightening the held-ball drift constraint (cache version 3):* moves 29, 34 and 35 no longer solved; the
    repair search replaced each by another limb first. The plan now has 38 moves (72 minutes to solve, failures
    included). The dense re-check (0.5 mm steps) finds one spot: in move 36, just before the left foot's ball lands
    in L12, a finger edge is 0.13 mm into L12 for an instant (0.05 mm allowed). The solver's 2 mm steps miss it.
    Left as is for now: it is a tenth of a millimetre.
  - *Level 1 ladder* (rules' generator, rungs 0.09 m apart): 13 moves were solved up to 0.82 m. The next 14, to 1.27 m,
    all came from the cache: the same layout one rung higher. In all, the cache answered 85 of the 99 moves the climb
    asked about.
  - *Unseen walls, level 2 (gaps) and level 3 (drift):* a cache filled on level 1 answered only the first moves,
    where the walls are alike. Gaps and drift make new layouts, solved once each, so the cache has to be filled on
    many walls of all three levels.
- **After a model change:** the cache's version changes, so it starts empty. Run `scripts/fill_cache.py` on as many
  generated walls as there is time for.

## Files
| File | What |
|---|---|
| `robot_model.py` | reads the climbing model's URDF (never edited), and its collision shapes and visual meshes from its MJCF; builds the CasADi functions (`limb_positions`, `limb_jacobians`, `com`, `gravity_torques`, `body_points`, `integrate`, `link_frames`, `point_jacobian`) once |
| `transition.py` | `TransitionConfig`, `solve_static_pose`, `solve_transition`; the physics of one sample, each as a small function; the collision loop `_solve_clear` |
| `collision.py` | the hold mesh and its convex pieces, the robot's convex shapes, exact signed distances and their gradients (`coal`), and `check_trajectory` along a motion |
| `edge_checks.py` | `feasibility(wall, a, b)` and `trace`: the checks above, and their cache |
| `move_cache.py` | `MoveCache`, `solve_move_cached`: solved moves stored by layout and reused on any wall |
| `__init__.py` | `optimise` (and `solve_plan`, `start_pose`, `iter_plan`, `to_trajectory`): chains the transitions, reusing solved motions, times them, pre-loads against sag; `feasibility` |

## Formulation
Configuration `q = [trunk position (3), trunk quaternion (w, x, y, z), 19 motors in MICROBAN_JOINTS order]`. Wall
frame as INTERFACES.md: **y into the wall**, robot at y < 0, gravity −z. (The task text said "y out of the wall";
`main` uses y into the wall, so "out of the wall" is −y here.)

Per transition (stance a → b, exactly one limb moves), samples k = 0..N:
- **Variables:** `dq_k` (25), the tangent vector at the start `q0`, `q_k = integrate(q0, dq_k)` as Pinocchio's
  free-flyer; `f_k^c` (3), the force each hold exerts on its ball.
- **Torques** are not variables: `tau_k = g(q_k) − Σ_c J_c(q_k)ᵀ f_k^c` (the 19 motor rows of static balance).
- **Phases:** A, k = 0..K, four contacts, the moving limb's force reaches 0 at K; B, k = K..N, three contacts, the
  moving limb travels and reaches its new hold at N.
- **Constraints:**
  - `dq_0 = 0`;
  - balls in contact on their holds' contact points; the moving ball on the new one at N (not the final pose);
  - trunk balance: `Σ f + m g = 0` and `Σ (x_c − com) × f = 0`;
  - `|tau| ≤ tau_limit`; joint limits from the URDF;
  - a friction pyramid per contact;
  - the moving ball at `y ≤ −clearance` on samples K+2..N−2;
  - (static pose only) every point of Microban's collision boxes at `y ≤ −wall_margin`; in a transition the exact
    check covers the wall (`wall_points` off: those 7200 rows were 70 % of the NLP);
  - the trunk within `max_trunk_shift` / `max_trunk_turn` of `q0`.
- **Cost:** `w_tau Σ ||tau_k||² + w_smooth Σ ||dq_{k+1} − dq_k||²` plus a reference posture at every sample and
  midpoint, as FARO regularises towards a nominal pose: `w_nominal (θ − θ_nom)ᵀ(θ − θ_nom)` on all twelve leg joints
  (left: hip yaw 6°, roll 14°, pitch −63°, knee 20°, ankle pitch 26°, roll 2°; the right leg mirrored) and
  `w_spine (spine_z − cos 25°)²`. The arms (the hand hooks are not mirror images, so mirrored arms cannot seat both)
  and the trunk's position are left to the holds. The first static pose's guess starts the legs in this posture.

`solve_static_pose(wall, stance, q_guess)` is the same with one sample and all four limbs in contact (cost
`w_tau ||tau||² + w_stay ||dq||²`). It gives the plan's first configuration and each transition's warm start (the
static pose of stance b, reached along a straight line in phase B; balanced, but not required to be clear: the
transition's own clearances decide). With one limb unloaded (and optionally moved out of its socket) it gives the
edge poses that `feasibility()` checks.

**Reusing solved motions** (`iter_plan`): a move starts from the same limb's last motion when the holds are laid out
alike (within 2 cm, `same_layout`): that motion shifted by the hold offset, joined to the current start over phase A
(`shifted_seed`), and the collision loop's first step is taken from it with its clearances linearised. If that fails,
the usual start is tried. On a regular wall (rows 75 mm apart, the earlier example, since removed) every move after the first four reuses one.

Each NLP is built once and re-solved with new numbers (start, holds and target are CasADi parameters). A
transition's structure depends only on which limb moves, so a plan builds at most four, plus the static pose.

Output: per sample, the 19 motor targets pre-loaded against sag, `q_cmd = q + tau / kp` with
`kp = KP_DEFAULT × KP_GAIN_PRM = 400 × 0.0022 = 0.88 N·m/rad` [Microban constants.py]. The times come after the solve
(it is quasi-static): `phase_a_duration` over phase A's K steps, `phase_b_duration` over phase B's, cumulative over
the plan.

## Collisions with the holds, the wall and itself (`collision.py`)
**Goal:** no part of the robot enters a hold, the wall or another part of the robot at any moment of the motion, except
the ball resting in its own socket. Nothing about *how* a hook should move is imposed; the exact geometry decides.

- **Geometry, from the sources:**
  - the organisers' simulator's hold mesh (`climbing_sim/assets/hold/hold_visual.stl`), placed as it places it (socket 26 mm out, 14 mm below the bolts; `SOCKET_OUT`, `SOCKET_BELOW` read from it);
  - the robot: the convex hull of each of the model's visual meshes (its collision boxes are smaller than the
    thighs and shins), except on the climbing forearms and feet, which use the model's own convex pieces, plus a
    sphere per ball [Microban MJCF]. A test checks that every drawn vertex is inside;
  - **the feet [Ours]:** the model is our fork of `microban_model`, where the climbing feet are cut as on our robot
    ([robot README](../robot/README.md)). Their 51 pieces (CoACD, at most 64 vertices each) follow the foot to 0.6 mm
    (Rhoban's 24 left parts of its foot up to 22.7 mm outside and bulged up to 7 mm past it), and a self check takes
    12 ms instead of 36. The cut (y 21–48 mm of the foot's frame) is away from where the tibia comes closest
    (y 0–17 mm), so it doesn't change the tibia/foot clearance; it removes the plate's end that hit holds;
  - the wall as the half-space y ≥ 0.

  The hold mesh is closed (every edge shared by two triangles), outward-oriented, and has genus 2 (the bolt holes).
- **Signed distance (`coal`):**
  - outside a hold, the exact mesh distance (e.g. 2.661 mm with the ball at the socket centre, −0.004 mm resting);
  - inside, the depth from the hold cut into 105 convex pieces (CoACD, cached under `build/holds` by the STL's sha256;
    about 20 s once);
  - a point-in-hold test on the pieces' faces, plus the hold's measured thickest part (an inscribed radius of 7.2 mm),
    decide which applies;
  - gradient: `uᵀ ∂x/∂dq` for the nearest point x, with an exact Jacobian from the CasADi model.
- **Continuous in time:** the motion is checked so densely that no robot point moves more than `step` (2 mm) between
  checks. With a 2 mm `margin`, nothing can pass through anything between two checks: a gap to a hold can close by at
  most `step`/2 between two checks that both keep `margin`, and between two robot shapes, both moving, by `step`.
- **Rules per pair:**
  - a ball resting in its socket for a whole interval is the contact itself, so it isn't checked against that hold;
  - a hook, and the link it is fixed on, may touch but not enter its own (old or new) holds. While its ball rests in
    a hold for a whole interval, it may press up to 0.5 mm into that hold (`RESTING_PLAY` [Ours]): that's the contact
    itself. At a touch the exact mesh gives no direction out, so without this a loaded foot's finger 0.13 mm in could
    not be moved out, and a move failed;
  - everything else keeps `margin` from every hold and the wall;
  - **itself [Ours]:** every pair of robot shapes keeps `margin`, except shapes on one link or on the two links of a
    joint (they touch by design), and four pairs on links two joints apart that overlap in every pose (hip and femur,
    tibia and foot, each side: joined by a small block). 2504 pairs; a pair is measured only when its bounding spheres
    are within `near`. Same SCP slots and gradient, the nearest point on each shape moving. Nothing stops the solver
    from crossing the legs or folding an arm into the trunk otherwise; the closest pair on the example climb is the
    trunk and a thigh, 0.5 mm past its margin.
- **Solving: sequential convex programming** [TrajOpt, Schulman et al. 2014], in `transition._solve_clear`:
  - linearise the tightest pairs into fixed constraint slots (parameters, so the NLP is never rebuilt);
  - ℓ1-penalised slack keeps every subproblem solvable;
  - re-solve within a trust region, and accept a step only if the exact merit (cost + 1000 × violation) improves as
    promised;
  - stop when clear and no longer improving. If it can't get clear, it raises an error naming the worst pair.
- **The first solve is steered [Ours], and only the first.** SCP is local, and from the collision-blind solution
  (hands entering the socket sideways) it cannot reach the right posture. The hold's socket opens upwards, so a finger
  leaves the resting ball only near vertical. So the first solve keeps every neck within 45° of up (`NECK_CONE`), and puts the moving ball 35 mm above
  its old and new sockets one sample after leaving and one before arriving (`LIFT`: the hold's top plus the ball's
  radius). Both are switched off before the exact loop. The result must pass the exact check, which these hints have
  no part in.
- **`approach_from_above` [Ours], off by default (runner: `--approach`):** the moving ball's way in, kept in every
  round. It is out from the wall until sample N−3, 35 mm above its new socket within 45° at N−2, and 10 mm above it
  within 10° at N−1 (`SEAT_LIFT`, `SEAT_CONE`), so the last interval is a short drop. Why: the climbing model's ball
  fills its socket (0.2 mm of play) and the socket is a tube, not a funnel (5 mm above the resting point the ball
  touches the rim ~1 mm off-axis, 10 mm above ~3 mm off). Coming in along one long last interval, the ball's path
  curved and grazed the rim by ~2.5 mm on N1's foot moves. The way out keeps the old 35 mm / 45° on the sample after
  release. Needs `n_samples` ≥ `release_sample` + 5.
- **Numbers derived from the hold:** `clearance` (the moving ball's distance out from the wall mid-air) is now
  **0.06 m**, not the task's 0.04 m. The hold sticks out 43 mm and the ball is 12.5 mm in radius, so 0.04 m could not
  pass in front of a hold.

**Cheaper exact checks:** a distance to the hold's convex hull (about 4 µs) bounds the distance to the hold, so shapes
far from it skip the mesh (whose query costs up to 1.5 ms far away). Within reach, the convex pieces (they cover the
mesh to 3 µm and swell past it by at most 0.4 mm) stand unless the shape is within 1 mm of its requirement, where the
exact mesh decides. The moments of a check are split over up to 8 worker processes. One check: 7.7 s → 0.2–0.4 s, same
verdicts.

Measured on the example plan's first row: every transition clear, re-checked independently at 0.5 mm steps (nothing touches another
hold or the wall; the closest pair is a ball settling into its own socket, 0.08 mm deep, at the mesh's precision).
Going around the holds costs torque: the hand moves run at the 0.448 N·m limit.

## Between samples: midpoint constraints
The controller interpolates the joints between samples, so the poses it passes through there must obey the limits
too. Checking only the samples hid +8 % torque and 4 mm of ball drift off the holds between them. So every interval's
midpoint (`midpoints`, on by default) is held to the same rules as a sample:
- the joints are interpolated (no new pose variables);
- it gets its own hold forces (12 variables), because the real distribution is whatever balances there;
- balance, friction, torque and joint limits apply;
- held balls must be within `midpoint_play` (1 mm) of their holds.

That last one isn't an equality: interpolated joints can't keep a closed chain exactly on four points, and the ball has
2.7 mm of play in its socket. The midpoint torques also join the cost. An independent check (the least torque any
balanced set of forces needs at each midpoint) finds 0.436 N·m against the 0.448 limit, and ball drift within the play.

**IPOPT stopping at `max_iter`:** if every constraint is satisfied there (to `FEASIBLE_TOL`, 1e-5; not looser, as
some constraints are squared distances), the point is used. It's a valid motion, just not certified optimal, and the
collision loop's acceptance test decides what's kept. A point that violates a constraint is still an error. The loop's
first solve starts far from its answer, so it may continue from where it stopped (`FIRST_SOLVE_PATIENCE`, up to ~1000
iterations); the rounds after it start next to theirs and stop at 150. Each solve without a reused motion starts from
fresh values (variables and multipliers at zero), so a move's result does not depend on what was solved before.

## Parameters (`TransitionConfig`)
| Parameter | Default | Tag |
|---|---|---|
| `n_samples` (N), `release_sample` (K) | 12, 4 | [Ours] |
| `tau_limit` | 0.7 × 0.605 = 0.424 N·m | 0.605 N·m: 1.75 A × 0.346 N·m/A, as the organisers' sim clips the servos [Microban]; 70 % [Ours] |
| `normals` | +z for all four limbs: the socket opens upwards and holds a resting ball up | from the sim's hold mesh [Ours] |
| `mu` | 0.8 per limb | [Ours] |
| `clearance` | 0.06 m, moving ball out from the wall on samples K+2..N−2 | from the hold's depth + ball [Ours] |
| `avoid_collisions` | True | |
| `collision_clearance` | `margin` 2 mm, `step` 2 mm, `near` 20 mm, `aim` 0.5 mm (`collision.Clearance`) | [Ours] |
| `collision_slots`, `w_collision` | 40 per interval, 1000 per m | [Ours] |
| `trust_radius`, `scp_iterations`, `scp_polish`, `retry_trust_radius` | 0.05, 30, 0, 0.02 (one more try from a smaller trust region) | [Ours] |
| `approach_from_above` | False: the moving ball 35 mm above its sockets, one sample after leaving and one before arriving, in the first solve only. True: in every round, and the way in in two steps (35 mm / 45° at N−2, 10 mm / 10° at N−1) | [Ours]; 10°: [the organisers' sim, `wall.py`] |
| `midpoints`, `midpoint_play` | True, 1 mm | [Ours] |
| `wall_margin`, `wall_points` | 0.002 m for every collision-box point; in transitions off (the exact check covers the wall) | [Ours] |
| `contact_offset` | (0, −0.026, −0.0142) m from `Hold.position`: the ball at the bottom of the hold's socket | [Rules Figure 3], as the sim |
| `max_trunk_shift`, `max_trunk_turn` | 0.3 m, 1.2 rad per transition | [Ours] |
| `w_tau`, `w_smooth`, `w_stay` | 1, 1, 0.001 | [Ours] |
| `nominal`, `w_nominal` | the twelve leg joints of a hand-designed pose on the example's start (legs mirrored, knees bent 20°, hips 100 mm off the wall, clear of the holds, 0.15 N·m); weight 0.3 per rad², which keeps the legs within a few degrees of it where the stance allows | [Ours] |
| `spine_target`, `w_spine` | 25° from vertical; weight 2 on `(spine_z − cos spine_target)²` | [Ours] |
| `phase_a_duration`, `phase_b_duration` | 1 s, 2 s | [Ours] |
| `ipopt` | `tol` 1e-6, `max_iter` 150, `mu_init` 1e-3 (rounds start next to their answer), silent; the static pose starts cold: IPOPT's own `mu_init` and 1000 iterations | [Ours] |

The balls come from `rcr.robot.hooks`, shared with the sim; `contact_offset` follows their radius.

## How to run
```bash
git submodule update --init                    # Microban, in third_party/microban (once)
uv sync --group optimiser --group sim          # casadi (IPOPT), coacd, coal, pin (tests); their sim for the runner
uv run pytest tests/test_optimiser.py                     # solves the example plan's first moves, checks them
uv run mjpython tests/test_optimiser.py --hard PLAN           # solves the plan, plays it in the sim
uv run mjpython tests/test_optimiser.py --hard PLAN --save    # the same, and saves build/trajectory_<wall id>.json
uv run mjpython tests/test_optimiser.py --hard PLAN --replay  # plays the saved trajectory, without solving
```
PLAN defaults to `build/plan_N1.json`. `--samples N`, `--clearance M` and `--approach` set `n_samples`, `clearance`
and `approach_from_above` for the run. `RCR_MOVE_CACHE=off` solves every move afresh, ignoring stored ones.
The runner (`tests/test_optimiser.py`, run directly) reads a contact plan file and its wall, the organisers' wall
named by the plan (`data/examples/<wall_id>.json`, played on their wall of that name in their sim). There is no
example plan yet. It solves each transition in a worker process and plays it in the sim as soon as
it's ready, while the next one is solved; between them the robot stays where it ended. The playback is kinematic:
the robot follows the optimiser's poses exactly, with no physics. Add `--headless` to run without a window.

**Saving and replaying [Ours]:** `--save` writes `build/trajectory_<wall id>.json` (`trajectory/2`) after every solved
transition, so a solve that stops part way keeps what it solved.
- `positions` is what the robot executes: motor targets pre-loaded against sag. Send it over the link.
- `planned_positions` and `trunk_poses` are the planned motion itself, which `--replay` plays. The robot ignores them.
```python
from rcr.optimiser import optimise, solve_plan, to_trajectory
from rcr.optimiser.transition import TransitionConfig

traj = optimise(wall, plan)  # defaults
results = solve_plan(
    wall, plan, TransitionConfig(tau_limit=0.5)
)  # or tune, and inspect q, tau, forces
traj = to_trajectory(results, TransitionConfig(tau_limit=0.5))
```
Errors say what failed: `ValueError` for a bad plan (naming the stance or transition); `OptimisationError`, naming the
transition, IPOPT's status and the largest constraint violation, when IPOPT finds no solution. Without the group or
Microban, `run_pipeline.py --check` reports the stage as `skip`.

## Measured (MacBook, Apple silicon)
On the earlier regular example (since removed: 12 moves, rows 75 mm apart), the whole plan, with collisions and midpoints, at 70 % torque: **116 s**. The first
move of each limb takes 11–41 s (about 9 s of it, once per limb, is CasADi building the NLP); the moves that reuse a
solved motion take 1–6 s, except one left-foot move at 19 s. Before the speed-ups the same plan took about 30 minutes.
In the collision loop each round is one IPOPT solve (~20 ms per iteration) plus one exact check (0.2–0.4 s).

`feasibility()`: 0.2–4.6 s per move from cold, nothing once cached. On the regular example it passes all 12 solved
moves. On `wall_hard.json` a search checked 337 candidate moves in 61 s and rejected 145 of them; 10 of the moves it
passed then failed the full solve, nearly all a shin or foot sweeping through the hold just above a foot while another
limb moved, so a coarse path check (`transition.coarse`: 6 samples, no midpoints) became the last tier, and the last
fallback of `solve_transition` (it rescued a hand move that failed from every other start). Even so that wall had no
plan to the top: R2 and L2 sat where the shins sweep. With them moved 15 mm outside the foot columns, a search found a
12-move plan (`contact_plan_hard.json`): 38 feasibility checks (18 rejected) in 253 s, then the full solve of all 12
moves in 270 s (5–55 s each, nothing to reuse on uneven rows) with no failure. Feasibility now takes 1–17 s per move.

With self-collision on, the regular plan takes **86 s**. The first move per limb takes 10–27 s, and reused moves 1–7 s.
The same search on `wall_hard.json` (`edge_checks.trace`) finds the same 12 moves. It makes 38 checks with 18 rejected:
6 at the stance, 3 at the release and 9 at the path. The checks took 325 s in all:
- the coarse path: 284 s, up to 45 s per move;
- the edge poses: 41 s;
- the reach tier: 0 s, rejecting none of the 38.

Of the 292 candidate stances the search's hand-picked span limits cut, the reach bound cuts 11. It is sound, but it
can't replace them. The remaining cost is the path tier.

## Physical facts to design around
- **Stand on the feet; the posture is a choice, not forced.** With the feet supported only by pushing out from the
  wall (the first model), the robot leaned out 44° and hung from its arms. With the sockets holding every limb up, the
  least-torque pose crouches (knees at 135° on the example's start), yet poses with the knees within 45° cost only a
  third of the torque limit. So a reference posture decides the shape: on the example the knees stay near 50°, the
  spine 23–28° from vertical and the hips 70–105 mm off the wall, at 0.14–0.17 N·m.
- **Torque-bound, not reach-bound.** With rows 0.15 m apart (the first example wall), the feet can move up a row, but a
  hand reaching up one needs more than 70 % of the servo torque: infeasible at the 70 % limit, with the feet on either
  row. That's why the example wall now has rows 75 mm apart.
- **Hands reach up only while the body is compact.** On 75 mm rows, at 70 %, with the limits held along the whole
  motion: a hand can move up from 225 mm above the feet (to 300 mm), but not from 300 mm (to 375 mm). So the climbing
  pattern that works is **both hands up, then both feet up**, and repeat. Moving the feet first doesn't help.
  `feasibility()` tells the planner this through the release pose.
  - XL330 limit ≈ **0.605 N·m** (the firmware current limit 1.75 A × BAM's kt 0.346 N·m/A) [Microban].
- **Servos sag under load.** The servo runs its own P loop: `KP_DEFAULT` = 400 ≈ 0.88 N·m/rad, so the position
  error is roughly τ / kp, about **0.34 rad (19°) at 0.3 N·m**. Hence the pre-load above.
- The climbing model weighs 0.808 kg [Microban]; the rules allow up to 2 kg [Rules §2.2].

## Known limitations
- **Two rungs at once is too far for a hand.** Both level 2 walls generated with gaps (rules' `level2`, seeds 0 and
  1) stopped the climb: one missing hold left a hand 0.18 m short of the next, which fails the checks at 70 % torque
  with the old limbs; not measured again with the climbing model yet.
- **The cache only helps with layouts it has seen.** A gap or a drift is a new layout, solved once (30 s – 3 min).
- **Not yet redone on the climbing model (29 Oct):** the measurements above and the example trajectory are from the old
  model. The example plan does not solve yet (a foot 2.6 mm too close to its shin). Then:
  `tests/test_optimiser.py --hard PLAN --save`.
- **Contact normals and mu are ours:** hands are supported only along +z; the real socket may hold more.
- **The robot's shapes** are convex hulls (a little larger where a part is not convex: conservative); the climbing
  limbs' pieces and balls lie within 0.21 mm of the drawn mesh.
- **SCP is local:** it needs the steered first solve. A plan whose moves can't be started that way may fail, with an
  error naming the pair; a sampling planner (RRT in the same `coal` scene) for the first path is the next remedy.
- **The first move of each limb is slow** (11–41 s, see Measured): nothing to reuse yet, and the NLP is built then.
  Saving solved motions and the built NLPs across runs would remove most of it. Hard moves near the torque limit can
  take minutes.
- **A hook leaving its own socket has no margin**, so the 2 mm checking step does not guarantee it between checks: an
  independent 4× denser check once found a ball 0.09 mm into its old socket.
- **Quasi-static only:** no inertia, so slow motion is assumed; the phase durations are chosen, not optimised.
- **Near the edge of feasibility IPOPT can stop at `max_iter`** at a point that isn't feasible, without proving
  infeasibility; the error says so.
- **`feasibility()` is not a guarantee:** its path is coarse (6 samples, no midpoints), so a move it passes can still
  fail the full solve, and its solves are local, so it could reject a move that is possible (none of the 15 moves the
  full solve has done is rejected).

## Open decisions
- Time per move (time is the tie-breaker in scoring).
- Whether `Trajectory` needs velocities, per-joint kp or the stance index (marked CHANGE in INTERFACES.md).
