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

The kinematics are written in CasADi straight from Microban's URDF (`robot_model.py`), not with
`pinocchio.casadi`: the PyPI `pin` wheel has no CasADi bindings. The tests check them against Pinocchio on the same
URDF to machine precision (positions, Jacobians, CoM, gravity torques, `integrate`).

## Files
| File | What |
|---|---|
| `robot_model.py` | reads the URDF (never edited) and Microban's collision shapes from its MJCF; adds the hooks; builds the CasADi functions (`limb_positions`, `limb_jacobians`, `com`, `gravity_torques`, `body_points`, `integrate`, `link_frames`, `point_jacobian`) once |
| `transition.py` | `TransitionConfig`, `solve_static_pose`, `solve_transition`; the physics of one sample, each as a small function; the collision loop `_solve_clear` |
| `collision.py` | the hold mesh and its convex pieces, the robot's convex shapes, exact signed distances and their gradients (`coal`), and `check_trajectory` along a motion |
| `__init__.py` | `optimise` (and `solve_plan`, `to_trajectory`): chains the transitions, times them, pre-loads against sag |

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
  - every point of Microban's collision shapes at `y ≤ −wall_margin` in every sample;
  - the trunk within `max_trunk_shift` / `max_trunk_turn` of `q0`.
- **Cost:** `w_tau Σ ||tau_k||² + w_smooth Σ ||dq_{k+1} − dq_k||²`.

`solve_static_pose(wall, stance, q_guess)` is the same with one sample and all four limbs in contact (cost
`w_tau ||tau||² + w_stay ||dq||²`). It gives the plan's first configuration and each transition's warm start (the
static pose of stance b, reached along a straight line in phase B). It is also the cheap check `feasibility()` can
use: a stance it cannot solve cannot be held at all.

Each NLP is built once and re-solved with new numbers (start, holds and target are CasADi parameters). A
transition's structure depends only on which limb moves, so a plan builds at most four, plus the static pose.

Output: per sample, the 19 motor targets pre-loaded against sag, `q_cmd = q + tau / kp` with
`kp = KP_DEFAULT × KP_GAIN_PRM = 400 × 0.0022 = 0.88 N·m/rad` [Microban constants.py]. The times come after the solve
(it is quasi-static): `phase_a_duration` over phase A's K steps, `phase_b_duration` over phase B's, cumulative over
the plan.

## Collisions with the holds and the wall (`collision.py`)
**Goal:** no part of the robot enters a hold or the wall at any moment of the motion, except the ball resting in its own
socket. Nothing about *how* a hook should move is imposed; the exact geometry decides.

- **Geometry, from the sources:**
  - the organisers' hold mesh `data/holds/hold.stl` [Rules v1.1 Figure 3], placed as the sim does;
  - Microban's 36 collision shapes [Microban MJCF] plus our hooks (a capsule per finger segment, a sphere per ball);
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
  checks. With a 2 mm `margin`, nothing can pass through anything between two checks.
- **Rules per pair:**
  - a ball resting in its socket for a whole interval is the contact itself, so it isn't checked against that hold;
  - a hook, and the link it is fixed on, may touch but not enter its own (old or new) holds;
  - everything else keeps `margin` from every hold and the wall.
- **Solving: sequential convex programming** [TrajOpt, Schulman et al. 2014], in `transition._solve_clear`:
  - linearise the tightest pairs into fixed constraint slots (parameters, so the NLP is never rebuilt);
  - ℓ1-penalised slack keeps every subproblem solvable;
  - re-solve within a trust region, and accept a step only if the exact merit (cost + 1000 × violation) improves as
    promised;
  - stop when clear and no longer improving. If it can't get clear, it raises an error naming the worst pair.
- **The first solve is steered [Ours], and only the first.** SCP is local, and from the collision-blind solution
  (hands entering the socket sideways, as the L-shaped hook does by default) it cannot reach the right posture. The
  hold's socket opens upwards: a 9 mm finger leaves the resting ball only within about 60° of vertical (measured on
  `hold.stl`). So the first solve keeps every neck within 45° of up (`NECK_CONE`), and puts the moving ball 35 mm above
  its old and new sockets one sample after leaving and one before arriving (`LIFT`: the hold's top plus the ball's
  radius). Both are switched off before the exact loop. The result must pass the exact check, which these hints have
  no part in.
- **Numbers derived from the hold:** `clearance` (the moving ball's distance out from the wall mid-air) is now
  **0.06 m**, not the task's 0.04 m. The hold sticks out 43 mm and the ball is 10 mm in radius, so 0.04 m could not
  pass in front of a hold.

Measured on the test ladder: every transition clear, re-checked independently at 0.5 mm steps (zero violations; the
closest pair is a ball touching its own socket). One row (4 transitions) takes about 80 s, against about 6 s without
collisions. Peak torque rises to the 0.448 N·m limit: going around the holds costs torque.

## Parameters (`TransitionConfig`)
| Parameter | Default | Tag |
|---|---|---|
| `n_samples` (N), `release_sample` (K) | 12, 4 | [Ours] |
| `tau_limit` | 0.7 × 0.64 = 0.448 N·m | 0.64 N·m [Microban constants.py: 1.75 A × 0.366 N·m/A]; 70 % [Ours] |
| `normals` | feet −y (out of the wall), hands +z (the hold holds the hand up) | placeholders [Ours] |
| `mu` | 0.8 per limb | [Ours] |
| `clearance` | 0.06 m, moving ball out from the wall on samples K+2..N−2 | from the hold's depth + ball [Ours] |
| `avoid_collisions` | True | |
| `collision_clearance` | `margin` 2 mm, `step` 2 mm, `near` 20 mm, `aim` 0.5 mm (`collision.Clearance`) | [Ours] |
| `collision_slots`, `w_collision` | 40 per interval, 1000 per m | [Ours] |
| `trust_radius`, `scp_iterations`, `scp_polish` | 0.05, 30, 3 | [Ours] |
| `wall_margin` | 0.002 m for every collision-shape point, every sample | [Ours] |
| `contact_offset` | (0, −0.026, −0.0167) m from `Hold.position`: the bottom of the hold's socket | [Rules Figure 3], as the sim |
| `max_trunk_shift`, `max_trunk_turn` | 0.3 m, 1.2 rad per transition | [Ours] |
| `w_tau`, `w_smooth`, `w_stay` | 1, 1, 0.001 | [Ours] |
| `phase_a_duration`, `phase_b_duration` | 1 s, 2 s | [Ours] |
| `ipopt` | `tol` 1e-6, `max_iter` 1000, silent | [Ours] |

The hooks (`HOOKS` in `robot_model.py`) are the sim's: the same balls, to 0.01 mm, and the same 8 g each (a test
checks both). Replace them, in that one place, when the climbing limbs are published [Rules §2.5].

## How to run
```bash
uv sync --group optimiser                      # casadi (IPOPT), coacd (hold pieces), pin (tests)
export MICROBAN_URDF=/path/to/microban/src/model/urdf/robot.urdf    # Rhoban/microban, tested at d0a128d
uv run pytest tests/test_optimiser.py
```
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
`MICROBAN_URDF`, `run_pipeline.py --check` reports the stage as `skip`.

## Measured (MacBook, Apple silicon)
On the test ladder in `tests/test_optimiser.py` (rows 75 mm apart, hands 300 mm above the feet), one row up:
| Transition | IPOPT, problem built | Iterations |
|---|---|---|
| right foot, left foot | 0.6–0.7 s | 36–41 |
| right hand, left hand | 1.5–2.6 s | 78–123 |

The first transition of each moving limb also builds its problem: about 2.5 s more. The static pose is about 0.4 s.
A transition NLP has 481 variables and 4801 constraints (most are the 288 collision points × 13 samples).

## Physical facts to design around
- **Torque-bound, not reach-bound.** On `data/examples/wall_l1.json` (rows 0.15 m apart) the feet can move up a row,
  but a hand needs the full 0.64 N·m to move up one with the feet a row up: infeasible at the 70 % limit.
  - XL330 limit ≈ **0.64 N·m** (Microban's firmware current limit 1.75 A × kt 0.366 N·m/A) [Microban constants.py].
- **Servos sag under load.** The servo runs its own P loop: `KP_DEFAULT` = 400 ≈ 0.88 N·m/rad, so the position
  error is roughly τ / kp, about **0.34 rad (19°) at 0.3 N·m**. Hence the pre-load above.
- The robot weighs 0.819 kg [Microban], 0.851 kg with our hooks; the rules allow up to 2 kg [Rules §2.2].

## Known limitations
- **Placeholder hands and holds:** the hooks, the contact normals and mu are ours until the climbing limbs and holds
  are published. Hands are supported only along +z; the real socket may hold more.
- **No self-collision yet.** Holds and the wall are avoided exactly and continuously (see above); robot against robot
  is not checked. The same machinery (`coal`, the slots) would add it.
- **The robot's shapes are Microban's approximations** (boxes and cylinders), plus our placeholder hooks.
- **SCP is local:** it needs the steered first solve. A plan whose moves can't be started that way may fail, with an
  error naming the pair; a sampling planner (RRT in the same `coal` scene) for the first path is the next remedy.
- **Slower:** about 20 s per transition with collisions.
- **Quasi-static only:** no inertia, so slow motion is assumed; the phase durations are chosen, not optimised.
- **The example plan is infeasible:** its stance 1 (a hand on the top row, the feet on the bottom one) cannot even
  be held, so with `MICROBAN_URDF` set `--check` fails at the optimiser stage. Fixing the example is an interface
  change, left for review.
- **Near the edge of feasibility IPOPT can stop at `max_iter`** instead of proving infeasibility; the error says so.
- `feasibility()` is still a stub. `solve_static_pose` (can b be held?) and then `solve_transition` (can a → b be
  done?) are the obvious implementation.

## Open decisions
- What `cost` in `Feasibility` means (e.g. the transition's peak |tau| over `tau_limit`).
- Time per move (time is the tie-breaker in scoring).
- Whether `Trajectory` needs velocities, per-joint kp or the stance index (marked CHANGE in INTERFACES.md).
