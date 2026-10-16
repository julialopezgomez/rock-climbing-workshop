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

## Physical facts to design around
- **Torque-bound, not reach-bound.**
  - XL330 limit ≈ **0.64 N·m** (Microban's firmware current limit 1.75 A × kt 0.366 N·m/A) [Microban constants.py].
  - Climbing poses with the body hanging off the arms can peak near this limit.
- **Servos sag under load.**
  - The servo runs its own P loop: Microban's `KP_DEFAULT` = 400 ≈ 0.886 N·m/rad.
  - So the position error is roughly τ / kp, about **0.34 rad (19°) at 0.3 N·m**.
  - A trajectory that ignores sag will miss holds.
- The robot weighs ≈ 0.82 kg [Microban]; the rules allow up to 2 kg [Rules §2.2].

## v0 could be
- **Trajectory:** one IK keyframe per stance (placo is already used by Microban), linearly interpolated over a fixed
  time per move.
- **Feasibility:** reachability plus a static torque estimate against 0.64 N·m with a margin.

## Candidates later (not decided)
- IK interpolation with gravity/sag compensation;
- trajectory optimisation;
- a learned policy (then the "trajectory" may become a reference plus a policy id: an interface change).

## Open decisions
- Method (D3, decide by Tue 20 Oct).
- What `cost` means.
- Time per move (time is the tie-breaker in scoring).
- Whether `Trajectory` needs velocities or gains (marked CHANGE in INTERFACES.md).
