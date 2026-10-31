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
  hold b (`ball_path`). It's a corridor clear of every hold: five straight pieces, each a degree-1 Bézier
  (`bezier.py`), at one constant speed (the corridor's length / `MOVE_TIME`). Each piece lasts its share of the
  length. The speed is the same everywhere, but the direction turns at each corner, and the ball starts and stops at
  full speed. The pieces are straight, so the path can't cut a corner into a hold or rise above where it's going:
  - **up** `ABOVE` out of socket a. A ball leaves a socket only within ~10° of vertical [simulator `wall.py`,
    `EXIT_BLOCK`];
  - **out** `OUT` from the wall;
  - **across** to in front of socket b;
  - **in** to above socket b;
  - **down** into socket b.
- **The IK step is a QP** (`ik_step_qp`, solved with `quadprog` through the ediadvancedrobotics tutorials' helper
  `quadprog_solve_qp`). Each step chooses the change of configuration v that:
  - **minimises** the four balls' errors (`‖J v − displacement‖²`; a held ball weighs `HELD_WEIGHT`, the moving
    one 1), plus `REGULARISATION ‖v‖²` (as `DAMPING` did), minus a gradient step up the moving limb's log
    manipulability (Yoshikawa's `w = √det(J Jᵀ)` over its own joints, forward differences, `MANIPULABILITY_GAIN`);
  - **must stay within** the joint limits (one Euler step) and the URDF's speed limits (6.6 rad/s, per IK step: over
    `DT / ITERATIONS`);
  - **tries to stay** `JOINT_MARGIN` inside the joint limits and under `SPEED_MARGIN` of the speed limits, through
    slack variables ≥ 0 penalised by `SLACK_WEIGHT`.
  For now it has the balls only: no trunk or posture task, so `height` and `posture` are ignored.
  `ik_step_pinv` (below) is the previous IK, kept for comparison.
- **`ik_step_pinv`, the previous IK:** the whole body follows by the planner's hierarchical IK (`astar.ik`, copied here, not imported). The held
  balls' tasks come first, in `TASK_ORDER`, then the moving ball's, each in the null space of the ones before, then
  the trunk upright and facing the wall. At each sample, `ITERATIONS` steps each close `GAIN` of every ball's error
  to its target. The moving ball's target is on the path; the held balls' targets are their sockets.
- **Out of reach** [Ours]: the held balls are solved exactly (their sockets are always in reach). The moving ball
  and the trunk use a damped least-squares inverse (`DAMPING`), so where the path is out of the limb's reach the
  moving ball only gets as close as it can, while the held balls stay in their sockets. With plain `pinv` for all
  and a fixed order, long hand moves with a big `OUT` / `ABOVE` diverged (NaN) or pulled a held hand out.
- **Joint limits:** a joint the step would push past a limit is locked there and the step solved again without it,
  then the joints are clipped (`projecttojointlimits`, as in the planner). With the clipping alone, the ankles and
  the shoulder roll saturate and a held ball drifted up to 10 mm off its socket on a level-2 climb.
- **Stance pose** (`start_pose`, cached per stance): the hang pose with its trunk fitted to the stance's sockets,
  then IK to 1 mm, the joints pulled toward the hang pose's (`POSTURE_GAIN`, last and lowest), until q stops moving.
  The same posture task runs during each move, and **every move ends exactly in its stance pose**:
  - within `JOINT_TOL`, it jumps there;
  - further, it blends there over `BLEND_TIME`, with the balls checked.
  So each move starts from its stance's pose, whatever came before.
- **Output:** `optimise(wall, plan) -> Trajectory`: the start pose, then every move's samples every `DT`, each move
  from where the previous one ended. `positions` = `planned_positions` = the IK's joint angles (kinematic);
  `trunk_poses` for replaying it.
- **`feasible_ik_traj(wall, a, b)`**: `track_move` from a's stance pose. That is the move `optimise` makes, so the
  verdict is cached per (a, b). `scripts/plan_sim_wall.py --feasibility ik_traj` runs the planner's
  `feasible_ik_contact` (reach, IK, collision of stance b) first, then this.

| Parameter | Value | |
|---|---|---|
| `ABOVE` | 0.025 m | up out of socket a before moving out; above socket b before coming down |
| `OUT` | 0.04 m | in front of the sockets while going across (66 mm from the wall) |
| `MOVE_TIME`, `DT` | 2.0 s, 0.02 s | per move; between samples (the robot's 50 Hz loop) |
| `WAIT_TIME` | 0.5 s | each move ends holding still in b's stance pose, where the next move starts |
| `POSTURE_GAIN`, `JOINT_TOL`, `BLEND_TIME` | 0.1, 0.01 rad, 0.3 s | the stance pose (above); the blend lasts long enough to stay under `SPEED_MARGIN` |
| `HELD_WEIGHT`, `REGULARISATION` | 1e3, `DAMPING`² | the QP: a held ball's weight (the moving one's is 1); on ‖v‖² |
| `JOINT_MARGIN`, `SPEED_MARGIN`, `SLACK_WEIGHT` | 0.1 rad, 0.8, 1e2 | the QP: how far inside its limits it tries to stay, and at what cost |
| `MANIPULABILITY_GAIN` | 1e-5 | the QP: on the gradient of the moving limb's log manipulability |
| `GAIN`, `ITERATIONS` | 0.5, 3 | fraction of each error (and of the trunk's tilt) closed per IK step; steps per sample |
| `DAMPING` | 0.03 | the moving ball's and the trunk's damped inverse |
| `MAX_ERROR` | 0.01 m | a held ball further from its socket: `ValueError` naming the move |

Measured on the simulator's collision shapes (ball radius 12.5 mm, holds 43 mm out of the wall), on the 440 hold → hold
paths of the 1113 moves A* asks about on L1 s1, L2 s5226, s7 and s66102, and L3 s1 and s2:
- **The corridor:** clears every hold by at least 6.3 mm, both as planned and as the IK tracks it. 1110 of the 1113
  moves are doable, and it never rises above `ABOVE` over the higher socket.
- **`ABOVE`:**
  - at 25 mm, moving out clears the ball's own hold by 7.9 mm (at 20 mm, only 2.9);
  - it also clears a hold 60 mm above and 15 mm beside (the routes allow one there) by 8.5 mm (at 30 mm, only 3.8).
- **`OUT`:** at 40 mm, the ball goes across 10.5 mm in front of every hold (at 35 mm, only 5.5).
- **Any single curve over** (up, one Bézier over, down) cut its corner into the next hold, by up to 14 mm. That held
  for every `ABOVE` from 30 to 45 mm, `OUT` 40 or 60 mm, and junction speed.

**QP vs pinv IK**, on 279 of those moves (every fourth):

| | moves doable | fastest joint vs its speed limit | closest to a joint limit | moving limb's mean log w |
|---|---|---|---|---|
| pinv | 279 | 3.72× (84 moves over) | 0 rad (277 moves within 0.1) | −7.46 |
| QP, `MANIPULABILITY_GAIN` 0 | 276 | 0.78× | 0.100 rad | −7.27 |
| **QP, 1e-5** | **279** | **0.77×** | **0.100 rad** | **−7.17** |
| QP, 1e-4 | 267 | 0.80× | 0.100 rad | −7.06 |
| QP, 1e-3 | 200 | 0.80× | 0.100 rad | −7.28 |

- **Cost:** with no posture task, every QP move ends away from b's stance pose and needs the blend. The L2 s66102 climb
  (35 moves) lasts 80.5 s instead of 74.5 s.
- **Speed:** the QP takes 48 ms a move against 139, and planning that wall with `ik_traj` takes 18 s against 40.

```bash
uv run mjpython scripts/plan_sim_wall.py --level 2 --seed 5226 --heuristic-weight 1.5 --play  # see it
uv run scripts/plan_sim_wall.py --level 2 --seed 5226 --save-trajectory build/trajectory.json
uv run scripts/plan_sim_wall.py --level 2 --seed 5 --save-npz seed5  # npz/<timestamp>_seed5.npz: q N x 26, 50 Hz
```

## The same motion with placo [Ours]
`placo_ik.py` makes the same moves with Rhoban's [placo](https://github.com/rhoban/placo) (MIT, source in
`third_party/placo`, pinned) instead of `ik_step`: ik.py's ball path, sockets, hang posture, trunk task, timing and
output; only the IK differs. placo's `KinematicsSolver` is a QP each step: the held balls are **hard** tasks, the
moving ball (weight `MOVING_WEIGHT` 300), the trunk upright and at its height (1e-2) and the hang posture (1e-3)
**soft**, the joint and speed limits constraints. After the path, instead of blending in joint space to b's stance
pose (which pulled a held ball 14 mm out on N3), it settles on b's sockets with all four balls hard.

placo pins pinocchio 3.8 and the project 4.x, so `scripts/placo_trajectory.py` runs in its own environment (its
inline dependencies, built by uv on first use); the project's environment is untouched. Same input and output as
`plan_sim_wall.py`'s IK motion: a contact plan in, `build/trajectory_<plan>_placo.json` and `.npz` (q N x 26, 50 Hz)
out, plus a per-move comparison with ik.py.
```bash
uv run scripts/plan_sim_wall.py --wall N3 --level 3 --feasibility ik_contact --save build/plan_N3_wall.json
uv run scripts/placo_trajectory.py build/plan_N3_wall.json --wall N3          # solve, compare, save
uv run scripts/placo_trajectory.py build/plan_N3_wall.json --wall N3 --play   # ... and replay it in MuJoCo
```
`--wall` is needed for a bundled wall: `plan_sim_wall.py --wall N3` names its plan `l<level>_s<seed>`, and rebuilt
from that id the wall would be a generated one (a plan whose holds the wall lacks is refused).

On the organisers' N3 (the A* plan with `ik_contact`, 26 moves, the highest hand at 0.715 m of 1.30 m): all 26
moves; the moving ball within 0.1 mm of its path, the held balls on their sockets, the fastest joint at 27 % of its
speed limit; 66.1 s of motion. ik.py at fe95f66 did 6 (its stance poses ended 2–37 mm off the sockets on the others,
two stopped on an undefined `SPEED_MARGIN`, which also stops the planner's default `ik_traj` feasibility).
**placo shaping the motion (`--generate`)**: before each release the body shifts its CoM into Stéphane Caron's
static-equilibrium polygon of the three held balls (placo's `CoMPolygonConstraint`, 15 mm margin, soft), and keeps
it there until the ball lands. **Physics test** (the script's comparison): each move in the organisers' sim with
gravity at kp_fw 1000, from its own stance pose, the joints sent the motion at 50 Hz; `--ki 2` adds an integral of
each joint's error to its target (capped at 10°), standing in for the robot servos' I gain, which the sim lacks.
On the organisers' N3 (26 moves):

| | motions | slot-ins, ki 0 | slot-ins / misses / falls, ki 2 | CoM margin in the air |
|---|---|---|---|---|
| ik.py | 7 | 0 | 4 / 2 / 1 | median −5.8 mm, outside on 4 |
| **placo** | **26** | 6 | **20 / 4 / 2** | median +6.5 mm, outside on 11 |
| placo + CoM | 26 | 6 | 16 / 4 / 6 | +15 mm on every move |

- Without the integral nearly every move misses its socket by 25–30 mm: the joints sag under load at kp_fw 1000.
- The CoM constraint holds its margin but makes it worse: the shift before the release drives joints to their
  speed limit, and those moves fall.
- Played as one climb (the whole .npz, ki 2), both placo motions slot in the first 3 moves in a row, then fail on
  move 3 (the right foot misses by 16 mm and the robot ends on fewer than three balls).

Open: some joints sit exactly on their limits (placo's are hard, with no margin as the QP of ik.py had); no
collisions or torques yet, as ik.py.

### placo making the route too (`--route`) [Ours]
`placo_motion.py` drops ik.py's corridor path. The moving ball's motion comes from placo itself, with only the start and
end fixed. The ball goes out of its cup straight from the wall (`EXIT` 40 mm, 0.4 s), across to above the target cup
(1.2 s), then into it (0.4 s). Each phase is a `CubicSpline3D`, a soft task. placo keeps the limb off the wall and the
holds: the wall and every hold within `NEAR` 25 cm go into placo's MJCF robot as boxes
(`AvoidSelfCollisionsConstraint`, 8 mm margin, soft 1e4). They sit on a 6-dof `environment` body held fixed in the
world by a hard frame task, because placo reads only the robot's body tree. During the transfer a soft pull towards
y = −70 mm keeps the limb from catching behind a hold (without it, a foot stopped 38 mm short). On N3's test moves:
7.4 mm clearance at worst, the ball 39 mm out, landing on its socket.

On the organisers' N1 (`plan_sim_wall.py --wall N1 --level 1 --feasibility ik_contact`, 51 moves to the top at
1.315 m; the plan repeats a 4-move cycle):
```bash
uv run scripts/placo_trajectory.py build/plan_N1_wall.json --wall N1 --ki 2 --route --out build/placo_n1/route.json
```

| ki 2, per move | motions | slot-ins / misses / falls | CoM margin in the air |
|---|---|---|---|
| ik.py | 0 | – | – (its start pose ends 5–38 mm off the sockets on every move) |
| **placo** (ik.py's path) | 51 | **33 / 7 / 11** | median −5.2 mm, outside on 26 |
| placo + CoM | 51 | 13 / 19 / 19 | +15 mm on every move |
| placo route | 51 | 26 / 7 / 18 | median −5.7 mm, outside on 26 |

- The hands slot in nearly every time with every placo motion. The falls are the feet: in each cycle the right foot
  (CoM +28 mm, but it falls) and the left foot after it (CoM −5 mm). The left foot's next move (CoM −16 to −18 mm,
  outside the polygon) falls on the route and slots in on ik.py's path.
- The route gets the moving joint to its speed limit (1.00x) on every move, against 0.2x on ik.py's 2 s path. The
  2 s are the same, but the exit and entry are 0.4 s each. That speed is the likeliest cause of its extra falls.
- Played as one climb (`route.npz`, ki 2): move 0 lands (right foot on its hold at 2.2 s; the hands lift out of
  their cups for 1 s while the foot unloads, then hook again), then move 1 (left foot) falls, as in the per-move test.

## The CasADi optimiser
Quasi-static direct transcription, one NLP per transition, CasADi + IPOPT, on Microban's URDF: in
[`rcr.optimiser`](../optimiser/), described in [its README](../optimiser/README.md).
```bash
uv sync --group optimiser --group sim
uv run pytest tests/test_optimiser.py
uv run mjpython tests/test_optimiser.py PLAN    # solve a contact plan, play it in the sim
```

## Open decisions
- Collisions with the holds, the wall and itself (none checked yet; the path's `OUT`/`ABOVE` are the only clearance).
- Torques: whether the motion can be held (0.605 N·m limit), and the sag pre-load of `positions`.
- What `feasibility` checks.
- Time per move (time is the tie-breaker in scoring).
- Whether `Trajectory` needs velocities, per-joint kp or the stance index (marked CHANGE in INTERFACES.md).
