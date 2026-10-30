# Interfaces

This file is the single source of truth for every type that crosses a module boundary and every file format.
The code is in [src/rcr/types.py](src/rcr/types.py) (types) and [src/rcr/io.py](src/rcr/io.py) (files).
The full examples are in [data/examples/](data/examples/).

**Change rule:** if you change a type or a format, change it together with this file, the examples and the
tests, all in one PR, and get a teammate's review (see [CONTRIBUTING.md](CONTRIBUTING.md)). Fields
marked **CHANGE** below are the ones we expect to change.

## Conventions

- **Units:** SI everywhere: m, s, rad, rad/s. No mm and no degrees in code or files.
- **Wall frame** [Ours]:
  - origin at the bottom-left corner of the wall face;
  - **x** to the right and **z** up, as seen from in front of the wall; **y** into the wall, so the robot and
    cameras are at y < 0. Right-handed, so it is MuJoCo's world frame as is.
  - Wall: 0.800 m wide × 1.400 m high × 0.010 m thick [Rules §3].
- **Grid** [Rules §3–4]:
  - 52 × 93 M3 holes at a 0.015 m pitch, with a 0.0175 m border left and right and 0.010 m top and bottom
    [Rules v1.1 Figure 2; the §3 text still says 0.010 m on all sides]. Hole (i, j) is at
    x = 0.0175 + 0.015·i, z = 0.010 + 0.015·j (i = 0..51, j = 0..92).
  - A hold is bolted through holes (i − 1, j) and (i + 1, j), two pitches (0.030 m) apart [Rules §4, Figure 3].
    Its reference point (`Hold.position`) is the bolt midpoint, which is hole (i, j) itself:
    x = 0.0175 + 0.015·i, z = 0.010 + 0.015·j, y = 0, with i = 1..50.
  - A hold fits a Ø0.060 × 0.100 m cylinder whose axis is normal to the wall, centred between the bolts.
  - The provided hold [Rules Figure 3; the organisers' simulator's mesh, `climbing_sim/assets/hold/hold_visual.stl`]: a bucket below the bolt line with a spherical
    socket (radius 0.0127 m) that takes the robot's ball end. The socket centre is 0.014 m below and 0.026 m out from
    `Hold.position` (wall frame offset (0, −0.026, −0.014)); the hold sticks out 0.043 m from the wall.
    `Hold.position` stays the bolt midpoint, so walls do not depend on the hold shape.
- **ArUco** [Rules §5.2]:
  - DICT_4X4_50 markers, 0.100 m code + 0.018 m white border (0.136 m total), with the outer border corner on the wall corner.
  - Marker centres in the wall frame (y = 0):

    | ID | where | centre (x, z) |
    |---|---|---|
    | 1 | top-left | (0.068, 1.332) |
    | 2 | top-right | (0.732, 1.332) |
    | 3 | bottom-left | (0.068, 0.068) |
    | 4 | bottom-right | (0.732, 0.068) |
    | 5 | on the robot (optional, any size) | — |

  - The plates cover the grid holes underneath them.
- **Quaternions:** (w, x, y, z), as in Microban.
- **Limbs** [Ours]: `right_hand`, `left_hand`, `right_foot`, `left_foot` (`rcr.types.LIMBS`).
- **Joint names:** Microban's motor names (`rcr.types.MICROBAN_JOINTS`, 19 of them) [Microban].
  **CHANGE:** the climbing limbs are not published yet.
- **Control rate:** 50 Hz, i.e. 20 ms per tick (`CONTROL_HZ`) [Microban].

## Types

| Type | Field | Type | Meaning |
|---|---|---|---|
| `Hold` | `id` | `str` | unique within the wall |
| | `position` | `(x, y, z)` m | bolt midpoint on the wall face, y = 0 |
| | `kind` | `str` | **CHANGE** hold shape / how to grip; `"unknown"` until the provided holds are published |
| `Stance` | `contacts` | `dict[Limb, str \| None]` | hold id per limb, `None` = free; all four limbs are always present. **CHANGE** `"ground"` is reserved for a ground start |
| `WallModel` | `id` | `str` | |
| | `level` | `int` | 1 ladder, 2 ladder with gaps, 3 lateral moves [Rules §3] |
| | `holds` | `list[Hold]` | |
| | `start` | `Stance` | the robot's contacts at the start of the run [Ours] |
| `ContactPlan` | `wall_id` | `str` | `WallModel.id` it was planned on |
| | `stances` | `list[Stance]` | `stances[0] == wall.start`; consecutive stances usually differ by one limb (not enforced) |
| `Trajectory` | `joint_names` | `list[str]` | Microban motor names; joints not listed keep Microban's neutral pose |
| | `times` | `list[float]` s | from the start of the climb, strictly increasing |
| | `positions` | `list[list[float]]` rad | one row per time, one column per joint: what the robot executes. **CHANGE** likely additions: velocities, gains, stance index |
| | `planned_positions` | `list[list[float]] \| None` rad | optional: the optimiser's planned angles, before `positions` were pre-loaded against servo sag. Only for replay; the robot ignores it [Ours] |
| | `trunk_poses` | `list[list[float]] \| None` | optional: the trunk per time, wall frame: `x, y, z` m, `qw, qx, qy, qz`. Only for replay; the robot ignores it [Ours] |
| `Feasibility` | `feasible` | `bool` | |
| | `cost` | `float` | lower is better; meaning owned by the optimiser. **CHANGE** |
| | `reason` | `str` | human-readable, for debugging |
| `FeasibilityFn` | | `(WallModel, Stance, Stance) -> Feasibility` | the optimiser owns it and passes it to the planner |
| `Pose` | `position` | `(x, y, z)` m | robot base, wall frame |
| | `quat` | `(w, x, y, z)` | |
| `RobotState` | `time` | `float` s | since the start of the climb |
| | `joint_positions` | `dict[str, float]` rad | |
| | `joint_velocities` | `dict[str, float]` rad/s | |
| | `imu_quat` | `(w, x, y, z)` | body orientation |
| | `gyro` | `(x, y, z)` rad/s | body frame |
| | `base_pose` | `Pose \| None` | from the cameras over the link, if used. **CHANGE** |
| `JointTargets` | `positions` | `dict[str, float]` rad | position targets; the servos run their own P loop; absent joints keep their last target [Microban dev_v2, docs/moves.md]. **CHANGE** per-joint kp |

`RobotState` and `JointTargets` are the controller's per-tick input and output, so they stay in memory and have no file
format. They are a subset of Microban's `observer.RobotState` and `MotorCommand`. `rcr.robot.to_state` / `to_targets` convert between
ours and Microban's, for the robot and the organisers' sim alike ([src/rcr/robot/README.md](src/rcr/robot/README.md)).

## Module signatures

| Module | Runs on | Signature |
|---|---|---|
| perception | laptop | `wall_model(images) -> WallModel`, `robot_pose(images) -> Pose` |
| planner | laptop | `plan(wall, feasible: FeasibilityFn) -> ContactPlan` |
| optimiser | laptop | `optimise(wall, plan) -> Trajectory`, `feasibility(wall, a, b) -> Feasibility` |
| link | both | `send_trajectory(trajectory or file, robot)`; `segment(trajectory, start)`; `LaptopLink(robot)`: `send_trajectory(final=True)`, `append_segment`, `finish`, `start`, `stop`, `send_pose`, `robot_state`, `ping`, `close`; `RobotLink()`: `trajectory`, `trajectory_complete`, `run_requested`, `pose`, `send_state`, `close`. Wire format: [link/protocol.py](src/rcr/link/protocol.py) |
| controller | robot | `Controller(trajectory).step(state: RobotState) -> JointTargets` |
| sim | laptop | the organisers' `climbing_sim.ClimbingSim` on their wall (`Wall.load(name)`), used as it is; their wall as ours: `walls.from_organisers(wall, level)`; `robot.to_targets(targets, JOINTS, last)` -> their `step()`; `robot.to_state(obs, JOINTS)` -> `RobotState` |

## File formats

All three formats are JSON with a `"schema"` field, and a load fails on any other schema. To change a format,
bump the version (`wall/1` → `wall/2`) and update the examples. Files may be hand-written; any JSON
layout is fine.

### Wall model: `wall/1`

```json
{
  "schema": "wall/1",
  "id": "N1",
  "level": 1,
  "holds": [
    {"id": "c26_r2", "position": [0.4225, 0.0, 0.04], "kind": "unknown"},
    {"id": "c21_r2", "position": [0.3475, 0.0, 0.04], "kind": "unknown"}
  ],
  "start": {"contacts": {"right_hand": "c29_r22", "left_hand": "c18_r22", "right_foot": "c26_r2", "left_foot": "c21_r2"}}
}
```
(Excerpt: the example walls are the organisers' own, [N1](data/examples/N1.json), N2 and N3, converted from their
simulator by `scripts/wall_generator.py`: hold id `c<col>_r<row>` (their hold's left bolt hole), position their bolt
midpoint, start their start holds.)

### Contact plan: `contact_plan/1`

```json
{
  "schema": "contact_plan/1",
  "wall_id": "N1",
  "stances": [
    {"contacts": {"right_hand": "c29_r22", "left_hand": "c18_r22", "right_foot": "c26_r2", "left_foot": "c21_r2"}},
    {"contacts": {"right_hand": "c29_r27", "left_hand": "c18_r22", "right_foot": "c26_r2", "left_foot": "c21_r2"}}
  ]
}
```
(An illustration on N1. There is no example plan file yet: our planner's plan on the organisers' walls comes next.)

### Trajectory: `trajectory/2`

```json
{
  "schema": "trajectory/2",
  "joint_names": ["left_shoulder_pitch", "left_shoulder_roll", "left_elbow"],
  "times": [0.0, 0.25],
  "positions": [[0.17, 0.17, -0.35], [0.18, 0.17, -0.36]],
  "planned_positions": [[0.12, 0.15, -0.30], [0.13, 0.15, -0.31]],
  "trunk_poses": [[0.40, -0.13, 0.31, 0.71, 0.0, 0.0, 0.71], [0.40, -0.13, 0.31, 0.71, 0.0, 0.0, 0.71]]
}
```
`trajectory/2` adds the two optional replay fields to `trajectory/1`; `null` when absent. The
[example](data/examples/trajectory.json) is kept only as a sample file for the link and the Pi: it was solved
for our former example wall and the old robot model, so it does not climb any current wall. A new one comes from
`tests/test_optimiser_v0.py --hard PLAN --save`, once there is a plan on the organisers' walls.
