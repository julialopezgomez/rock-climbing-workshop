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
  - M3 holes at a 0.015 m pitch with a 0.010 m border, so hole (i, j) is at x = 0.010 + 0.015·i, z = 0.010 + 0.015·j (i = 0..52, j = 0..92).
  - A hold uses two horizontally adjacent holes, so a hold's reference point is x = 0.0175 + 0.015·i, z = 0.010 + 0.015·j, y = 0.
  - A hold fits a Ø0.060 × 0.100 m cylinder whose axis is normal to the wall.
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
| | `positions` | `list[list[float]]` rad | one row per time, one column per joint. **CHANGE** likely additions: velocities, gains, stance index |
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
| `JointTargets` | `positions` | `dict[str, float]` rad | position targets; the servos run their own P loop; absent joints go to neutral [Microban `MotorCommand`]. **CHANGE** per-joint kp |

`RobotState` and `JointTargets` are the controller's per-tick input and output, so they stay in memory and have no file
format. They are a subset of Microban's `observer.RobotState` and `MotorCommand`. The Move wrapper converts between ours and Microban's
([src/rcr/robot/README.md](src/rcr/robot/README.md)).

## Module signatures

| Module | Runs on | Signature |
|---|---|---|
| perception | laptop | `wall_model(images) -> WallModel`, `robot_pose(images) -> Pose` |
| planner | laptop | `plan(wall, feasible: FeasibilityFn) -> ContactPlan` |
| optimiser | laptop | `optimise(wall, plan) -> Trajectory`, `feasibility(wall, a, b) -> Feasibility` |
| link | both | `LaptopLink`: `send_trajectory`, `start`, `stop`, `send_pose`, `robot_state`; `RobotLink`: `trajectory`, `run_requested`, `pose`, `send_state` |
| controller | robot | `Controller(trajectory).step(state: RobotState) -> JointTargets` |
| sim | laptop | `Sim(wall).reset() -> RobotState`, `.step(targets: JointTargets) -> RobotState` |

## File formats

All three formats are JSON with a `"schema"` field, and a load fails on any other schema. To change a format,
bump the version (`wall/1` → `wall/2`) and update the examples. Files may be hand-written; any JSON
layout is fine.

### Wall model: `wall/1`

```json
{
  "schema": "wall/1",
  "id": "example_l1",
  "level": 1,
  "holds": [
    {"id": "L0", "position": [0.3175, 0.0, 0.16], "kind": "unknown"},
    {"id": "R0", "position": [0.4825, 0.0, 0.16], "kind": "unknown"}
  ],
  "start": {"contacts": {"right_hand": "R2", "left_hand": "L2", "right_foot": "R0", "left_foot": "L0"}}
}
```
(Excerpt: the [full example](data/examples/wall_l1.json) has 8 holds in two columns, with rows at z = 0.16 / 0.31 / 0.46 / 0.61.)

### Contact plan: `contact_plan/1`

```json
{
  "schema": "contact_plan/1",
  "wall_id": "example_l1",
  "stances": [
    {"contacts": {"right_hand": "R2", "left_hand": "L2", "right_foot": "R0", "left_foot": "L0"}},
    {"contacts": {"right_hand": "R3", "left_hand": "L2", "right_foot": "R0", "left_foot": "L0"}},
    {"contacts": {"right_hand": "R3", "left_hand": "L3", "right_foot": "R0", "left_foot": "L0"}}
  ]
}
```

### Trajectory: `trajectory/1`

```json
{
  "schema": "trajectory/1",
  "joint_names": ["left_shoulder_pitch", "left_shoulder_roll", "left_elbow",
                  "right_shoulder_pitch", "right_shoulder_roll", "right_elbow"],
  "times": [0.0, 1.0, 2.0],
  "positions": [[0.17, 0.17, -0.35, 0.17, -0.17, -0.35],
                [0.17, 0.17, -0.35, 1.2, -0.17, -0.6],
                [1.2, 0.17, -0.6, 1.2, -0.17, -0.6]]
}
```
The values in the examples are **placeholders** that are valid against the types. They are not a motion anyone has checked.
