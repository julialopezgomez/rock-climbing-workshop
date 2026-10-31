# controller

**Purpose:** at every tick, turn the trajectory and the current robot state into joint targets.

| | |
|---|---|
| Input | `Trajectory` (once, at construction); `RobotState` every tick |
| Output | `JointTargets` every tick, i.e. every 20 ms (50 Hz) |
| Runs on | **the robot** (Raspberry Pi Zero 2W), inside Microban's Scheduler; on the laptop when driven by the organisers' sim |

- `Controller.step(state) -> JointTargets` mirrors Microban's `Move.step(obs, command)`: state in, joint targets out.
- The servos run their own P loop on the targets [Microban].
- Later, a thin wrapper makes it a real Microban Move (see [../robot/README.md](../robot/README.md)). We do not
  rewrite Microban's Move or Scheduler.
- The controller does **not** know whether it runs on the sim or the robot. Whoever owns the loop calls `step()`.

## Pi-safe
- **Allowed imports:** the stdlib, `numpy`, and, if needed, `placo` and `onnxruntime`. These are what Microban's Pi env
  already has [Microban pyproject.toml].
- **Never:** mujoco, opencv, or anything from `perception/`, `planner/`, `optimiser/` or `sim/`.
- `tests/test_interfaces.py` checks the imports.
- **Budget:** the Pi Zero 2W is weak, and one 20 ms tick also covers Microban's bus reads/writes and IMU. `step()` must be a
  small fraction of that.
- No planning or optimisation here. If we replan, the laptop does it and sends a new trajectory over the link.

## v0 could be
Trajectory playback: interpolate `positions` at `state.time` and return them.

## Replay on the real robot (laptop-driven, PID only)
Run it with `uv run robot-replay --host <robot> --kp <P> --ki <I> --kd <D>` (the `[project.scripts]` entry in
`pyproject.toml`). `robot_replay.py` streams each frame's joint targets to the robot-side `replay` move (our Microban
fork, `src/moves/replay.py`). The servos' goal is the target alone: no feedforward torque offset, and the XL330's
velocity / acceleration feedforward gains held at 0. The servo firmware's P, I and D gains (register units, 0..16383)
do all the tracking. [Ours]

Each replay is logged tick by tick (target sent, `q` measured, the robot's `q_target`, the IMU's `imu_quat`, the
trajectory's base quaternion) to `runs/replay/replay-<date>-<time>.npz` (git-ignored). When it ends, the page plots
each joint's measured position against its target and the base rotation error (IMU trunk vs trajectory base,
roll / pitch / yaw in the trunk frame), and the terminal prints per-joint RMS / max errors. The IMU's yaw is arbitrary,
so the trajectory's base is turned about z to match it at the replay's first tick [Ours]. Replot a saved log with
`uv run robot-replay --plot runs/replay/<log>.npz`.

## Candidates later (not decided)
- Playback plus feedforward sag compensation (≈ τ / kp, see optimiser README);
- a PD correction on top of the servo P loop;
- an RL policy run with onnxruntime (as Microban's walk does).

## Open decisions
- Method (D4, decide by Tue 20 Oct).
- What to do when tracking fails (stop? hold pose?).
- Whether it uses `state.base_pose` from the cameras.
- Whether to set per-joint kp (Microban sets kp once in `main.py`).
