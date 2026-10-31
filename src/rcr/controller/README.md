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

## Residual tracking (exploring on `controller/rl-residual`, D4 still open) [Ours]
`Controller(trajectory, policy=None)`: every tick, targets = clip(feedforward + α × action, joint limits), action in
[−1, 1], so α (rad) is the largest correction and action 0 is playback (residual policy learning, Silver et al. 2018).
The feedforward is `positions` with the pre-load rescaled from kp_fw 400 to the policy's kp_fw. The policy sees the
error to `planned_positions` now and 0.1 s ahead, q, dq, gravity and gyro (82 numbers). Without a policy it is
playback; without `planned_positions` (the link's robot part) it plays `positions` back as they are.
[tracking.py](tracking.py) holds the numbers training and the robot must share (joint order, interpolation,
observation, targets); `scripts/train_controller.py` trains the policy with PPO in the organisers' sim (last layer
started at zero, so training starts from playback; DeepMimic-style bounded reward) and exports it to ONNX, with a
`.json` (action "residual", α, kp_fw, joint order, limits) that `Controller` checks.

On N1 (`wall-1-ik.npz`), 100 single moves at kp_fw 1000: playback 47 slot-ins, residual policies 59–81 (the gain is on
hand moves, which playback never completes). The two best are in `data/policies/`. At kp_fw 2000–5000 playback is
already 66–76 and the residual adds 0–6.

## Candidates later (not decided)
- Playback plus feedforward sag compensation (≈ τ / kp, see optimiser README);
- a PD correction on top of the servo P loop;
- an RL policy run with onnxruntime (as Microban's walk does).

## Open decisions
- Method (D4, decide by Tue 20 Oct).
- What to do when tracking fails (stop? hold pose?).
- Whether it uses `state.base_pose` from the cameras.
- Whether to set per-joint kp (Microban sets kp once in `main.py`).
