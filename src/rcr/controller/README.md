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

### On the robot: the policy on the laptop, over Wi-Fi [Ours]
`scripts/robot_policy.py` runs `Controller` with the ONNX policy on the laptop (default
`res_a0.2_kp1000_ls-1_s0`): each robot frame (q, dq, IMU, gyro) comes in over the dashboard link, the 19
targets go back to the robot's `replay` move, the servos P only at the policy's kp_fw. This bends the
"nothing time-critical over the link" rule for an experiment: a late frame is a stale observation the policy
never saw in training. So the climb **stops and holds the measured pose** (move on, torque on, until Ctrl+C)
when `step()` raises or gives a non-finite target, takes more than one tick, the frame's q is non-finite, or
no new frame came for `--max-stale` s (0.1 s). The reason is printed and saved in `runs/policy/*.npz`.
Laptop `step()` ≈ 0.1 ms (p99 0.2 ms). Dry run on full N1 (sim, chained moves, no reset): policy falls at
22.9 s, playback at 6.8 s.

### On the robot: an mjlab_microban policy, on the Pi [Ours]
`mjlab_policy.py` runs a policy trained in `../mjlab_microban` (`Mjlab-Climbing-Microban[-Adaptive-Phase]`, MLP,
single board) on the robot itself, at 50 Hz, with numpy and onnxruntime only. Nothing goes over the link during the
climb. Its Microban move is [`../robot/climb_policy_move.py`](../robot/climb_policy_move.py).
- **Input:** a bundle made by mjlab_microban's `train_climbing.sh --deploy` (`policy.onnx` and `climb_bundle.npz`:
  the planned climb, the actor's observation layout, the action scale, the adaptive phase parameters, the robot's
  kinematic tree), in `~/climb_policy` or `$MICROBAN_CLIMB_BUNDLE`. Every tick, Microban's observation (q, dq,
  gyro, gravity, quat).
- **Output:** 19 joint targets = the reference of the frame the step ends on + action scale × action.
- **Adaptive phase:** the reference advances at `v_ref + σ·clip(dv)`. `v_ref` needs the trunk pose. It is
  estimated from the IMU (yaw aligned at the start on the start holds) and from the forward kinematics of the balls
  the plan has on holds. In mjlab's env the error is about 2 mm (max about 1 cm). A ball that slips off its hold
  is not seen.
- **Checked against mjlab's env:** the export runs the env and this file side by side, tick by tick (observation
  and action within 2e-5, targets within 1e-6 rad, over a full 102 s climb). Change it together with the export
  (`BUNDLE_FORMAT`).
- **Closed loop:** `scripts/sim_check_climb_policy.py <bundle>` runs the move in the organisers' sim. With
  `model_2000.pt` of run `tfz2d1g7` on L1 seed 0, it gained up to 38 cm and fell at frame 1932 (39 s of the plan);
  with ideal sensors, at frame 3532 (71 s). The sensor delays are the main gap.
- **Budget:** a tick takes 0.5 ms on the laptop (mostly the forward kinematics); on the Pi Zero 2W this is not
  measured yet.
- `tests/test_mjlab_policy.py` pins the phase, the history, the targets and the estimate on a synthetic bundle.

## Candidates later (not decided)
- Playback plus feedforward sag compensation (≈ τ / kp, see optimiser README);
- a PD correction on top of the servo P loop;
- an RL policy run with onnxruntime (as Microban's walk does): done for mjlab policies (above).

## Open decisions
- Method (D4, decide by Tue 20 Oct).
- What to do when tracking fails (stop? hold pose?).
- Whether it uses `state.base_pose` from the cameras.
- Whether to set per-joint kp (Microban sets kp once in `main.py`).
