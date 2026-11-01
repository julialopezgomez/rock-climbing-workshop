# Wall one (N1): the best policy and the files for the robot

Everything for wall one in one place. The policy is a residual tracking policy (`rcr.controller`, `scripts/train_controller.py`) on Julia's Dynamic trajectory for N1.

## Files

| File | What it is |
|---|---|
| `wall1_policy_P5_19100k.onnx` + `.json` | **The policy**: run P5 (trained on level 1, level 2 and Dynamic), checkpoint at 19.1M steps. Residual action, alpha 0.1 rad, servos P only at kp_fw 3000 (in the `.json`). |
| `trajectory_N1-Dyn.npz` | **The reference** it corrects: Julia's Dynamic trajectory for N1 (153 s, 51 moves). |
| `ROBOT_wall1_first4moves_x13_to_top.npz` | **For the robot (best)**: the policy's first four moves (11.92 s) repeated 13 times, 1 s blend at each seam; reaches the top holds in the sim. |
| `ROBOT_wall1_full_route_31moves.npz` | For the robot: the whole Dynamic route with the policy baked in (open loop); 31 of 51 moves in a row in the sim. |

## Results in the sim (organisers' simulator, wall N1)

| | Single moves (100, each from its reference start) | Full route from the start, open loop |
|---|---|---|
| Playback of the Dynamic trajectory (no policy) | 79 slot-ins, 17 falls, 4 timeouts | 3 of 51 moves |
| **The policy** | **98 slot-ins, 0 falls, 2 timeouts** | 31 of 51 moves |
| The repeat file (first four moves × 13) | | **the top**, at kp 2000 to 8000 |

## Run it in the sim

From the repository root, on this branch (`controller/rl-residual`):

```bash
git submodule update --init
uv sync --group rl                     # then always uv run --no-sync

# 1. The reference as a data folder (once; about 3 min): build/rl/wall1_dyn
uv run --no-sync scripts/train_controller.py import data/wall1_best/trajectory_N1-Dyn.npz --wall N1 --data build/rl/wall1_dyn

# 2. The 98 %: 100 single moves with the policy (about 2 min)
uv run --no-sync scripts/train_controller.py evaluate --data build/rl/wall1_dyn \
    --policy data/wall1_best/wall1_policy_P5_19100k.onnx --episodes 100

# 3. The same without the policy, for comparison: 79 %
uv run --no-sync scripts/train_controller.py baseline --data build/rl/wall1_dyn --kp 3000 --episodes 100

# 4. Watch it in MuJoCo's viewer, the moves chained (macOS: mjpython; Linux: python)
uv run --no-sync mjpython scripts/train_controller.py evaluate --data build/rl/wall1_dyn \
    --policy data/wall1_best/wall1_policy_P5_19100k.onnx --episodes 51 --watch

# 5. The full route from the start, open loop vs closed loop (prints moves in a row)
uv run --no-sync python scripts/closed_vs_open.py --data build/rl/wall1_dyn \
    --policy data/wall1_best/wall1_policy_P5_19100k.onnx --npz data/wall1_best/ROBOT_wall1_full_route_31moves.npz
```

Each `evaluate` / `baseline` line printed is one move: `n/m success, f failed, t timeout`. Adding up the success counts gives the slot-ins out of 100. The policy's `.json` sets the gain (kp_fw 3000) and alpha: `evaluate` uses them by itself. Steps 2 and 3 are deterministic (seed 0), so they print the same numbers as here.

## Run it on the robot

Replay `ROBOT_wall1_first4moves_x13_to_top.npz` as any trajectory (`q`: trunk 7 + 19 joints, 0.02 s per frame, 8348 frames, 167 s). The sim numbers are for P-only servos at kp 3000; on the robot, Kp 1000 / Ki 600 / Kd 1000 at 1.25x speed has worked. Keep the rope on.
