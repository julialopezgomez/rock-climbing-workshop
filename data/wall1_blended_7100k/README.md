# Wall one (N1): the blended policy run on the robot (P5 at 7.1M steps)

The policy behind the blended repeat file that was run on the real robot on 31 Oct and worked
(`data/executable-files/v3_P5_dyn_first4x13_blend1s.npz`, byte for byte the same as the file here).
For the later checkpoint (P5 at 19.1M steps, more robust in the sim) see `../wall1_best/`.

## Files

| File | What it is |
|---|---|
| `wall1_policy_P5_7100k.onnx` + `.json` | **The policy**: run P5 (trained on level 1, level 2 and Dynamic), checkpoint at 7.1M steps. Residual action, alpha 0.1 rad, servos P only at kp_fw 3000 (in the `.json`). |
| `ROBOT_wall1_first4moves_x13_to_top_P5-7100k.npz` | **For the robot: the blended file you ran.** The policy's first four moves (11.92 s) repeated 13 times, 1 s blend at each seam; 8348 frames, 167 s. Reaches the top holds in the sim. |
| `ROBOT_wall1_full_route_P5-7100k.npz` | The whole Dynamic route with this policy baked in (open loop); 31 of 51 moves in a row in the sim. |

The reference it corrects is Julia's Dynamic trajectory, `../wall1_best/trajectory_N1-Dyn.npz` (the same file, not copied twice).

## Results in the sim

| | Single moves (100, each from its reference start) | From the start, open loop |
|---|---|---|
| Playback of the Dynamic trajectory (no policy) | 79 slot-ins, 17 falls | 3 of 51 moves |
| **This policy** | **98 slot-ins, 0 falls, 2 timeouts** | 31 of 51 moves |
| The blended repeat file | | **the top**, at kp 2200 to 4000 |

## Run it in the sim (from the repository root, branch `controller/rl-residual`)

```bash
git submodule update --init
uv sync --group rl                     # then always uv run --no-sync

# 1. The reference as a data folder (once; about 3 min)
uv run --no-sync scripts/train_controller.py import data/wall1_best/trajectory_N1-Dyn.npz --wall N1 --data build/rl/wall1_dyn

# 2. Watch the policy climb in MuJoCo's viewer, live, from the first move to the top (macOS: mjpython)
uv run --no-sync mjpython scripts/train_controller.py evaluate --data build/rl/wall1_dyn \
    --policy data/wall1_blended_7100k/wall1_policy_P5_7100k.onnx --watch

# 3. Watch the blended robot file replayed open loop, as the robot plays it (--speed 1.25 as on the robot)
uv run --no-sync mjpython scripts/watch_npz.py data/wall1_blended_7100k/ROBOT_wall1_first4moves_x13_to_top_P5-7100k.npz

# 4. The 98 %: 100 single moves with the policy (about 2 min, deterministic)
uv run --no-sync scripts/train_controller.py evaluate --data build/rl/wall1_dyn \
    --policy data/wall1_blended_7100k/wall1_policy_P5_7100k.onnx --episodes 100

# 5. The policy live from the start as on the robot, with the sim playing the robot
uv run --no-sync mjpython scripts/robot_policy.py --dry-run --watch --data build/rl/wall1_dyn \
    --policy data/wall1_blended_7100k/wall1_policy_P5_7100k.onnx
```

## Run it on the robot

Replay `ROBOT_wall1_first4moves_x13_to_top_P5-7100k.npz` as before: it worked at Kp 1000 / Ki 600 / Kd 1000, 1.25x speed. Rope on.
