# Wall three: the best policy so far and its files

Wall three is `climbing_sim.routes.generate_wall(3, 41545)` (the wall of `data/results/to_n3/plan_N3.json`,
`l3_s41545`), saved here as `wall3_l3_s41545.json`; not the organisers' bundled N3.

## Files

| File | What it is |
|---|---|
| `wall3_policy_P7_8000k.onnx` + `.json` | **The policy**: run P7 (trained on wall three's IK and TO trajectories together), checkpoint at 8.0M steps, the best of those tested (2M to 12.3M). Residual action, alpha 0.1 rad, servos P only at kp_fw 3000. |
| `wall-3-ik.npz` | **The reference** it corrects: the IK trajectory for wall three (`data/results/to_n3/wall-3-ik.npz`, 92 s, 46 moves). The IK, not the TO: on the TO the policy reaches only 27 single moves of 100. |
| `wall3_l3_s41545.json` | The wall, for the sim. |
| `ROBOT_wall3_IK_full_route_P7-8000k.npz` | For the robot: the IK route with the policy baked in, open loop (4601 frames, 92 s). |

## Results in the sim

| | Single moves (100, each from its reference start) | From the start, open loop |
|---|---|---|
| Playback of the IK trajectory (no policy) | 41 slot-ins, 59 falls | 2 of 46 moves |
| **The policy** | **59 slot-ins, 41 falls** | 3 of 46 moves (left hand and left foot out at about 8 s) |

Unlike wall one, the route breaks within its first moves, so the full climb does not get far; and the wall
does not repeat (no block of 4 or 8 moves lands the limbs on the same holds one row up), so wall one's
repeat-to-the-top does not apply here. The first moves of the trajectory are what limit it.

## Run it in the sim (from the repository root, branch `controller/rl-residual`)

```bash
git submodule update --init
uv sync --group rl                     # then always uv run --no-sync

# 1. The reference on wall three as a data folder (once; about 2 min)
uv run --no-sync scripts/train_controller.py import data/wall3_best/wall-3-ik.npz \
    --wall data/wall3_best/wall3_l3_s41545.json --data build/rl/wall3_ik

# 2. Watch the policy climb in MuJoCo's viewer, live, from the first move (macOS: mjpython)
uv run --no-sync mjpython scripts/train_controller.py evaluate --data build/rl/wall3_ik \
    --policy data/wall3_best/wall3_policy_P7_8000k.onnx --watch

# 3. Watch the robot file replayed open loop, as the robot plays it
uv run --no-sync mjpython scripts/watch_npz.py data/wall3_best/ROBOT_wall3_IK_full_route_P7-8000k.npz \
    --wall data/wall3_best/wall3_l3_s41545.json

# 4. 100 single moves with the policy: 59 (deterministic); without it (baseline --kp 3000): 41
uv run --no-sync scripts/train_controller.py evaluate --data build/rl/wall3_ik \
    --policy data/wall3_best/wall3_policy_P7_8000k.onnx --episodes 100
```

## Run it on the robot

Replay `ROBOT_wall3_IK_full_route_P7-8000k.npz` like wall one's files (it worked there at Kp 1000 / Ki 600 / Kd 1000, 1.25x). Expect a fall early: in the sim it is at move 3. Rope on.
