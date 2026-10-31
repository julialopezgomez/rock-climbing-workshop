# RL tracking on wall N1: results of 30–31 Oct 2026

The PPO tracking policies trained on the IK reference `wall-1-ik.npz` (51 moves on wall N1) in the
organisers' sim, their training plots, and one common evaluation of all of them. The data folder
they were trained on is rebuilt with `scripts/train_controller.py import` (see that script).

## Comparison: the same 100 single-move episodes

Every policy below was evaluated with its own branch's code on the same 100 episodes (random
reference starts, fixed seed; 44 of the 51 moves drawn), servo gain kp_fw 1000. Per-move lines:
`eval/`.

| Policy | Action | Slot-ins | Falls | Timeouts | Hand moves | Foot moves | Joint error | Ball error |
|---|---|---|---|---|---|---|---|---|
| playback (no policy) | – | 47 | 51 | 2 | 0 / 39 | 47 / 61 | 0.029 rad | 17 mm |
| `data/policies/clone_kp1000_ls-1_s2` (curriculum start) | blend, alpha 0.1 | 58 | 38 | 4 | 26 / 39 | 32 / 61 | 0.041 rad | 15 mm |
| `curriculum_s0/alpha0.2` | blend, alpha 0.2 | 62 | 38 | 0 | 25 / 39 | 37 / 61 | 0.073 rad | 18 mm |
| `curriculum_s0/alpha0.3` (did not pass its gate) | blend, alpha 0.3 | 15 | 48 | 37 | 6 / 40 | 9 / 60 | 0.105 rad | 29 mm |
| **`residual/res_a0.2_kp1000_ls-1_s0`** | residual, 0.2 rad | **82** | **16** | 2 | 25 / 39 | 57 / 61 | 0.031 rad | 12 mm |
| **`residual/res_a0.2_kp1000_ls-0.5_s0`** | residual, 0.2 rad | **81** | **17** | 2 | 23 / 39 | 58 / 61 | 0.040 rad | 13 mm |

- **Blend** (this branch): target = (1 - alpha) * reference + alpha * (action mapped onto the joint
  range); alpha is the policy's share. **Residual** (branch `controller/rl-residual`): target =
  reference + alpha * action; alpha is the largest correction in rad, and action 0 is playback.
  The two read alpha differently: run each policy with its own branch's code (this branch's
  evaluate would run a residual policy as a blend, without an error).
- Playback never lands a hand (the loaded joints sag 5–7° at kp_fw 1000); every trained policy
  lands 23–26 of 39. The residual policies also fix nearly all foot moves.
- The alpha curriculum (blend) passed alpha 0.2 and stopped at 0.3 (three 1M-step chunks, slot-ins
  10–14 % against a gate of 45 %).

## What is here

| Folder | Contents |
|---|---|
| `residual/` | the two best policies (.onnx, .json), copies of `data/policies/` on `controller/rl-residual`; no training plots here |
| `curriculum_s0/` | the alpha curriculum, seed 0, 10 sims, from `data/policies/clone_kp1000_ls-1_s2`: each stage's policy (.zip to resume, .onnx, .json), `*_history.png` / `.npz`, `STATUS.md` and `curriculum.tsv` (every evaluation), the log |
| `normreward_kp125/` | the first run: blend alpha 0.1, kp_fw 125, `--norm-reward`, 2M steps; policy, history plot, log, and its evaluation against playback at kp_fw 125 (0 slot-ins for both) |
| `eval/` | the common evaluation above, one file per policy |

## Watching a policy in the sim

From the repository root, with the policy's own branch checked out (macOS needs `mjpython`):

```bash
# residual policies: git checkout controller/rl-residual
uv run --no-sync mjpython scripts/train_controller.py evaluate --data build/rl/n1 --watch \
  --policy data/policies/res_a0.2_kp1000_ls-1_s0.onnx
# blend policies: git checkout controller/rl-tracking
uv run --no-sync mjpython scripts/train_controller.py evaluate --data build/rl/n1 --watch \
  --policy data/results/rl_n1/curriculum_s0/alpha0.2.onnx
# playback, for comparison
uv run --no-sync mjpython scripts/train_controller.py baseline --data build/rl/n1 --kp 1000 --watch
```
`--watch` chains the moves like a climb: a success carries on to the next move.
