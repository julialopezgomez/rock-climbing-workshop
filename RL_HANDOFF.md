# Planner and simulation changes for RL training (handoff)

State as of 2026-10-31, branch `optimiser/julia`. This file lists what changed in this repo (the planner and
`scripts/plan_sim_wall.py`) and around the simulator so that the RL climbing task in `../mjlab_microban`
(`Mjlab-Climbing-Microban*`, the multi-board curriculum) can be trained from our plans. Follow it to redo the
work on a fresh checkout or on `main`.

## 1. What the RL side needs from us

`mjlab_microban` calls only one entry point here, `scripts/plan_sim_wall.py` with the repo's `.venv`:

```bash
# ../mjlab_microban/src/mjlab_microban/scripts/make_multi_board_motions.py runs, per board:
.venv/bin/python scripts/plan_sim_wall.py --level L --seed S --save-npz <motion>_q.npz   # cwd = this repo
# then converts it (on the RL side):
python src/mjlab_microban/scripts/workshop_to_npz.py --input-file <motion>_q.npz --output-file <motion>.npz
```

The `.npz` that `--save-npz` writes is the contract **[Ours]**. It has three arrays:

| key | shape / type | content |
|---|---|---|
| `q` | `(N, 26)` float | a row every 0.02 s (50 Hz, `rcr.kinematics.ik.DT`) from t = 0: trunk `x, y, z` [m], `qw, qx, qy, qz` (wall frame), then the 19 joints [rad] in `rcr.types.MICROBAN_JOINTS` order |
| `used_holds` | `(K,)` str | sorted hold ids `r<row>_c<col>` that some limb is on in some stance of the plan (start included; `None` and `"ground"` left out). The RL env gives these holds full 5-box collision cups and the others 1-box blocks. |
| `wall` | 0-d str | the wall as `mjlab_microban`'s `WallSpec` text: `"level:seed"` (e.g. `"2:5"`), or the `--wall` value (`N1`/`N2`/`N3` or a JSON path) |

`workshop_to_npz.py` checks that the balls of frame 0 sit in the start sockets (5 mm tolerance) and fails
otherwise, so `q[0]` must be the plan's start stance. If `used_holds` or `wall` is missing (an older file), it
needs `--contact-plan <plan.json>` (from `--save`) and `--level/--seed` or `--wall`.

## 2. Changes to `scripts/plan_sim_wall.py`

Committed on this branch (diff against `main`):

- `from rcr import io, kinematics`, `Trajectory` imported from `rcr.types`.
- New flags:
  - `--play`: plays the optimiser's trajectory kinematically in MuJoCo (`ClimbingSim(sim_wall, cameras=[])`,
    `qpos[0:7] = trunk pose`, joints by name, `mj_forward`, real-time sleep). On macOS use `uv run mjpython`.
  - `--save-trajectory PATH`: `io.save_trajectory(trajectory, PATH)`.
  - `--save-npz PATH`: `kinematics.convert_trajectory_to_npz(trajectory, PATH)` (writes `q` only).
- When any of those flags is given and the plan has more than one stance, it runs
  `trajectory = kinematics.optimise(wall, contact_plan)` and prints `trajectory: X s of motion, solved in Y s`
  (`make_multi_board_motions.py` greps for that line and for `" moves in "`).

**Not committed yet** (working tree only): `--save-npz` also writes `used_holds` and `wall`, right after
`convert_trajectory_to_npz`:

```python
import numpy as np  # top of the file

# in main(), inside `if args.save_npz:` after kinematics.convert_trajectory_to_npz(...)
used = sorted({h for s in contact_plan.stances for h in s.contacts.values() if h not in (None, "ground")})
wall_text = str(args.wall) if args.wall else f"{args.level}:{seed}"
np.savez(args.save_npz, **dict(np.load(args.save_npz)), used_holds=np.array(used), wall=np.array(wall_text))
print(f"saved {args.save_npz} ({len(used)} holds used)")
```

`seed` is the variable `main()` already has (`args.seed`, or the random one it printed). This change currently
**fails `ruff check` / `ruff format --check`** (lines over 100 characters). Run `ruff format scripts/plan_sim_wall.py`
before committing, or CI goes red. A cleaner version would put `used_holds` and `wall` into
`kinematics.convert_trajectory_to_npz` (and its README) rather than reopening the file.

## 3. Changes to the planner (`src/rcr/planner/astar.py`)

These are committed on this branch (mainly `2c7d3a9` "fixes to limbs not to overreach", `02f30a5` "without
trunk", `3a38697` "added constraints", plus the parameter commits; `git log main..HEAD -- src/rcr/planner/astar.py`). The RL motions in `mjlab_microban/src/mjlab_microban/motion/` were made
with them, so a reproduction needs them as well. Without them the optimiser fails on more moves (limbs overreach).

| constant | `main` | this branch | meaning |
|---|---|---|---|
| `HAND_MAX_APART` | 0.66 | **0.70** | fraction of reach + roots' distance for a pair that includes a hand |
| `MAX_LEVEL` | 0.12 m | **0.10 m** | most one hand is above the other / one foot above the other |
| `HAND_FEET_MAX_APART` | none | **0.36 m** (new) | most the higher hand is above the lower foot (the solved example plan has 0.35 m) |
| `MAX_TRANSITION_DISTANCE` | none | **0.09 m** (new) | most a limb's hold moves between two consecutive stances (3D straight line) |

All are **[Ours]**. The code changes:

1. `reach(targets)` gets one more check after the foot level check:
   `abs(max(right_hand.z, left_hand.z) - min(right_foot.z, left_foot.z)) > HAND_FEET_MAX_APART` → `fail("reach: a hand too far above a foot ...")`.
2. A new function `acceptable_transition(wall, a, b)` fails if any limb's hold in stance `b` is more than
   `MAX_TRANSITION_DISTANCE` from its hold in stance `a`. A limb that is off any hold (`None`) in `a` or `b` is skipped.
3. In `solve()`, after `children` are filtered with `reach(...)`, they are also filtered with
   `acceptable_transition(wall, stance(k), stance(c)) is None`.
4. In `ik()`, the last null-space task (keep the trunk upright and facing the wall, `TRUNK_UPRIGHT`) is
   **commented out**, so the trunk orientation is left free. The comment `[simulator README, Frames]` on
   `TRUNK_UPRIGHT` was dropped as well.

Consequence for RL: on level 2+ walls the planner often stops before the top ("NOT reached"). The script still
optimises and saves the partial climb, and `make_multi_board_motions.py` uses it as is. Seeds whose IK fails
on some move get no motion. Pick other seeds in `mjlab_microban/.../boards/multi_board.json`. The seeds used
now are level 1 seed 0, and level 2 seeds 0, 19, 43, 45, 48, 53, 71, 74, 85.

## 4. Simulation

- **The organisers' simulator was not edited.** `third_party/simulator` has only moved from `6c180ff` to `17d4341`
  (`origin/main`, a README-only commit). Either commit works. Commit the pointer bump or `git submodule update` back.
- The RL env **does not import the simulator**. `mjlab_microban` copies `climbing_sim`'s `wall` and `routes`
  modules and their assets into `src/mjlab_microban/tasks/climbing/climbing_sim`. It rebuilds the same wall with
  `generate_wall(level, seed)` / `Wall.load(name)`, which is why the `wall` key in the `.npz` is enough. If the
  simulator's wall generation changes, that copy has to be refreshed, or the holds stop matching our plans.
- Frame mapping (done on the RL side, `tasks/climbing/wall.py`). Wall frame (ours: x right, y into the wall,
  z up) → env frame (x towards the wall, y to the robot's left, z up):
  `p_env = SIM_TO_ENV_ROT @ p_sim + SIM_TO_ENV_OFFSET`, with `SIM_TO_ENV_ROT = [[0,1,0],[-1,0,0],[0,0,1]]`
  (−90° yaw). The offset puts the sockets (26 mm out of the wall face, `SOCKET_OUT`) on x = 0 and the middle of
  the 0.8 m wall on y = 0. Quaternions are composed with `SIM_TO_ENV_QUAT`. Nothing to do here as long as `q`
  stays in our wall frame.
- `third_party/microban` currently points to `294817b` on `feature/expose-pid-params` (PID / feedforward gains,
  replay moves) and has uncommitted doc edits and an untracked `tests/`. **This is unrelated to RL training.**
  Do not commit this pointer by accident: CLAUDE.md pins `dev_v2`, and we never edit Microban's files.

## 5. Reproduce from scratch

```bash
# this repo
git checkout optimiser/julia            # or cherry-pick the astar.py changes of section 3 onto main
# apply the section 2 --save-npz change if it isn't committed yet, then:
.venv/bin/ruff format scripts/plan_sim_wall.py && .venv/bin/ruff check .
uv sync --group sim
uv run scripts/plan_sim_wall.py --level 1 --seed 0 --save-npz build/l1s0_q.npz
python -c "import numpy as np; d=np.load('build/l1s0_q.npz'); print(d['q'].shape, d['used_holds'], d['wall'])"
#   expect (N, 26), a list of r<row>_c<col> ids, and '1:0'

# RL side
cd ../mjlab_microban
uv run python src/mjlab_microban/scripts/workshop_to_npz.py \
    --input-file ../rock-climbing-workshop/build/l1s0_q.npz --output-file src/mjlab_microban/motion/l1s0.npz
uv run python src/mjlab_microban/scripts/make_multi_board_motions.py --force   # all boards in multi_board.json
src/mjlab_microban/scripts/train_climbing.sh                  # single wall
src/mjlab_microban/scripts/train_climbing.sh --multi-board    # curriculum over the boards
```

If `uv run` fails with "Read-only file system ... .cache/uv" (sandbox), call `.venv/bin/python` directly, as
`make_multi_board_motions.py` does.

## 6. Open items

- Commit the section 2 change (formatted). Per CLAUDE.md the `.npz` is a file format, so add it to
  `INTERFACES.md` with a test (`q` shape, `used_holds`, `wall`) in the same PR.
- The planner constants in section 3 were tuned by hand on this branch. Before merging, record them in
  `src/rcr/planner/README.md`. Also decide whether the disabled trunk-upright IK task stays off.
- Stray files on this branch that should not reach `main`: `MUJOCO_LOG.TXT` (~110k lines) and `*.npz` at the
  repo root (`iktrajv0.npz`, `trajlevel3.npz`, `wall-1-ik.npz`, `level3-2steps-highergains.npz`).
