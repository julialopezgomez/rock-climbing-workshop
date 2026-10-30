# rock-climbing-workshop

This is our team project for the [Rock Climbing Robots](https://rock-climbing-robots.github.io/) hackathon
(Bordeaux, 29 Oct – 1 Nov 2026). The robot is Microban-based ([Rhoban/microban](https://github.com/Rhoban/microban)), and the rules are
[v1.1](https://github.com/rock-climbing-robots/rules/releases/tag/v1.1).

The repository defines the pipeline contracts and its implementations. Module READMEs
and the pipeline check identify implemented stages and remaining integration work.

## Pipeline

```
  LAPTOP                                                                       ROBOT (Pi, 50 Hz)
  perception ─► wall model ─► planner ─► contact plan ─► optimiser ─► trajectory ══ link ══► controller ─► joint targets
                                 ▲          (full)          │           (full)                (a Microban Move)
                                 └────── feasibility ◄──────┘                     ◄══ robot state ══
```

- What crosses each arrow is defined in [INTERFACES.md](INTERFACES.md), which is the single source of truth, and implemented in
  [src/rcr/types.py](src/rcr/types.py) and [src/rcr/io.py](src/rcr/io.py).
- Planning and optimisation run on the laptop. The 50 Hz loop runs on the robot. Nothing time-critical crosses the
  [link](src/rcr/link/README.md).
- The sim is the organisers' own ([Simulation](#simulation)): it plays the robot's role, joint targets in, robot
  state out. The controller does not know which backend it is running on.

## Layout

```
INTERFACES.md            every type and file format, with examples
ROADMAP.md               phases, tasks, decisions with dates
CLAUDE.md                rules for us and for Claude Code
CONTRIBUTING.md          branches, PRs, reviews
data/examples/           tiny L1 wall, 2-move contact plan, tiny trajectory
scripts/run_pipeline.py  --check: examples through every stage's signature
scripts/make_wall.py     random L1/L2/L3 walls (show, save to data/walls/), or --check a wall file
tests/                   format round-trips, Pi-safe imports, planner boundary
src/rcr/
  types.py  io.py        contracts and save/load (Pi-safe)
  walls.py               seeded random walls per level, and the rule checks a wall must pass
  perception/            laptop: camera images -> wall model + robot pose
  planner/               laptop: wall model -> contact plan
  optimiser/             laptop: contact plan -> trajectory; owns feasibility
  controller/            ROBOT:  trajectory + robot state -> joint targets (Pi-safe)
  link/                  both:   trajectory, start/stop, pose ->; <- robot state
  robot/                 how our code meets Microban: where it is (microban.py), the climbing balls (hooks.py),
                         its observation <-> our types (observation.py)
third_party/microban/    Rhoban/microban, a git submodule (not our code, GPL-3.0)
third_party/simulator/   the organisers' simulator, rock-climbing-robots/simulator, a git submodule
```

Each module has a `README.md` that covers purpose, input, output, where it runs, v0 ideas and open decisions.
Each module also has an `__init__.py`, which is its public interface; implementation files go next to it.

## Run

Requires [uv](https://docs.astral.sh/uv/) and git.

```bash
git clone --recursive https://github.com/julialopezgomez/rock-climbing-workshop.git   # with Microban
cd rock-climbing-workshop
uv sync                                    # base; add --group sim --group optimiser for those
```

Microban ([Rhoban/microban](https://github.com/Rhoban/microban), GPL-3.0) is a git submodule in `third_party/microban`,
on branch **`dev_v2`**, what the competition robot runs. Cloned without
`--recursive`? Run `git submodule update --init`. After a `git pull` that moves the pin, run it again.

The organisers' simulator ([rock-climbing-robots/simulator](https://github.com/rock-climbing-robots/simulator)) is a
second submodule, in `third_party/simulator`, next to Microban as its README asks: see [Simulation](#simulation).

**The robot model** is the package `microban_model`, as for Microban `dev_v2` and the simulator: we use its climbing
variant (climbing forearms and feet with ball ends), from our fork
([MohammadrezaDindarloo/microban_model](https://github.com/MohammadrezaDindarloo/microban_model), branch `cut-foot`):
Rhoban's model at the simulator's commit with only the climbing feet cut, as on our robot
([src/rcr/robot](src/rcr/robot/README.md)). `uv sync --group optimiser` / `--group sim` installs it, and the
simulator loads it too. It needs git-lfs for the
meshes: `brew install git-lfs && git lfs install`, once.

```bash
uv run scripts/run_pipeline.py --check
```

```bash
uv run pytest
```

```bash
uv run ruff check . && uv run ruff format --check .
```

`--check` prints one line per stage, either `stub` or `ok`. A stage counts as done for Phase 1 when it says `ok`.
CI runs all of the above on every push and PR.

Random walls for testing (`rcr.walls`, see its docstring for what each level means):

```bash
uv run scripts/make_wall.py --level 3 --seed 0 --show    # or --save, or --check data/examples/N1.json
```

## Simulation

We use the organisers' simulator (`climbing_sim`, in `third_party/simulator`) as it is: we never edit it, and
whatever does not fit is converted on our side. `uv sync --group sim` installs it from the submodule, with
Rhoban's `microban_sim` (the robot, its servos, its sensors) pinned to the commit its lock file uses.

| Theirs | Ours | Converted by |
|---|---|---|
| `Wall` (hold = left bolt hole `(col, row)`) | `WallModel` (hold id `c<col>_r<row>`, bolt midpoint), for the planner and the optimiser | `rcr.walls.from_organisers`; `scripts/wall_generator.py` saves `data/examples/N*.json` |
| 19 angles in `JOINTS` order, the observation dict | `JointTargets`, `RobotState` | `rcr.robot.to_targets` / `to_state` |

The wall frame is the same, and our optimiser's kinematics match theirs to 1e-9 m (a test checks it).

```bash
uv run mjpython tests/test_sim.py N2                      # the robot on their wall N2 (macOS: mjpython)
uv run mjpython tests/test_optimiser.py PLAN       # solve a contact plan, play it in their sim
uv run scripts/wall_generator.py N1 N2 N3                 # their walls -> data/examples/N*.json
```

## Perception

The [perception runtime](docs/perception/README.md) includes the selected RGB/CAD
hold estimator, trained Microban keypoint models, configured robot tracking and
refinement, replay data, and two-RealSense RGB capture/calibration entrypoints.
The [report](docs/perception/perception_report.pdf) describes the method,
measured results and limitations. Start with:

```bash
python3.12 scripts/perception/setup.py --robot --cuda
.venv-perception/bin/python scripts/perception/run.py reproduce --out outputs/perception/holds
```

Use the [runbook](docs/perception/RUNBOOK.md) for robot replay and physical
cameras. The runtime uses an isolated environment and explicit measurement
packets. The legacy images-only functions remain stubs; the
[team adapters](docs/perception/TEAM_INTEGRATION.md) convert runtime outputs
without changing existing contracts or inventing start contacts.
