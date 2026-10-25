# rock-climbing-workshop

This is our team project for the [Rock Climbing Robots](https://rock-climbing-robots.github.io/) hackathon
(Bordeaux, 29 Oct – 1 Nov 2026). The robot is Microban-based ([Rhoban/microban](https://github.com/Rhoban/microban)), and the rules are
[v1.1](https://github.com/rock-climbing-robots/rules/releases/tag/v1.1).

The repo is a skeleton for the pipeline: contracts, stubs and a check, with no algorithms yet. We have
**not** chosen the planner, the optimiser or the controller. When we choose or change one, the change
stays inside its module.

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
- The [sim](src/rcr/sim/README.md) plays the robot's role: joint targets in, robot state out. The controller does not
  know which backend it is running on.

## Layout

```
INTERFACES.md            every type and file format, with examples
ROADMAP.md               phases, tasks, decisions with dates
CLAUDE.md                rules for us and for Claude Code
CONTRIBUTING.md          branches, PRs, reviews
data/examples/           tiny L1 wall, 2-move contact plan, tiny trajectory
scripts/run_pipeline.py  --check: examples through every stage's signature
tests/                   format round-trips, Pi-safe imports, planner boundary
src/rcr/
  types.py  io.py        contracts and save/load (Pi-safe)
  perception/            laptop: camera images -> wall model + robot pose
  planner/               laptop: wall model -> contact plan
  optimiser/             laptop: contact plan -> trajectory; owns feasibility
  controller/            ROBOT:  trajectory + robot state -> joint targets (Pi-safe)
  link/                  both:   trajectory, start/stop, pose ->; <- robot state
  sim/                   laptop: backend with the robot's role
  robot/                 how our code meets Microban; our climbing hooks (hooks.py)
```

Each module has a `README.md` that covers purpose, input, output, where it runs, v0 ideas and open decisions.
Each module also has an `__init__.py`, which is its public interface; implementation files go next to it.

## Run

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

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
