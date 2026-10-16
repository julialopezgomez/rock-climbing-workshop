# CLAUDE.md

This is the team project for the Rock Climbing Robots hackathon (Bordeaux, 29 Oct – 1 Nov 2026). Read
[README.md](README.md) for the pipeline and [INTERFACES.md](INTERFACES.md) for the contracts first.

## Layout
- `src/rcr/types.py` holds the contracts and `src/rcr/io.py` holds the file formats. Both are Pi-safe.
- `src/rcr/<module>/`: `perception`, `planner`, `optimiser`, `controller`, `link`, `sim`, plus `robot` (README only).
  - Each module's `__init__.py` is its public interface. Put implementations in files next to it and have `__init__`
    delegate to them, so that swapping a method never changes callers.
  - Each module's `README.md` covers purpose, input, output, where it runs and open decisions. Update it when you change the module.
- `data/examples/` holds the canonical example files. `scripts/run_pipeline.py --check` runs them through every stage.
- `ROADMAP.md` lists phases, tasks and decisions. Tasks are GitHub issues labelled by module.

## Commands
```bash
uv sync                                   # base + dev deps; add --group sim / --group perception
uv run scripts/run_pipeline.py --check    # examples through every stage (stub/ok per stage)
uv run pytest
uv run ruff check . && uv run ruff format --check .
```
CI runs all of these. Keep `main` green.

## Rules
- **Interfaces:** change `types.py` or a file format only together with `INTERFACES.md`, the examples and the
  tests, all in **one PR**, and get a teammate's review. Bump the `"schema"` version when a format changes.
- **Planner never imports optimiser or controller.** Feasibility is passed in as a `FeasibilityFn`. A test
  enforces this.
- **`controller/` stays Pi-safe.** Allowed imports are the stdlib and numpy, plus placo and onnxruntime if needed; these are
  what Microban's Pi env has. Never mujoco, opencv or anything laptop-only. The same goes for `link/` (robot side), `types.py` and
  `io.py`. A test enforces this.
- **Nothing time-critical over the link.** The 50 Hz loop lives on the robot. Planning and optimisation never run
  on the Pi; if we replan, the laptop does it and sends the result.
- **The controller does not know its backend.** It sees `RobotState` in and `JointTargets` out, nothing else.
- **Do not rewrite Microban's Move or Scheduler.** Wrap our controller as a Move (see `src/rcr/robot/README.md`).
- **SI units and the wall frame** everywhere: m, s, rad. Origin at the bottom-left of the wall face, x right, z up,
  y out of the wall. Quaternions are (w, x, y, z).
- **Keep it simple.** Use plain dataclasses and functions; no frameworks, no class hierarchies, few files. If something
  isn't needed yet, write a line in the module README instead of creating a file.
- **Tag design choices** in docs and comments: **[Rules]** (from the rulebook, cite the section), **[Microban]**
  (from Rhoban/microban), or **[Ours]** (our choice, so it can be changed).
- No method decision (planner / optimiser / controller / link transport / Microban location) is made in code
  before its ROADMAP decision task is closed.
