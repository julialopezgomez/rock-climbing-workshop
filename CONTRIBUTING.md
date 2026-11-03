# Contributing

- **One branch per task.** Each task is a GitHub issue, and branch names look like `planner/ladder-v0` (`<module>/<short-name>`).
- **PR into `main`.** Reference the issue (`Closes #N`). Keep PRs small, one task each.
- **Interface changes need a teammate's review.** This covers anything touching `src/rcr/types.py`, `src/rcr/io.py`, a file
  format or a module's public signature. Such a PR must also update `INTERFACES.md`, `data/examples/` and the tests
  in the same PR.
- Other PRs can be merged once CI is green. A quick look from someone else is welcome but not required.
- **Keep `main` green.** CI runs ruff, pytest and `run_pipeline.py --check`; run them locally before pushing
  (commands in [CLAUDE.md](CLAUDE.md)). If `main` breaks, fixing it comes before anything else.
- **Microban is a submodule** (`third_party/microban`, branch `dev_v2`). To take its latest: `git submodule update
  --remote third_party/microban`, run the tests, and commit the new pin in its own PR. The robot model is the
  `microban-model` package: its commit is pinned in `pyproject.toml` (with `MODEL_COMMIT` in `src/rcr/robot/microban.py`);
  a new one may move the balls, which `test_the_balls_are_the_models` catches. Never commit changes inside a submodule.
- When a module gets a v0 or changes, update its `README.md`, including what the v0 is and which decisions are still open.
