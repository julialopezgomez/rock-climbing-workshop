# Local robot tools

These launchers use this repository's `.venv`; no activation or `uv run` is needed. Run `uv sync` once to create it.
They locate the repository relative to this folder, so the checkout can be moved. `RCR_PROJECT_ROOT` can override the
checkout and `RCR_PYTHON` can select another compatible Python environment.

From the repository root:

```sh
sh local-tools/run-dashboard.sh --robot 192.168.111.104 --model-name Climber
sh local-tools/run-robot-replay.sh --host 192.168.111.104 --trajectory /absolute/path/to/trajectory.npz --kp 1000 --ki 600 --kd 1000
sh local-tools/run-wall-editor.sh
```

Dashboard: http://127.0.0.1:8780. Replay viewer: http://localhost:8080.
Wall editor: http://127.0.0.1:8765.

The replay launcher controls the real robot and starts its `replay` move on launch. It follows the NPZ `q` array with
the requested servo gains. The robot must have the compatible Microban move deployed first. Run only one replay
viewer at a time because it listens on port 8080. Support the robot before deployment or initial testing, and stop if
a motor reports a hardware error or tracking error becomes excessive.

Dashboard recordings are written to the ignored `local-tools/recordings/` directory. Temporary trajectories are not
versioned; pass an absolute trajectory path. Explicit relative paths are resolved against the current working directory.
