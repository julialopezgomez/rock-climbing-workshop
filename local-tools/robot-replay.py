"""Run the branch's replay using the existing climbing-sim environment."""

import inspect
import os
import sys
from pathlib import Path

import microban_model

sys.path.insert(0, str(Path(__file__).resolve().parent / "replay-runtime"))

# This environment's model calls this argument `climb`; the branch uses `climber`.
if "climber" not in inspect.signature(microban_model.mjcf).parameters:
    original_mjcf = microban_model.mjcf

    def compatible_mjcf(climber=False):
        return original_mjcf(climb=climber)

    microban_model.mjcf = compatible_mjcf

from rcr.controller.robot_replay import main

if __name__ == "__main__":
    main()
