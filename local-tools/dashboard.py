"""Launch Microban's dashboard using the existing climbing-sim environment."""

import runpy
import os
import sys
from pathlib import Path

import microban_model

root = Path(os.environ.get("RCR_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(root / "third_party/microban/src"))
sys.argv[1:1] = [
    "--model", f"Classic={microban_model.mjcf()}",
    "--model", f"Climber={microban_model.mjcf(True)}",
    "--title", "Microban",
    "--recordings", str(Path(__file__).resolve().parent / "recordings"),
]
runpy.run_module("humanoid_dashboard.app", run_name="__main__")
