"""Run the wall editor without importing the MuJoCo simulator."""

import os
import sys
import types
from pathlib import Path

os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

project_root = Path(os.environ.get("RCR_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
package = types.ModuleType("climbing_sim")
package.__path__ = [str(project_root / "third_party/simulator/climbing_sim")]
sys.modules["climbing_sim"] = package

from climbing_sim.tools.wall_editor import main

main(sys.argv[1:])
