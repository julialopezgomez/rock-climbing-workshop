"""Runtime resources, independent of the original experiment workspace."""

import importlib.util
import os
from pathlib import Path

CACHE = (
    Path(os.environ.get("RCR_PERCEPTION_HOME", Path.home() / ".cache/rcr-perception"))
    .expanduser()
    .resolve()
)
WEIGHTS = Path(__file__).parent / "assets/weights"


def model_path():
    override = os.environ.get("RCR_MICROBAN_MODEL")
    if override:
        return Path(override)
    spec = importlib.util.find_spec("microban_model")
    if spec is None or spec.origin is None:
        raise RuntimeError(
            "Install the pinned microban-model package with the perception setup script"
        )
    return Path(spec.origin).parent / "mjcf/robot_climb.xml"


def simulator_asset(relative):
    override = os.environ.get("RCR_SIMULATOR_ROOT")
    if override:
        return Path(override) / "climbing_sim" / relative
    repo = Path(__file__).resolve().parents[4]
    candidate = repo / "third_party/simulator/climbing_sim" / relative
    if candidate.exists():
        return candidate
    raise RuntimeError("Initialise third_party/simulator or set RCR_SIMULATOR_ROOT")
