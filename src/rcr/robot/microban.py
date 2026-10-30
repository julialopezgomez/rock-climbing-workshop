"""Where Microban and its robot model are [Ours, D1].

The robot's model is the package microban_model (github.com/Rhoban/microban_model, its URDF and
MJCF, meshes in Git LFS), as for Microban's branch dev_v2 and the organisers' simulator [Microban].
We use its climbing variant: the climbing forearms and feet, with the ball ends that go into the
holds (robot_climb.urdf / robot_climb.xml). It comes with `uv sync --group optimiser` (or sim),
pinned in pyproject.toml to the commit the simulator uses.

Microban's own code (its program, moves, the Pi side) is the git submodule third_party/microban,
on branch dev_v2; nothing of it is copied into ours (GPL-3.0). A fresh clone gets it with
`git clone --recursive`, or `git submodule update --init`.

To try another model file, set MICROBAN_URDF (what the optimiser reads) or MICROBAN_XML (what the
sim reads) to it; each replaces only its own file.

Pi-safe: stdlib only (microban_model is imported only when a path is asked for).
"""

from __future__ import annotations

import os
from pathlib import Path

SUBMODULE = Path(__file__).resolve().parents[3] / "third_party" / "microban"
BRANCH = "dev_v2"  # the submodule's branch: what the competition robot runs [Microban]
MODEL_COMMIT = "8e42ef7"  # microban_model, as pinned in pyproject.toml
CLIMBER = True  # the climbing variant of the model (False: the walking one)


def urdf_path(climber: bool = CLIMBER) -> Path:
    """The robot's URDF (climbing variant by default); raises FileNotFoundError if missing."""
    return _find("MICROBAN_URDF", "urdf", climber)


def mjcf_path(climber: bool = CLIMBER) -> Path:
    """The robot's MJCF (climbing variant by default); raises FileNotFoundError if missing."""
    return _find("MICROBAN_XML", "mjcf", climber)


def _find(variable: str, kind: str, climber: bool) -> Path:
    if os.environ.get(variable):
        path = Path(os.environ[variable]).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"{variable} leads to {path}, which does not exist")
        return path
    try:
        import microban_model
    except ImportError:
        raise FileNotFoundError(
            "the robot model (microban_model) is not installed: run `uv sync --group optimiser` "
            "(or --group sim); it needs git-lfs (brew install git-lfs && git lfs install)"
        ) from None
    path = Path(getattr(microban_model, kind)(climber))
    if not path.exists():
        raise FileNotFoundError(f"microban_model has no {path}: git-lfs missing when installed?")
    return path
