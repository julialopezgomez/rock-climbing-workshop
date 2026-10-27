"""Where Microban is: the git submodule third_party/microban, unless told otherwise [Ours, D1].

Microban (github.com/Rhoban/microban, GPL-3.0) is never copied into our code: it sits in the
submodule, pinned to the commit we test against, and is read from there as is. A fresh clone
gets it with

    git clone --recursive <our repo>        # or, in an existing clone:
    git submodule update --init

To use another checkout instead (the organisers' climbing variant, say), set MICROBAN_URDF to its
src/model/urdf/robot.urdf or MICROBAN_XML to its src/model/mjcf/robot.xml; either is enough, the
other file is found next to it. The optimiser reads the URDF, the sim the MJCF.

Pi-safe: stdlib only.
"""

from __future__ import annotations

import os
from pathlib import Path

SUBMODULE = Path(__file__).resolve().parents[3] / "third_party" / "microban"
CHECKED_COMMIT = "d0a128d"  # the commit the submodule is pinned to, and our code is tested against
URDF = Path("src/model/urdf/robot.urdf")  # in a Microban checkout
MJCF = Path("src/model/mjcf/robot.xml")


def checkout() -> tuple[Path, str]:
    """(Microban checkout, where that came from): from MICROBAN_URDF or MICROBAN_XML if set,
    else the submodule."""
    for name, inside in (("MICROBAN_URDF", URDF), ("MICROBAN_XML", MJCF)):
        if os.environ.get(name):
            path = Path(os.environ[name]).expanduser().resolve()
            return path.parents[len(inside.parts) - 1], f"{name}={path}"
    return SUBMODULE, "the submodule third_party/microban"


def urdf_path() -> Path:
    """Microban's URDF. Raises FileNotFoundError saying what to do."""
    return _find(URDF)


def mjcf_path() -> Path:
    """Microban's MJCF. Raises FileNotFoundError saying what to do."""
    return _find(MJCF)


def _find(inside: Path) -> Path:
    root, source = checkout()
    path = root / inside
    if path.exists():
        return path
    if root == SUBMODULE:
        raise FileNotFoundError(
            f"Microban not found at {path}: the submodule is empty. Run "
            "`git submodule update --init` in the repository (or clone it with --recursive)"
        )
    raise FileNotFoundError(f"Microban not found at {path} (from {source})")
