"""The provided hold's mesh [Rules v1.1 Figure 3].

The organisers publish the real hold: `figures/hold.stl` in rock-climbing-robots/rules,
meshed from their CAD (`assets/hold.step`, ~20k triangles, in mm), and Figure 3 is drawn
from it. Our copy is data/holds/hold.stl (see data/holds/README.md).

Its frame, per their `figures/gen_hold.py`: origin at the socket centre, +y towards the wall,
back face at y = +26 mm, bolt holes at x = ±15 mm, z = +14 mm. +y towards the wall is our wall
frame's +y too, so the mesh goes on the wall with a translation only (geometry.CupHold).
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

HOLD_STL = Path(__file__).resolve().parents[3] / "data" / "holds" / "hold.stl"
HOLD_STL_SCALE = 0.001  # the STL is in mm


def read_stl(path: Path) -> np.ndarray:
    """Triangles of a binary STL, shape (n, 3, 3), in the file's units."""
    raw = path.read_bytes()
    (n,) = struct.unpack("<I", raw[80:84])
    if len(raw) != 84 + 50 * n:
        raise ValueError(f"{path} is not a binary STL")
    tri = np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    return np.frombuffer(raw, dtype=tri, count=n, offset=84)["v"].astype(float)
