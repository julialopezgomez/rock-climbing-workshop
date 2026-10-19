"""Generated meshes for shapes MuJoCo's primitives cannot express.

The organisers' cup: a flat trapezoidal backplate against the wall with a
hemispherical bowl standing out in front of its lower part, opening upwards. Every MuJoCo
primitive is solid, so it is written out as an STL. It is a visual part (holds
never collide with the robot), so MuJoCo's convex hull for contact does not
matter.

Hold-local frame: origin at the midpoint of the two bolts on the wall face,
y out of the wall (towards the robot), z up. scene.py places it turned 180° about z,
since the wall frame's y points into the wall; the cup is symmetric in x.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CupShape:
    """Measured off a 10x crop of their video (frame 520), then scaled to fit
    the Ø0.060 m envelope centred between the bolts [Rules §4]."""

    # Theirs is ~38 mm across the top and reaches ~31 mm below the bolt line,
    # just outside the envelope; these push to its edge (29.5 mm).
    plate_top: float = 0.004  # top edge, above the bolt line
    plate_bottom: float = -0.027  # bottom edge, below the bolt line
    plate_top_width: float = 0.038
    plate_bottom_width: float = 0.024
    plate_thickness: float = 0.003
    bowl_radius: float = 0.0175
    bowl_rim: float = -0.0095  # rim height, ~13 mm below the plate top as in theirs
    bowl_depth: float = 0.019  # rim to the bottom of the bowl
    wall: float = 0.0018
    segments: int = 48
    steps: int = 16


BOWL_STANDOFF = 0.7  # bowl centre this many radii in front of the plate


def _write_stl(path: Path, triangles: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(b"\0" * 80)
        f.write(struct.pack("<I", len(triangles)))
        for tri in triangles:
            normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            norm = np.linalg.norm(normal)
            f.write(struct.pack("<3f", *(normal / norm if norm else normal)))
            for vertex in tri:
                f.write(struct.pack("<3f", *vertex))
            f.write(b"\0\0")
    return path


def _two_sided(tris: list) -> list:
    """Emit each triangle with both windings, so it shows from either side."""
    return tris + [[t[0], t[2], t[1]] for t in tris]


def _plate(s: CupShape) -> list:
    """The trapezoidal backplate, a thin prism flush on the wall."""
    ht, hb = s.plate_top_width / 2, s.plate_bottom_width / 2
    front = [(-ht, s.plate_top), (ht, s.plate_top), (hb, s.plate_bottom), (-hb, s.plate_bottom)]
    back = [np.array([x, 0.0, z]) for x, z in front]
    fwd = [np.array([x, s.plate_thickness, z]) for x, z in front]
    tris = [
        [fwd[0], fwd[2], fwd[1]],
        [fwd[0], fwd[3], fwd[2]],  # front face
        [back[0], back[1], back[2]],
        [back[0], back[2], back[3]],  # back face
    ]
    for i in range(4):
        j = (i + 1) % 4
        tris += [[back[i], fwd[i], fwd[j]], [back[i], fwd[j], back[j]]]  # edges
    return tris


def _bowl(s: CupShape) -> list:
    """A full hemispherical bowl, open upwards, standing out in front of the
    plate with its back edge touching it — side-on in their video (frame 880)
    the bowl is a whole hemisphere, not half of one cut by the plate."""
    # Sunk a little into the plate so the two read as one moulded part.
    centre = np.array([0.0, s.plate_thickness + s.bowl_radius * BOWL_STANDOFF, s.bowl_rim])
    phis = np.linspace(0.0, 2 * np.pi, s.segments + 1)
    t = np.linspace(0.0, np.pi / 2, s.steps)

    def surface(radius: float, depth: float) -> list:
        prof = np.stack([radius * np.cos(t), -depth * np.sin(t)], axis=1)  # rim -> bottom
        tris = []
        for (r0, z0), (r1, z1) in zip(prof, prof[1:], strict=False):
            for a, b in zip(phis, phis[1:], strict=False):
                da = np.array([np.cos(a), np.sin(a), 0.0])
                db = np.array([np.cos(b), np.sin(b), 0.0])
                p00 = centre + r0 * da + [0, 0, z0]
                p01 = centre + r0 * db + [0, 0, z0]
                p10 = centre + r1 * da + [0, 0, z1]
                p11 = centre + r1 * db + [0, 0, z1]
                tris += [[p00, p10, p11], [p00, p11, p01]]
        return tris

    outer = surface(s.bowl_radius, s.bowl_depth)
    inner = surface(s.bowl_radius - s.wall, s.bowl_depth - s.wall)
    rim = []
    for a, b in zip(phis, phis[1:], strict=False):
        da = np.array([np.cos(a), np.sin(a), 0.0])
        db = np.array([np.cos(b), np.sin(b), 0.0])
        ro, ri = s.bowl_radius, s.bowl_radius - s.wall
        rim += [
            [centre + ro * da, centre + ro * db, centre + ri * db],
            [centre + ro * da, centre + ri * db, centre + ri * da],
        ]
    return outer + inner + rim


def cup_stls(out_dir: Path, shape: CupShape | None = None) -> tuple[Path, Path]:
    """Plate and bowl as two meshes, so each can carry its own colour."""
    shape = shape or CupShape()
    plate = _write_stl(out_dir / "cup_plate.stl", np.array(_two_sided(_plate(shape))))
    bowl = _write_stl(out_dir / "cup_bowl.stl", np.array(_two_sided(_bowl(shape))))
    return plate, bowl


def envelope_radius(shape: CupShape) -> float:
    """Farthest point of the cup from the bolt midpoint, in the wall plane."""
    corners = [
        (shape.plate_top_width / 2, shape.plate_top),
        (shape.plate_bottom_width / 2, shape.plate_bottom),
        (shape.bowl_radius, shape.bowl_rim),
        (0.0, shape.bowl_rim - shape.bowl_depth),
    ]
    return max(float(np.hypot(x, z)) for x, z in corners)
