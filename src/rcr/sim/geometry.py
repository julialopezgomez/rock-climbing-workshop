"""Wall, hold and end-effector geometry, in the wall frame.

Wall frame [Ours, INTERFACES.md]: origin at the bottom-left corner of the wall
face, x right, z up, y into the wall (the robot is at y < 0). The MuJoCo world
*is* this frame, so nothing converts coordinates.

The grid follows Rules v1.1 Figure 2 and the holds are the organisers' own
mesh (Figure 3). The limbs follow the organisers' simulator video
(images/microban_climb_poster.jpg on the workshop site): ball-ended limbs that
sit in the holds' sockets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rcr.sim.meshes import HOLD_STL, HOLD_STL_SCALE, read_stl
from rcr.types import GRID_MARGIN_X, GRID_MARGIN_Z, GRID_PITCH, WALL_HEIGHT, WALL_WIDTH

WALL_THICKNESS = 0.010  # [m] [Rules §3]
HOLE_DIAMETER = 0.0034  # [m] M3 through-hole [Rules §3]

N_COLS = round((WALL_WIDTH - 2 * GRID_MARGIN_X) / GRID_PITCH) + 1  # 52 [Rules Figure 2]
N_ROWS = round((WALL_HEIGHT - 2 * GRID_MARGIN_Z) / GRID_PITCH) + 1  # 93

ENVELOPE_RADIUS = 0.030  # [m] hold must fit Ø0.060 x 0.100 m [Rules §4]
ENVELOPE_LENGTH = 0.100


def hole_xz(i: int, j: int) -> tuple[float, float]:
    """Centre of grid hole (i, j) on the wall face [Rules §3]."""
    if not (0 <= i < N_COLS and 0 <= j < N_ROWS):
        raise ValueError(f"hole ({i}, {j}) outside the {N_COLS}x{N_ROWS} grid")
    return GRID_MARGIN_X + i * GRID_PITCH, GRID_MARGIN_Z + j * GRID_PITCH


def hold_position(i: int, j: int) -> tuple[float, float, float]:
    """Reference point of the hold bolted through holes (i - 1, j) and (i + 1, j): the bolt
    midpoint on the face, i.e. hole (i, j) itself [Rules §4, Figure 3]."""
    if not 1 <= i <= N_COLS - 2:
        raise ValueError(f"a hold centred on column {i} has a bolt off the grid")
    x, z = hole_xz(i, j)
    return x, 0.0, z


def nearest_hole(x: float, z: float) -> tuple[int, int]:
    """Grid indices (i, j) of the hold whose reference point is nearest (x, z)."""
    i = round((x - GRID_MARGIN_X) / GRID_PITCH)
    j = round((z - GRID_MARGIN_Z) / GRID_PITCH)
    return max(1, min(N_COLS - 2, i)), max(0, min(N_ROWS - 1, j))


BOLT_SPACING = 2 * GRID_PITCH  # holes two pitches apart, 0.030 m [Rules §4, Figure 3]


@dataclass(frozen=True)
class CupHold:
    """The provided hold, as the organisers' own mesh (`meshes.HOLD_STL`) [Rules v1.1 Figure 3]:
    a 3 mm back plate bolted through two holes two pitches (0.030 m) apart, and a bucket with a
    spherical socket that takes the robot's ball end. This adds the bolts and the physics.

    Offsets from `Hold.position` (the bolt midpoint on the wall face), wall frame:
    socket centre at (0, -socket_out, -socket_drop) [Rules Figure 3, hold.stl].
    """

    mesh: Path = HOLD_STL
    socket_out: float = 0.026  # socket centre, out from the wall face [Rules Figure 3]
    socket_drop: float = 0.014  # socket centre, below the bolt line [Rules Figure 3]
    socket_radius: float = 0.0127  # the spherical socket, measured on hold.stl [Ours]
    plate_thickness: float = 0.003  # [hold.stl]
    bolt_radius: float = 0.0030  # M3 head
    bolt_spacing: float = BOLT_SPACING
    friction: float = 1.0

    @property
    def mesh_pos(self) -> tuple[float, float, float]:
        """Where the mesh origin (the socket centre) sits relative to `Hold.position`."""
        return 0.0, -self.socket_out, -self.socket_drop

    def vertices(self) -> np.ndarray:
        """Mesh vertices relative to `Hold.position`, in the wall frame [m]."""
        return read_stl(self.mesh).reshape(-1, 3) * HOLD_STL_SCALE + self.mesh_pos

    def validate(self) -> None:
        v = self.vertices()
        radial = max(
            float(np.hypot(v[:, 0], v[:, 2]).max()), self.bolt_spacing / 2 + self.bolt_radius
        )
        # Their mesh reaches 30.08 mm, which their own gen_hold.py accepts: tessellation, 0.1 mm.
        if radial > ENVELOPE_RADIUS + 1e-4:
            raise ValueError(f"hold radius {radial * 1e3:.1f} mm exceeds the 30 mm envelope")
        if -v[:, 1].min() > ENVELOPE_LENGTH or v[:, 1].max() > 1e-6:
            raise ValueError("hold is not within the 100 mm envelope in front of the wall")

    def grasp_point(self, position: tuple[float, float, float], ball_radius: float):
        """Where a limb's ball centre rests, in the wall frame: at the bottom of the spherical
        socket, i.e. the socket centre lowered by the clearance between socket and ball."""
        x, _, z = position
        sag = max(0.0, self.socket_radius - ball_radius)
        return (x, -self.socket_out, z - self.socket_drop - sag)


@dataclass(frozen=True)
class LimbTip:
    """The climbing end-effector on each hand and foot, from their video.

    Side-on (frames 880-940) each forearm ends in a short bent finger with a
    grey ball on the end, a J-hook pointing at the wall when the arm is
    raised. The feet carry the same hook past the toe, bent down out of the
    sole, so the plate stays above the cup [Ours]. The ball is what sits in a
    cup. Dimensions [Ours], read against the
    forearm (~0.020 m wide), and kept parametric until the climbing limbs are
    published [Rules §2.5].
    """

    ball_radius: float = 0.010
    finger_radius: float = 0.0045
    reach: float = 0.008  # straight on past the end of the link
    hook: float = 0.018  # then bent towards the wall
    toe: float = 0.006  # foot: straight on past the toe
    toe_hook: float = 0.022  # then bent down, out of the sole
    mass: float = 0.008
