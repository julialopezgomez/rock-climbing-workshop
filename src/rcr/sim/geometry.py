"""Wall, hold and end-effector geometry, in the wall frame.

Wall frame [Ours, INTERFACES.md]: origin at the bottom-left corner of the wall
face, x right, z up, y into the wall (the robot is at y < 0). The MuJoCo world
*is* this frame, so nothing converts coordinates.

Shapes follow the organisers' own simulator video (images/microban_climb_poster.jpg
on the workshop site): a pegboard wall, orange cup holds on a trapezoidal
backplate with two bolts, and ball-ended limbs that sit in the cups.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rcr.sim.meshes import BOWL_STANDOFF, CupShape, envelope_radius
from rcr.types import GRID_MARGIN, GRID_PITCH, WALL_HEIGHT, WALL_WIDTH

WALL_THICKNESS = 0.010  # [m] [Rules §3]
HOLE_DIAMETER = 0.0034  # [m] M3 through-hole [Rules §3]

N_COLS = int((WALL_WIDTH - 2 * GRID_MARGIN) / GRID_PITCH) + 1  # 53 [Rules §3]
N_ROWS = int((WALL_HEIGHT - 2 * GRID_MARGIN) / GRID_PITCH) + 1  # 93

ENVELOPE_RADIUS = 0.030  # [m] hold must fit Ø0.060 x 0.100 m [Rules §4]
ENVELOPE_LENGTH = 0.100


def hole_xz(i: int, j: int) -> tuple[float, float]:
    """Centre of grid hole (i, j) on the wall face [Rules §3]."""
    if not (0 <= i < N_COLS and 0 <= j < N_ROWS):
        raise ValueError(f"hole ({i}, {j}) outside the {N_COLS}x{N_ROWS} grid")
    return GRID_MARGIN + i * GRID_PITCH, GRID_MARGIN + j * GRID_PITCH


def hold_position(i: int, j: int) -> tuple[float, float, float]:
    """Hold reference point: midpoint of the two bolts, on the face [Rules §4]."""
    x, z = hole_xz(i, j)
    return x + GRID_PITCH / 2, 0.0, z


def nearest_hole(x: float, z: float) -> tuple[int, int]:
    i = round((x - GRID_MARGIN - GRID_PITCH / 2) / GRID_PITCH)
    j = round((z - GRID_MARGIN) / GRID_PITCH)
    return max(0, min(N_COLS - 2, i)), max(0, min(N_ROWS - 1, j))


BOLT_SPACING = GRID_PITCH  # two horizontally adjacent holes, 0.015 m apart [Rules §4]


@dataclass(frozen=True)
class CupHold:
    """An orange cup, as in the organisers' render: a trapezoidal backplate
    with a half-bowl fused to its lower part, opening upwards. The shape lives
    in `meshes.CupShape`; this adds the bolts and the physics.

    Attachment [Rules §4]: two M3 bolts through two *horizontally adjacent*
    holes, 0.015 m apart, nuts on the back. Their render puts the bolts at the
    plate's top corners, two pitches (0.030 m) apart, which the draft rules do
    not allow; we follow the rules and keep their bolt line at the top.
    """

    shape: CupShape = field(default_factory=CupShape)
    bolt_radius: float = 0.0030  # M3 head
    bolt_spacing: float = BOLT_SPACING
    friction: float = 1.0

    def validate(self) -> None:
        radial = max(envelope_radius(self.shape), self.bolt_spacing / 2 + self.bolt_radius)
        if radial > ENVELOPE_RADIUS:
            raise ValueError(f"hold radius {radial * 1e3:.1f} mm exceeds the 30 mm envelope")
        if self.shape.plate_thickness + 2 * self.shape.bowl_radius > ENVELOPE_LENGTH:
            raise ValueError("hold is longer than the 100 mm envelope")

    def grasp_point(self, position: tuple[float, float, float], ball_radius: float):
        """Where a limb's ball rests in the bottom of the bowl, in the wall frame."""
        x, _, z = position
        s = self.shape
        bottom = z + s.bowl_rim - s.bowl_depth + s.wall
        return (x, -(s.plate_thickness + s.bowl_radius * BOWL_STANDOFF), bottom + ball_radius)


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
