"""The contracts between modules. INTERFACES.md is the documentation of record for this file.

Units are SI (m, s, rad). Wall frame [Ours]: origin at the bottom-left corner of the wall face,
x to the right, z up (seen from the front), y into the wall: the robot is at y < 0. Right-handed.

Fields marked CHANGE are expected to change; change them only together with INTERFACES.md,
the examples in data/examples/ and the tests (see CLAUDE.md).

Pi-safe: stdlib only, Python 3.9 or newer (the annotations are not evaluated at run time).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

# --- Constants ----------------------------------------------------------------------------------

WALL_WIDTH = 0.800  # [m] [Rules §3]
WALL_HEIGHT = 1.400  # [m] [Rules §3]
GRID_PITCH = 0.015  # [m] M3 hole grid [Rules §3]
# [m] border before the first hole: 17.5 mm left/right, 10 mm top/bottom, so 52 x 93 holes
# [Rules v1.1 Figure 2]. (The §3 text still says 10 mm on all sides; the figure is newer.)
GRID_MARGIN_X = 0.0175
GRID_MARGIN_Z = 0.010

CONTROL_HZ = 50  # robot loop rate [Microban scheduler]

# Microban motor names [Microban constants.py]. CHANGE: the climbing limbs are not published yet.
MICROBAN_JOINTS = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_elbow",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_elbow",
    "head",
)

Limb = Literal["right_hand", "left_hand", "right_foot", "left_foot"]
LIMBS: tuple[Limb, ...] = ("right_hand", "left_hand", "right_foot", "left_foot")


# --- Wall ---------------------------------------------------------------------------------------


@dataclass
class Hold:
    id: str
    # [m] wall frame: midpoint between the hold's two bolts, on the wall face (y = 0) [Rules §4].
    # The bolts are two pitches apart, so this is the grid hole between them.
    position: tuple[float, float, float]
    # CHANGE: hold shape / how to grip it. The provided holds are not published yet.
    kind: str = "unknown"


@dataclass
class Stance:
    # Limb -> hold id, or None if the limb touches nothing. All four limbs are always present.
    # CHANGE: "ground" is reserved as a hold id for a ground start [Rules §8].
    contacts: dict[Limb, str | None]


@dataclass
class WallModel:
    """Everything the planner needs about the problem: the holds and where the robot starts."""

    id: str
    level: int  # 1 ladder, 2 ladder with gaps, 3 lateral moves [Rules §3]
    holds: list[Hold]
    start: Stance  # the robot's contacts at the start of the run [Ours]


# --- Plan and trajectory (laptop -> robot) ------------------------------------------------------


@dataclass
class ContactPlan:
    wall_id: str
    # stances[0] == wall.start. Consecutive stances usually differ by one limb (not enforced).
    stances: list[Stance]


@dataclass
class Trajectory:
    """The full joint-space motion the robot executes. Joints not listed keep Microban's neutral."""

    joint_names: list[str]  # Microban motor names
    times: list[float]  # [s] from the start of the climb, strictly increasing
    positions: list[list[float]]  # [rad] one row per time, one column per joint
    # Optional, to replay the plan exactly in the sim; the robot uses `positions` only [Ours].
    # planned_positions: the optimiser's joint angles (today the same as `positions`: no pre-load
    # against servo sag). trunk_poses: the trunk per time, wall frame: x, y, z [m], qw, qx, qy, qz.
    planned_positions: list[list[float]] | None = None  # [rad] like `positions`
    trunk_poses: list[list[float]] | None = None
    # CHANGE: likely additions: velocities, per-joint gains, index of the stance of each sample.


@dataclass
class Feasibility:
    feasible: bool
    cost: float = 0.0  # lower is better; meaning owned by the optimiser. CHANGE
    reason: str = ""  # human-readable, for debugging


# Owned by the optimiser and passed to the planner as an argument: (wall, from, to) -> verdict.
FeasibilityFn = Callable[[WallModel, Stance, Stance], Feasibility]


# --- Robot loop (robot side, 50 Hz) -------------------------------------------------------------


@dataclass
class Pose:
    position: tuple[float, float, float]  # [m] wall frame
    quat: tuple[float, float, float, float]  # (w, x, y, z) [Microban order]


@dataclass
class RobotState:
    """What the controller sees each tick. Subset of Microban's observer.RobotState."""

    time: float  # [s] since the start of the climb
    joint_positions: dict[str, float]  # [rad]
    joint_velocities: dict[str, float]  # [rad/s]
    imu_quat: tuple[float, float, float, float]  # body orientation (w, x, y, z)
    gyro: tuple[float, float, float]  # [rad/s] body frame
    # Robot base in the wall frame, from the cameras over the link, if available. CHANGE
    base_pose: Pose | None = None


@dataclass
class JointTargets:
    # [rad] motor name -> position target; the servos run their own P loop [Microban MotorCommand].
    # Joints absent from the dict keep their last target [Microban dev_v2]. CHANGE: per-joint kp.
    positions: dict[str, float]
