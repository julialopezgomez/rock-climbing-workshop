"""Build the MuJoCo scene for a wall model.

Assembled with MjSpec from the Microban model, so the upstream clone stays
untouched and swapping in the organisers' simulator means changing MICROBAN_XML
and the limb body names. The generated XML is written out for inspection and
for placo, which needs a single kinematic tree.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from rcr.robot import hooks
from rcr.sim import markers as mk
from rcr.sim.geometry import (
    HOLE_DIAMETER,
    N_COLS,
    N_ROWS,
    WALL_THICKNESS,
    CupHold,
    LimbTip,
    hole_xz,
)
from rcr.sim.meshes import HOLD_STL_SCALE
from rcr.types import WALL_HEIGHT, WALL_WIDTH, WallModel


def _microban_xml() -> Path:
    """Microban's MJCF: MICROBAN_XML; else the one next to MICROBAN_URDF in the same checkout
    (src/model/urdf/robot.urdf -> src/model/mjcf/robot.xml), so either variable is enough;
    else ~/Code/rock-climbing/microban."""
    if os.environ.get("MICROBAN_XML"):
        return Path(os.environ["MICROBAN_XML"]).expanduser()
    if os.environ.get("MICROBAN_URDF"):
        return Path(os.environ["MICROBAN_URDF"]).expanduser().parent.parent / "mjcf" / "robot.xml"
    return Path.home() / "Code/rock-climbing/microban/src/model/mjcf/robot.xml"


MICROBAN_XML = _microban_xml()

# Per limb: the Microban parent body, where the end-effector starts, and the
# path of its finger as unit directions (each followed for one length from
# LimbTip), all in the parent's frame. The ball sits at the end. The hooks
# themselves, and why they are shaped so, are in rcr.robot.hooks [Ours].
LIMB_PARENTS = {
    limb: (hook.link, hook.start, tuple(direction for direction, _ in hook.segments))
    for limb, hook in hooks.HOOKS.items()
}

SERVO_TORQUE_LIMIT = 0.64  # [N.m] 1.75 A x 0.366 N.m/A [Microban constants.py]
ROBOT_PREFIX = "rob_"


@dataclass
class SceneConfig:
    hold: CupHold = field(default_factory=CupHold)
    limb: LimbTip = field(default_factory=LimbTip)
    add_markers: bool = True
    add_cameras: bool = True
    camera_distance: float = 1.5
    torque_limit: float = SERVO_TORQUE_LIMIT
    robot_start: tuple[float, float, float] = (0.40, -0.14, 0.75)


def pegboard_texture(path: Path, px_per_mm: int = 2) -> Path:
    """Render the M3 hole grid as a texture — 4836 hole geoms would be absurd."""
    import cv2

    w = int(WALL_WIDTH * 1000 * px_per_mm)
    h = int(WALL_HEIGHT * 1000 * px_per_mm)
    img = np.full((h, w, 3), (150, 199, 232), dtype=np.uint8)  # plywood tan (cv2 writes BGR)
    r = max(1, int(HOLE_DIAMETER / 2 * 1000 * px_per_mm))
    for j in range(N_ROWS):
        for i in range(N_COLS):
            x, z = hole_xz(i, j)
            cx = int(x * 1000 * px_per_mm)
            cy = int((WALL_HEIGHT - z) * 1000 * px_per_mm)  # image y runs down
            cv2.circle(img, (cx, cy), r, (60, 75, 95), -1, lineType=cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    return path


def _add_wall(spec: mujoco.MjSpec, wall: WallModel, cfg: SceneConfig, asset_dir: Path) -> None:
    board = spec.worldbody.add_body(name="wall", pos=[0, 0, 0])

    # The organisers' hold mesh, in mm. Absolute: meshdir points at Microban's assets.
    spec.add_mesh(name="hold", file=str(cfg.hold.mesh.resolve()), scale=[HOLD_STL_SCALE] * 3)

    # The hole grid is always there; the markers are a separate, optional thing.
    pegboard_texture(asset_dir / "pegboard.png")
    spec.add_texture(name="pegboard", type=mujoco.mjtTexture.mjTEXTURE_2D, file="pegboard.png")
    spec.add_material(name="pegboard_mat", textures=["", "pegboard"], texuniform=False)

    board.add_geom(
        name="wall_board",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[WALL_WIDTH / 2, WALL_THICKNESS / 2, WALL_HEIGHT / 2],
        size=[WALL_WIDTH / 2, WALL_THICKNESS / 2, WALL_HEIGHT / 2],
        material="pegboard_mat",
        rgba=[0.95, 0.88, 0.75, 1.0],
        friction=[0.8, 0.005, 0.0001],
        # Stiff contact: the default lets the robot sink ~10 mm into the board.
        solref=[0.002, 1.0],
        solimp=[0.98, 0.999, 0.0005, 0.5, 2.0],
    )

    cup = cfg.hold
    cup.validate()
    for hold in wall.holds:
        x, _, z = hold.position
        _add_cup(board, hold.id, x, z, cup)

    if cfg.add_markers:
        for mid in sorted(mk.CORNERS):
            png = mk.marker_png(mid, asset_dir)
            spec.add_texture(
                name=f"aruco_{mid}", type=mujoco.mjtTexture.mjTEXTURE_2D, file=png.name
            )
            spec.add_material(
                name=f"aruco_{mid}_mat", textures=["", f"aruco_{mid}"], texuniform=False
            )
            mx, mz = mk.plate_centre(mid)
            board.add_geom(
                name=f"marker_{mid}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[mx, -0.0005, mz],
                size=[mk.PLATE / 2, 0.0005, mk.PLATE / 2],
                material=f"aruco_{mid}_mat",
                contype=0,
                conaffinity=0,
            )


def _add_cup(board, hold_id: str, x: float, z: float, cup: CupHold) -> None:
    """One hold — the organisers' mesh, moved only: its frame is the wall's — plus its bolts.

    Holds live on contype/conaffinity 2 so they do not collide with the robot:
    the weld in attach.py is the grip model (see sim/README.md).
    """
    off_wall = 0.0004  # clear of the board face, or the plate z-fights with it
    dx, dy, dz = cup.mesh_pos
    board.add_geom(
        name=f"hold_{hold_id}",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="hold",
        pos=[x + dx, dy - off_wall, z + dz],
        material="cup_mat",
        friction=[cup.friction, 0.005, 0.0001],
        contype=2,
        conaffinity=2,
    )
    for sign in (-1, +1):
        board.add_geom(
            name=f"hold_{hold_id}_bolt{'lr'[sign > 0]}",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=[x + sign * cup.bolt_spacing / 2, -(cup.plate_thickness + off_wall), z],
            size=[cup.bolt_radius, 0.0001, 0],
            euler=[np.pi / 2, 0, 0],
            rgba=[0.20, 0.20, 0.22, 1.0],
            contype=2,
            conaffinity=2,
        )


def _robot_spec(cfg: SceneConfig) -> mujoco.MjSpec:
    if not MICROBAN_XML.exists():
        raise FileNotFoundError(
            f"Microban model not found at {MICROBAN_XML}: clone github.com/Rhoban/microban and "
            "set MICROBAN_URDF to its src/model/urdf/robot.urdf, or MICROBAN_XML to its "
            "src/model/mjcf/robot.xml (see src/rcr/sim/README.md)"
        )
    spec = mujoco.MjSpec.from_file(str(MICROBAN_XML))
    spec.meshdir = str((MICROBAN_XML.parent / "assets").resolve())
    grey = [0.62, 0.63, 0.66, 1.0]
    tip = cfg.limb
    for limb, (parent_name, start, path) in LIMB_PARENTS.items():
        parent = spec.body(parent_name)
        if parent is None:
            raise ValueError(f"{parent_name} missing from {MICROBAN_XML}")
        body = parent.add_body(name=limb, pos=start)
        lengths = (tip.reach, tip.hook) if "hand" in limb else (tip.toe, tip.toe_hook)
        points = [np.zeros(3)]
        for direction, length in zip(path, lengths, strict=True):
            points.append(points[-1] + np.array(direction) * length)
        for k, (a, b) in enumerate(zip(points, points[1:], strict=False)):
            body.add_geom(
                name=f"{limb}_finger{k}",
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                fromto=[*a, *b],
                size=[tip.finger_radius, 0, 0],
                mass=tip.mass * hooks.SEGMENT_MASS_SHARE,
                group=2,  # Microban's visible group; 3 is its hidden collision group
                rgba=grey,
            )
        ball = points[-1] + np.array(path[-1]) * tip.ball_radius * hooks.BALL_INSET
        body.add_geom(
            name=f"{limb}_ball",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            pos=list(ball),
            size=[tip.ball_radius, 0, 0],
            mass=tip.mass * (1 - hooks.SEGMENT_MASS_SHARE * len(path)),
            group=2,
            rgba=grey,
        )
        body.add_site(name=f"{limb}_tip", pos=list(ball), size=[0.003] * 3, group=4)

    for act in spec.actuators:
        act.forcerange = [-cfg.torque_limit, cfg.torque_limit]
        act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
    return spec


def robot_only_xml(
    cfg: SceneConfig | None = None, out: Path = Path("build/robot_climb.xml")
) -> Path:
    """The robot with its balls, alone — placo needs a single kinematic tree."""
    spec = _robot_spec(cfg or SceneConfig())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(spec.to_xml())
    return out


def build_scene(
    wall: WallModel, cfg: SceneConfig | None = None, out: Path | None = Path("build/scene.xml")
) -> mujoco.MjSpec:
    cfg = cfg or SceneConfig()
    spec = mujoco.MjSpec()
    spec.modelname = f"climbwall_{wall.id}"
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.iterations = 100
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960

    asset_dir = (out.parent / "assets") if out else Path("build/assets")
    asset_dir.mkdir(parents=True, exist_ok=True)
    spec.texturedir = str(asset_dir.resolve())
    spec.meshdir = str((MICROBAN_XML.parent / "assets").resolve())

    _add_wall(spec, wall, cfg, asset_dir)
    spec.visual.headlight.ambient = [0.22, 0.22, 0.24]
    spec.visual.headlight.diffuse = [0.35, 0.35, 0.35]
    for pos, direction, shadow in (
        ([WALL_WIDTH / 2 - 0.5, -1.6, 2.2], [0.2, 0.35, -1], True),
        ([WALL_WIDTH / 2, -2.2, 0.8], [0, 1, -0.05], False),
    ):
        spec.worldbody.add_light(
            pos=pos,
            dir=direction,
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
            castshadow=shadow,
        )
    spec.add_texture(
        name="sky",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.38, 0.45, 0.55],
        rgb2=[0.22, 0.27, 0.34],
        width=256,
        height=256,
    )
    spec.add_material(name="floor_mat", rgba=[0.16, 0.17, 0.20, 1.0], reflectance=0.25)
    spec.add_material(
        name="cup_mat", rgba=[0.80, 0.32, 0.10, 1.0], specular=0.25, shininess=0.3, reflectance=0.0
    )
    spec.add_material(
        name="cup_plate_mat", rgba=[0.86, 0.44, 0.22, 1.0], specular=0.05, shininess=0.05
    )
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=[WALL_WIDTH / 2, -0.6, 0],
        size=[3, 3, 0.05],
        material="floor_mat",
    )
    if cfg.add_cameras:
        centre = spec.worldbody.add_body(
            name="wall_centre", pos=[WALL_WIDTH / 2, 0.0, WALL_HEIGHT / 2], mass=0.0
        )
        centre.add_site(name="wall_centre_site", size=[0.001] * 3, group=4)
        for name, side in (("cam_left", -1.0), ("cam_right", +1.0)):
            spec.worldbody.add_camera(
                name=name,
                pos=[WALL_WIDTH / 2 + side * 0.60, -cfg.camera_distance, WALL_HEIGHT * 0.5],
                mode=mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY,
                targetbody="wall_centre",
                fovy=70.0,
            )

    frame = spec.worldbody.add_frame(pos=list(cfg.robot_start), quat=_facing_wall())
    spec.attach(_robot_spec(cfg), prefix=ROBOT_PREFIX, frame=frame)

    for limb in LIMB_PARENTS:
        eq = spec.add_equality(
            name=f"grip_{limb}",
            type=mujoco.mjtEq.mjEQ_WELD,
            objtype=mujoco.mjtObj.mjOBJ_BODY,
            name1="wall",
            name2=f"{ROBOT_PREFIX}{limb}",
            active=False,
        )
        eq.data = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0.0]

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(spec.to_xml())
        robot_only_xml(cfg, out.parent / "robot_climb.xml")
    return spec


def _facing_wall() -> list[float]:
    """Microban's front is +x in its own frame; the wall is at +y here."""
    half = np.pi / 4
    return [float(np.cos(half)), 0.0, 0.0, float(np.sin(half))]
