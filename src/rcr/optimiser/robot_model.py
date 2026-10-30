"""The robot for the optimiser: the climbing model's URDF as published, as CasADi.

The URDF (microban_model, climbing variant: rcr.robot.microban) is read as is, never edited or
copied. Its climbing forearms and feet end in balls, the limbs' contact points, found from
rcr.robot.hooks (as in the sim). The body's shape, for keeping it off the cups, comes from the
visual meshes of the model's MJCF (keep_out_points).

Configuration q (NQ = 26), the same layout as the sim's MuJoCo qpos:
    q[0:3]   trunk position in the wall frame [m]
    q[3:7]   trunk orientation, quaternion (w, x, y, z)   [our convention, INTERFACES.md]
    q[7:26]  the 19 motor angles [rad], in rcr.types.MICROBAN_JOINTS order
Tangent dq (NV = 25): dq[0:3] linear, dq[3:6] angular (both in the trunk frame, as Pinocchio's
free-flyer), dq[6:25] motor angles.

Wall frame (INTERFACES.md): x right, z up, y into the wall; the robot is at y < 0. Gravity -z.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import casadi as ca
import numpy as np

from rcr.robot import hooks, microban
from rcr.robot.hooks import HOOKS, ball_centre
from rcr.types import LIMBS, MICROBAN_JOINTS

NQ, NV, NJ = 26, 25, len(MICROBAN_JOINTS)
GRAVITY = 9.81  # [m/s^2], along -z of the wall frame
ROOT_LINK = "trunk"  # the URDF link the free-flyer base is attached to [Microban]

# --- Reading Microban's files -------------------------------------------------------------------


@dataclass(frozen=True)
class UrdfJoint:
    name: str
    revolute: bool  # else fixed (Microban's URDF has only these two types)
    parent: str
    child: str
    xyz: np.ndarray  # [m] origin in the parent link frame
    rotation: np.ndarray  # 3x3, from the origin's rpy
    axis: np.ndarray  # unit axis in the child frame (revolute only)
    lower: float = 0.0  # [rad]
    upper: float = 0.0


@dataclass(frozen=True)
class UrdfLink:
    name: str
    mass: float  # [kg]
    com: np.ndarray  # [m] centre of mass in the link frame


@dataclass(frozen=True)
class Urdf:
    links: dict[str, UrdfLink]
    joints: list[UrdfJoint]  # in an order where each parent is placed before its child


def urdf_path() -> Path:
    """The climbing model's URDF (rcr.robot.microban: microban_model, or MICROBAN_URDF)."""
    return microban.urdf_path()


def rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr, cp, sp, cy, sy = (np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch),
                              np.cos(yaw), np.sin(yaw))  # fmt: skip
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _floats(text: str | None, default: str) -> np.ndarray:
    return np.array([float(v) for v in (text or default).split()])


def read_urdf(path: Path) -> Urdf:
    """Links (mass, CoM) and joints (origin, axis, limits) from a URDF, joints parent-first."""
    root = ET.parse(path).getroot()
    links = {}
    for link in root.findall("link"):
        inertial = link.find("inertial")
        mass = float(inertial.find("mass").get("value")) if inertial is not None else 0.0
        origin = inertial.find("origin") if inertial is not None else None
        com = _floats(origin.get("xyz") if origin is not None else None, "0 0 0")
        links[link.get("name")] = UrdfLink(link.get("name"), mass, com)

    joints = []
    for joint in root.findall("joint"):
        kind = joint.get("type")
        if kind not in ("revolute", "fixed"):
            raise ValueError(f"{path}: joint {joint.get('name')} is {kind!r}; only revolute/fixed")
        origin = joint.find("origin")
        xyz = _floats(origin.get("xyz") if origin is not None else None, "0 0 0")
        rpy = _floats(origin.get("rpy") if origin is not None else None, "0 0 0")
        axis = _floats(joint.find("axis").get("xyz") if joint.find("axis") is not None else None,
                       "1 0 0")  # fmt: skip
        limit = joint.find("limit")
        joints.append(
            UrdfJoint(
                name=joint.get("name"),
                revolute=kind == "revolute",
                parent=joint.find("parent").get("link"),
                child=joint.find("child").get("link"),
                xyz=xyz,
                rotation=rpy_matrix(*rpy),
                axis=axis / np.linalg.norm(axis) if kind == "revolute" else axis,
                lower=float(limit.get("lower")) if limit is not None else 0.0,
                upper=float(limit.get("upper")) if limit is not None else 0.0,
            )
        )
    return Urdf(links=links, joints=_parent_first(joints))


def _parent_first(joints: list[UrdfJoint]) -> list[UrdfJoint]:
    placed, ordered, todo = {ROOT_LINK}, [], list(joints)
    while todo:
        ready = [j for j in todo if j.parent in placed]
        if not ready:
            raise ValueError(
                f"URDF links not reachable from {ROOT_LINK}: {[j.child for j in todo]}"
            )
        for joint in ready:
            ordered.append(joint)
            placed.add(joint.child)
            todo.remove(joint)
    return ordered


# --- Kinematics, symbolically -------------------------------------------------------------------


def quaternion_matrix(quat):
    """3x3 rotation from a unit quaternion (w, x, y, z)."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    return ca.vertcat(
        ca.horzcat(1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        ca.horzcat(2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        ca.horzcat(2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
    )


def axis_rotation(axis: np.ndarray, angle):
    """3x3 rotation by `angle` [rad] about a unit `axis` (Rodrigues)."""
    k = ca.DM([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return ca.DM.eye(3) + ca.sin(angle) * k + (1 - ca.cos(angle)) * (k @ k)


def link_poses(urdf: Urdf, q) -> dict[str, tuple]:
    """(position, rotation) of every link in the wall frame, for configuration q."""
    angle = {name: q[7 + i] for i, name in enumerate(MICROBAN_JOINTS)}
    poses = {ROOT_LINK: (q[0:3], quaternion_matrix(q[3:7]))}
    for joint in urdf.joints:
        p_parent, r_parent = poses[joint.parent]
        position = p_parent + r_parent @ joint.xyz
        rotation = r_parent @ joint.rotation
        if joint.revolute:
            rotation = rotation @ axis_rotation(joint.axis, angle[joint.name])
        poses[joint.child] = (position, rotation)
    return poses


def _skew(v):
    return ca.vertcat(
        ca.horzcat(0, -v[2], v[1]), ca.horzcat(v[2], 0, -v[0]), ca.horzcat(-v[1], v[0], 0)
    )


def _exp_coefficients(angle_sq):
    """sin(t)/t, (1-cos t)/t^2, (t-sin t)/t^3 for t^2 = angle_sq, with Taylor series near 0."""
    t = ca.sqrt(angle_sq + 1e-300)
    small = angle_sq < 1e-8
    a = ca.if_else(small, 1 - angle_sq / 6, ca.sin(t) / t)
    b = ca.if_else(small, 0.5 - angle_sq / 24, (1 - ca.cos(t)) / angle_sq)
    c = ca.if_else(small, 1.0 / 6 - angle_sq / 120, (t - ca.sin(t)) / (angle_sq * t))
    return a, b, c


def _quaternion_exp(omega):
    """Unit quaternion (w, x, y, z) of the rotation vector omega."""
    angle_sq = ca.dot(omega, omega)
    t = ca.sqrt(angle_sq + 1e-300)
    half_sinc = ca.if_else(angle_sq < 1e-8, 0.5 - angle_sq / 48, ca.sin(t / 2) / t)
    return ca.vertcat(ca.cos(t / 2), half_sinc * omega)


def _quaternion_product(p, r):
    return ca.vertcat(
        p[0] * r[0] - p[1] * r[1] - p[2] * r[2] - p[3] * r[3],
        p[0] * r[1] + p[1] * r[0] + p[2] * r[3] - p[3] * r[2],
        p[0] * r[2] - p[1] * r[3] + p[2] * r[0] + p[3] * r[1],
        p[0] * r[3] + p[1] * r[2] - p[2] * r[1] + p[3] * r[0],
    )


def integrate_expression(q, dq):
    """q (+) dq, as Pinocchio's free-flyer: the trunk moves by exp(dq[0:6]) in its own frame."""
    v, omega = dq[0:3], dq[3:6]
    _, b, c = _exp_coefficients(ca.dot(omega, omega))
    k = _skew(omega)
    left_jacobian = ca.DM.eye(3) + b * k + c * (k @ k)  # SE(3) exp: translation = R0 V(omega) v
    position = q[0:3] + quaternion_matrix(q[3:7]) @ (left_jacobian @ v)
    quaternion = _quaternion_product(q[3:7], _quaternion_exp(omega))
    return ca.vertcat(position, quaternion, q[7:] + dq[6:])


def difference(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    """dq (NV) with integrate(q0, dq) = q1: the inverse of integrate_expression (numpy)."""
    w0, x0, y0, z0 = q0[3:7]
    relative = (
        np.array(  # conj(quaternion0) * quaternion1: the trunk's turn, in its own frame
            [
                [w0, x0, y0, z0],
                [-x0, w0, z0, -y0],
                [-y0, -z0, w0, x0],
                [-z0, y0, -x0, w0],
            ]
        )
        @ np.asarray(q1[3:7], dtype=float)
    )
    relative *= np.sign(relative[0]) or 1.0  # the shorter way round
    sine = np.linalg.norm(relative[1:])
    angle = 2 * np.arctan2(sine, relative[0])
    omega = relative[1:] * (angle / sine if sine > 1e-12 else 2.0)
    k = np.array([[0, -omega[2], omega[1]], [omega[2], 0, -omega[0]], [-omega[1], omega[0], 0]])
    if angle > 1e-6:
        b, c = (1 - np.cos(angle)) / angle**2, (angle - np.sin(angle)) / angle**3
    else:
        b, c = 0.5, 1.0 / 6.0
    rotation0 = np.array(quaternion_matrix(ca.DM(q0[3:7])))
    local = rotation0.T @ (np.asarray(q1[0:3]) - np.asarray(q0[0:3]))
    v = np.linalg.solve(np.eye(3) + b * k + c * (k @ k), local)
    return np.concatenate([v, omega, np.asarray(q1[7:]) - np.asarray(q0[7:])])


# --- The body's shape, for keeping it off the cups ----------------------------------------------

# [m] around each ball, the part of its forearm or foot (the hook, the ball) that may go into a
# cup; beyond it, the rest of the limb is kept out (the forearm reaches 88 mm from its ball, the
# foot's sole plate 115 mm) [Ours]
BALL_ZONE = 0.05


# [m] the hook's shaft: the part of a forearm or foot this far from its ball (beyond the ball's
# surface), whose direction is the hook's (hook_axes) [Ours]
HOOK_SHAFT = 0.03


@lru_cache(maxsize=1)
def link_shapes() -> dict[str, np.ndarray]:
    """link -> Nx3 points of its drawn shape in the link frame: the vertices of the visual meshes
    of Microban's MJCF, whose bodies are the URDF's links (checked to 1e-8 m)."""
    import mujoco  # with the organisers' simulator, which the optimiser needs anyway

    m = mujoco.MjModel.from_xml_path(str(microban.mjcf_path()))
    shapes = {}
    for body in range(1, m.nbody):
        shape = []
        for g in np.flatnonzero((m.geom_bodyid == body) & (m.geom_group == 2)):
            if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mesh = m.geom_dataid[g]
            start, count = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
            rotation = np.zeros(9)
            mujoco.mju_quat2Mat(rotation, m.geom_quat[g])
            shape.append(
                m.mesh_vert[start : start + count] @ rotation.reshape(3, 3).T + m.geom_pos[g]
            )
        if shape:
            shapes[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body)] = np.vstack(shape)
    return shapes


@lru_cache(maxsize=1)
def keep_out_points() -> list[tuple[str, np.ndarray]]:
    """(link, point in the link frame): the 8 corners of a box around each link's shape
    (link_shapes), without the part of a forearm or foot within BALL_ZONE of its ball. A box
    contains its shape, so with all 8 corners on one side of a plane, the whole link is."""
    balls = {hook.link: ball_centre(hook) for hook in HOOKS.values()}
    points = []
    for link, shape in link_shapes().items():
        if link in balls:
            shape = shape[np.linalg.norm(shape - balls[link], axis=1) > BALL_ZONE]
        low, high = shape.min(axis=0), shape.max(axis=0)
        points += [(link, np.where([i & 1, i & 2, i & 4], high, low)) for i in range(8)]
    return points


@lru_cache(maxsize=1)
def hook_axes() -> dict[str, np.ndarray]:
    """limb -> unit vector in its link's frame, from its ball along its hook: towards the middle
    of the shape between the ball's surface and HOOK_SHAFT from its centre. The hands' hooks are
    straight shafts along the forearm (-x right, +x left); the feet's, a stub along +y."""
    axes = {}
    for limb, hook in HOOKS.items():
        shape, centre = link_shapes()[hook.link], ball_centre(hook)
        distance = np.linalg.norm(shape - centre, axis=1)
        shaft = shape[(distance > hooks.BALL_RADIUS + 0.002) & (distance < HOOK_SHAFT)]
        axis = shaft.mean(axis=0) - centre
        axes[limb] = axis / np.linalg.norm(axis)
    return axes


# --- The model ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class RobotModel:
    """CasADi functions of the robot. Each takes q (NQ); see the module docstring.

    limb_positions(q) -> 3x4   ball centres [m], one column per limb in rcr.types.LIMBS order
    body_points(q) -> 3xP      the corners of the body's boxes [m] (keep_out_points)
    hook_directions(q) -> 3x4  each limb's hook direction, a unit vector (hook_axes), LIMBS order
    limb_jacobians(q) -> 3x76  d(ball centre)/d(motor angles), the four 3x19 blocks side by side
    com(q) -> 3                centre of mass [m]
    gravity_torques(q) -> 19   motor torques [N·m] that hold the robot against gravity alone
    integrate(q, dq) -> NQ     q (+) dq
    """

    limb_positions: ca.Function
    body_points: ca.Function
    hook_directions: ca.Function
    limb_jacobians: ca.Function
    com: ca.Function
    gravity_torques: ca.Function
    integrate: ca.Function
    mass: float  # [kg] the whole climbing model
    lower: np.ndarray = field(repr=False)  # [rad] motor limits from the URDF, MICROBAN_JOINTS order
    upper: np.ndarray = field(repr=False)

    def limb_jacobian(self, q, limb: str):
        """3x19 translational Jacobian of one limb's ball centre w.r.t. the motor angles."""
        i = LIMBS.index(limb)
        return self.limb_jacobians(q)[:, i * NJ : (i + 1) * NJ]


@lru_cache(maxsize=1)
def load_robot_model() -> RobotModel:
    """Read the robot's URDF (climbing model), find the balls, build the CasADi functions (once)."""
    urdf = read_urdf(urdf_path())
    revolute = {j.name: j for j in urdf.joints if j.revolute}
    if set(revolute) != set(MICROBAN_JOINTS):
        extra, missing = set(revolute) - set(MICROBAN_JOINTS), set(MICROBAN_JOINTS) - set(revolute)
        raise ValueError(
            f"URDF motors differ from MICROBAN_JOINTS: extra {extra}, missing {missing}"
        )
    for limb, hook in HOOKS.items():
        if hook.link not in urdf.links:
            raise ValueError(f"hook {limb} is on link {hook.link!r}, which is not in the URDF")

    q = ca.SX.sym("q", NQ)
    poses = link_poses(urdf, q)
    balls = [poses[h.link][0] + poses[h.link][1] @ ball_centre(h) for h in HOOKS.values()]
    limb_positions = ca.horzcat(*[balls[LIMBS.index(limb)] for limb in LIMBS])

    # every link's mass, the climbing forearms and feet with their balls included [Microban]
    point_masses = [(link.mass, link.com, link.name) for link in urdf.links.values()]
    mass = sum(m for m, _, _ in point_masses)
    com = sum(m * (poses[n][0] + poses[n][1] @ p) for m, p, n in point_masses) / mass

    points = ca.horzcat(*[poses[link][0] + poses[link][1] @ p for link, p in keep_out_points()])
    axes = hook_axes()
    hook_dirs = ca.horzcat(*[poses[HOOKS[limb].link][1] @ axes[limb] for limb in LIMBS])

    angles = q[7:]
    jacobians = ca.horzcat(*[ca.jacobian(limb_positions[:, i], angles) for i in range(4)])
    gravity = mass * GRAVITY * ca.jacobian(com[2], angles).T  # d(potential energy)/d(angles)

    dq = ca.SX.sym("dq", NV)
    return RobotModel(
        limb_positions=ca.Function("limb_positions", [q], [limb_positions]),
        body_points=ca.Function("body_points", [q], [points]),
        hook_directions=ca.Function("hook_directions", [q], [hook_dirs]),
        limb_jacobians=ca.Function("limb_jacobians", [q], [jacobians]),
        com=ca.Function("com", [q], [com]),
        gravity_torques=ca.Function("gravity_torques", [q], [gravity]),
        integrate=ca.Function("integrate", [q, dq], [integrate_expression(q, dq)]),
        mass=float(mass),
        lower=np.array([revolute[n].lower for n in MICROBAN_JOINTS]),
        upper=np.array([revolute[n].upper for n in MICROBAN_JOINTS]),
    )
