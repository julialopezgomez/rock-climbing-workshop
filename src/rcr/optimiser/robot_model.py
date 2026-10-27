"""The robot for the optimiser: Microban's URDF as published, plus our climbing hooks, as CasADi.

Microban's URDF is read as is (never edited or copied: it is GPL-3.0) from its checkout, the
submodule third_party/microban or the one MICROBAN_URDF names (rcr.robot.microban). Its climbing
limbs are not published yet [Rules §2.5], so the four hooks are added to it here, from
rcr.robot.hooks (as in the sim).
The URDF has no collision shapes; Microban's own ones are in its MJCF, next to it in the same
checkout (../mjcf/robot.xml), whose bodies are the URDF's links. They are read from there.

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

from rcr.robot import microban
from rcr.robot.hooks import HOOKS, ball_centre, masses
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
    """Microban's URDF (rcr.robot.microban: the submodule, or MICROBAN_URDF / MICROBAN_XML)."""
    return microban.urdf_path()


def mjcf_path() -> Path:
    """Microban's MJCF, in the same checkout as the URDF (src/model/mjcf/robot.xml)."""
    return microban.mjcf_path()


@dataclass(frozen=True)
class CollisionShape:
    """One of Microban's collision shapes [Microban MJCF], in its link's frame."""

    link: str
    kind: str  # "box" or "cylinder" (the axis is the shape's local z)
    size: np.ndarray  # MJCF convention: box half-sizes (3); cylinder radius and half-length (2)
    offset: np.ndarray  # [m] the shape's centre, link frame
    rotation: np.ndarray  # 3x3, the shape's axes in the link frame


def read_collision_shapes(path: Path) -> list[CollisionShape]:
    """Microban's collision shapes (MJCF geoms of class "collision"), on the URDF's links."""
    shapes = []
    for body in ET.parse(path).getroot().iter("body"):
        for geom in body.findall("geom"):
            if geom.get("class") != "collision":
                continue
            kind = geom.get("type")
            if kind not in ("box", "cylinder"):
                raise ValueError(f"{path}: collision geom of type {kind!r} is not handled")
            shapes.append(
                CollisionShape(
                    link=body.get("name"),
                    kind=kind,
                    size=_floats(geom.get("size"), "0 0 0")[: 3 if kind == "box" else 2],
                    offset=_floats(geom.get("pos"), "0 0 0"),
                    rotation=_quat_rotation(geom.get("quat")),
                )
            )
    return shapes


@dataclass(frozen=True)
class VisualShape:
    """One of Microban's visual meshes [Microban MJCF]: what the sim draws, in its link's frame."""

    link: str
    mesh: Path  # STL, in metres
    offset: np.ndarray  # [m] the mesh frame's origin, link frame
    rotation: np.ndarray  # 3x3, the mesh frame's axes in the link frame


def read_visual_shapes(path: Path) -> list[VisualShape]:
    """Microban's visual meshes (MJCF geoms of class "visual"), on the URDF's links."""
    root = ET.parse(path).getroot()
    meshdir = path.parent / root.find("compiler").get("meshdir", ".")
    files = {}
    for mesh in root.iter("mesh"):
        file = mesh.get("file")
        files[mesh.get("name", Path(file).stem)] = meshdir / file
    shapes = []
    for body in root.iter("body"):
        for geom in body.findall("geom"):
            if geom.get("class") == "visual" and geom.get("type") == "mesh":
                shapes.append(
                    VisualShape(
                        link=body.get("name"),
                        mesh=files[geom.get("mesh")],
                        offset=_floats(geom.get("pos"), "0 0 0"),
                        rotation=_quat_rotation(geom.get("quat")),
                    )
                )
    return shapes


def read_collision_points(path: Path) -> dict[str, np.ndarray]:
    """link -> P x 3 points [m, link frame] on Microban's collision shapes, to keep out of the wall.

    Boxes give their 8 corners; cylinders 8 points on the rims of their two ends.
    """
    points: dict[str, list[np.ndarray]] = {}
    for shape in read_collision_shapes(path):
        if shape.kind == "box":
            local = [np.array([sx, sy, sz]) * shape.size
                     for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]  # fmt: skip
        else:
            radius, half = shape.size
            rim = ((1, 0), (0, 1), (-1, 0), (0, -1))
            local = [np.array([radius * c, radius * s, half * end])
                     for c, s in rim for end in (-1, 1)]  # fmt: skip
        points.setdefault(shape.link, []).extend(shape.offset + shape.rotation @ p for p in local)
    return {link: np.array(p) for link, p in points.items()}


def _quat_rotation(text: str | None) -> np.ndarray:
    """3x3 rotation of an MJCF quat attribute (w, x, y, z)."""
    w, x, y, z = _floats(text, "1 0 0 0") / np.linalg.norm(_floats(text, "1 0 0 0"))
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


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


# --- The model ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class RobotModel:
    """CasADi functions of the robot. Each takes q (NQ); see the module docstring.

    limb_positions(q) -> 3x4   ball centres [m], one column per limb in rcr.types.LIMBS order
    limb_jacobians(q) -> 3x76  d(ball centre)/d(motor angles), the four 3x19 blocks side by side
    com(q) -> 3                centre of mass [m]
    body_points(q) -> 3xP      points on Microban's collision shapes [m], to keep out of the wall
    gravity_torques(q) -> 19   motor torques [N·m] that hold the robot against gravity alone
    integrate(q, dq) -> NQ     q (+) dq
    """

    limb_positions: ca.Function
    limb_jacobians: ca.Function
    com: ca.Function
    body_points: ca.Function
    gravity_torques: ca.Function
    integrate: ca.Function
    # link_frames(q) -> (3xL positions, 3x3L rotations side by side), links in link_names order
    link_frames: ca.Function
    # point_jacobian[link](q0, dq, p) -> (x, dx/ddq): the world position of point p (link frame)
    # at q = integrate(q0, dq), and its 3xNV Jacobian w.r.t. dq, for collision gradients
    point_jacobian: dict[str, ca.Function] = field(repr=False)
    link_names: tuple[str, ...] = field(repr=False)
    mass: float  # [kg] Microban plus the hooks
    lower: np.ndarray = field(repr=False)  # [rad] motor limits from the URDF, MICROBAN_JOINTS order
    upper: np.ndarray = field(repr=False)

    def limb_jacobian(self, q, limb: str):
        """3x19 translational Jacobian of one limb's ball centre w.r.t. the motor angles."""
        i = LIMBS.index(limb)
        return self.limb_jacobians(q)[:, i * NJ : (i + 1) * NJ]


@lru_cache(maxsize=1)
def load_robot_model() -> RobotModel:
    """Read Microban's URDF, add the hooks, build the CasADi functions (once)."""
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

    point_masses = [(link.mass, link.com, link.name) for link in urdf.links.values()]
    point_masses += [(m, p, h.link) for h in HOOKS.values() for m, p in masses(h)]
    mass = sum(m for m, _, _ in point_masses)
    com = sum(m * (poses[n][0] + poses[n][1] @ p) for m, p, n in point_masses) / mass

    collision = read_collision_points(mjcf_path())
    unknown = set(collision) - set(urdf.links)
    if unknown:
        raise ValueError(f"MJCF collision shapes on links not in the URDF: {sorted(unknown)}")
    body = ca.horzcat(*[poses[n][0] + poses[n][1] @ p for n, pts in collision.items() for p in pts])

    angles = q[7:]
    jacobians = ca.horzcat(*[ca.jacobian(limb_positions[:, i], angles) for i in range(4)])
    gravity = mass * GRAVITY * ca.jacobian(com[2], angles).T  # d(potential energy)/d(angles)

    dq = ca.SX.sym("dq", NV)
    names = tuple(poses)
    frames = ca.Function(
        "link_frames",
        [q],
        [ca.horzcat(*[poses[n][0] for n in names]), ca.horzcat(*[poses[n][1] for n in names])],
    )
    return RobotModel(
        limb_positions=ca.Function("limb_positions", [q], [limb_positions]),
        limb_jacobians=ca.Function("limb_jacobians", [q], [jacobians]),
        com=ca.Function("com", [q], [com]),
        body_points=ca.Function("body_points", [q], [body]),
        gravity_torques=ca.Function("gravity_torques", [q], [gravity]),
        integrate=ca.Function("integrate", [q, dq], [integrate_expression(q, dq)]),
        link_frames=frames,
        point_jacobian=_point_jacobians(urdf, names),
        link_names=names,
        mass=float(mass),
        lower=np.array([revolute[n].lower for n in MICROBAN_JOINTS]),
        upper=np.array([revolute[n].upper for n in MICROBAN_JOINTS]),
    )


@lru_cache(maxsize=1)
def limb_reach() -> dict[str, tuple[np.ndarray, float]]:
    """limb -> (its first joint's origin in the trunk frame [m], the furthest its ball can get
    from there [m]), from the URDF and the hooks. An upper bound, never an underestimate:
        the first joint turns about an axis through that origin, so it does not change the
        distance; down the chain (joint origins, then the ball), the distance between every other
        point depends on the one joint between them only; its largest over that joint's range
        (sampled, plus how far the point can move between two samples), summed, bounds the reach.
    Arms 0.153 m and legs 0.238 m, against 0.150 m and 0.219 m over 20000 random poses."""
    urdf = read_urdf(urdf_path())
    by_child = {j.child: j for j in urdf.joints}
    out = {}
    for limb, hook in HOOKS.items():
        chain, link = [], hook.link
        while link != ROOT_LINK:
            chain.append(by_child[link])
            link = chain[-1].parent
        chain.reverse()
        # hops[m]: point m+1 from point m, in the frame of the link point m is the origin of
        hops = [j.xyz for j in chain[1:]] + [ball_centre(hook)]
        reach = 0.0
        for m in range(0, len(hops), 2):
            if m + 1 == len(hops):
                reach += float(np.linalg.norm(hops[m]))
                continue
            joint = chain[m + 1]  # between points m+1 and m+2: the only angle their distance has
            angles = np.linspace(joint.lower, joint.upper, 2001) if joint.revolute else [0.0]
            step = (joint.upper - joint.lower) / 2000 if joint.revolute else 0.0
            far = max(
                np.linalg.norm(hops[m] + joint.rotation @ _rodrigues(joint.axis, a) @ hops[m + 1])
                for a in angles
            )
            reach += float(far + np.linalg.norm(hops[m + 1]) * step / 2)
        out[limb] = (chain[0].xyz, reach)
    return out


def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def _point_jacobians(urdf: Urdf, names: tuple[str, ...]) -> dict[str, ca.Function]:
    """Per link: (q0, dq, p) -> (x, dx/ddq) for a point p fixed on that link, x in the wall frame
    at q = integrate(q0, dq). The Jacobian is w.r.t. the optimisation variable dq itself."""
    q0, dq, point = ca.SX.sym("q0", NQ), ca.SX.sym("dq", NV), ca.SX.sym("p", 3)
    poses = link_poses(urdf, integrate_expression(q0, dq))
    out = {}
    for i, name in enumerate(names):  # (CasADi names cannot hold Microban's double underscores)
        x = poses[name][0] + poses[name][1] @ point
        out[name] = ca.Function(f"point_on_link_{i}", [q0, dq, point], [x, ca.jacobian(x, dq)])
    return out
