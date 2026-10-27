"""Exact clearances of the robot from the holds, the wall and itself, with their gradients (coal).

Obstacles:
    each hold   the organisers' mesh (data/holds/hold.stl [Rules v1.1 Figure 3]), placed as the
                sim does: at Hold.position + HOLD_MESH_OFFSET, not rotated (its frame is the wall's)
    the wall    the half-space y >= 0 (its face is y = 0; the robot is at y < 0)
Robot: the convex hull of each of Microban's visual meshes [Microban MJCF], so what the sim draws
is inside what is checked (its own collision boxes are smaller than the thighs and shins), and our
hooks (a capsule per finger segment and a sphere for the ball), each fixed on a link. The robot
is also kept clear of itself: every pair of its shapes in self_pairs().

Signed distance to a hold, from two forms of the same mesh:
    exact   the mesh itself (a BVH): exact outside, ~0 when touching, but no depth inside;
    pieces  the mesh cut into convex pieces (CoACD, cached): a depth and a way out when inside,
            but they swell into the socket by up to ~0.4 mm, so they are not used outside.
The exact distance is used unless the robot is inside the hold, where the pieces' is.

Gradient: for the nearest point x on the robot, d(distance)/d(dq) = u^T dx/ddq, with u the unit
direction that moves x away from the obstacle (minus coal's normal) and dx/ddq from the
robot model; between two robot shapes, the nearest point on each moves (in opposite directions).
Wall frame and q as in robot_model.py.
"""

from __future__ import annotations

import hashlib
import multiprocessing
import os
import struct
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import coal
import numpy as np

from rcr.optimiser import robot_model as rm
from rcr.robot import hooks
from rcr.types import Hold

REPO = Path(__file__).resolve().parents[3]
HOLD_STL = REPO / "data" / "holds" / "hold.stl"  # the organisers' hold, in mm [Rules v1.1 §4]
HOLD_STL_SCALE = 0.001  # [m/mm]
HOLD_MESH_OFFSET = (0.0, -0.026, -0.014)  # [m] socket centre (the mesh origin) from Hold.position
SOCKET_RADIUS = 0.0127  # [m] the hold's spherical socket, measured on hold.stl (rcr.sim CupHold)
# The convex pieces, keyed by the STL's sha256, kept with the mesh in the repository: CoACD gives
# slightly different pieces on another machine, and so slightly different solutions [Ours].
PIECES_CACHE = REPO / "data" / "holds"
COACD_THRESHOLD = 0.02  # CoACD concavity: 106 pieces, swelling < ~0.4 mm into the socket [Ours]
TOUCHING = 5e-5  # [m] the exact mesh reports a ball resting in its socket as -4e-6 m
# [m] how far a hook may press into the hold its ball rests in for a whole interval [Ours]: that is
# the contact itself (and the hooks are placeholders). At a touch the exact mesh gives no direction
# out, so with -TOUCHING alone a loaded foot's finger at 0.13 mm in could not be moved out, and the
# move failed (example plan, move 9).
RESTING_PLAY = 5e-4
SWELL = 1e-3  # [m] how far the pieces may swell past the mesh: deeper means truly inside
# The deepest point inside the hold's material: 7.2 mm from its surface, measured on hold.stl
# (sha256 8c8df8c6...) by sampling, +1 mm for the sampling. A shape of half-thickness r lying
# wholly inside is at most HOLD_MAX_DEPTH - r from the surface, so beyond that it is outside.
HOLD_MAX_DEPTH = 0.0082  # [m]


# --- Geometry -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class RobotShape:
    """A convex shape fixed on a robot link."""

    name: str
    link: str
    geometry: coal.CollisionGeometry
    offset: np.ndarray  # [m] centre, link frame
    rotation: np.ndarray  # 3x3, link frame
    radius: float  # [m] bounding sphere about the centre
    inner: float  # [m] half its smallest thickness: it cannot hide in less material than 2x this
    limb: str | None  # the hook's limb, or None for Microban's own shapes


@dataclass(frozen=True)
class HoldGeometry:
    """The hold mesh in its own frame (origin at the socket centre, metres)."""

    mesh: coal.BVHModelOBBRSS
    pieces: list
    piece_centres: np.ndarray  # P x 3, bounding sphere per piece
    piece_radii: np.ndarray
    centre: np.ndarray  # bounding sphere of the whole hold
    radius: float
    planes: np.ndarray  # F x 4: (outward normal, offset) of every face of every piece
    piece_of_plane: np.ndarray  # F: which piece each face belongs to (sorted)
    # The mesh's convex hull: a distance to it is a lower bound on the distance to the hold,
    # ~4 us against up to ~1.5 ms for the mesh far away (coal's BVH search has no upper bound)
    hull: coal.ConvexBase

    def contains(self, point: np.ndarray) -> bool:
        """Is `point` (hold frame) inside the material? Inside one of the convex pieces."""
        excess = self.planes[:, :3] @ point - self.planes[:, 3]
        starts = np.flatnonzero(np.r_[True, np.diff(self.piece_of_plane) != 0])
        return bool((np.maximum.reduceat(excess, starts) <= 0.0).any())


def _z_to(direction: np.ndarray) -> np.ndarray:
    """A rotation taking the local z axis to `direction` (unit)."""
    z = direction / np.linalg.norm(direction)
    x = np.cross([0.0, 1.0, 0.0] if abs(z[1]) < 0.9 else [1.0, 0.0, 0.0], z)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


@lru_cache(maxsize=1)
def robot_shapes() -> tuple[RobotShape, ...]:
    """The convex hulls of Microban's visual meshes, plus our hooks (hooks.HOOKS)."""
    shapes = [
        _hull_shape(f"{s.link}_{i}", s) for i, s in enumerate(rm.read_visual_shapes(rm.mjcf_path()))
    ]
    for limb, hook in hooks.HOOKS.items():
        point = np.array(hook.start, dtype=float)
        for k, (direction, length) in enumerate(hook.segments):
            end = point + np.array(direction, dtype=float) * length
            shapes.append(
                RobotShape(
                    f"{limb}_finger{k}",
                    hook.link,
                    coal.Capsule(hooks.FINGER_RADIUS, length),
                    (point + end) / 2,
                    _z_to(np.array(direction, dtype=float)),
                    length / 2 + hooks.FINGER_RADIUS,
                    hooks.FINGER_RADIUS,
                    limb,
                )
            )
            point = end
        shapes.append(
            RobotShape(
                f"{limb}_ball",
                hook.link,
                coal.Sphere(hooks.BALL_RADIUS),
                hooks.ball_centre(hook),
                np.eye(3),
                hooks.BALL_RADIUS,
                hooks.BALL_RADIUS,
                limb,
            )
        )
    return tuple(shapes)


def _hull_shape(name: str, visual: rm.VisualShape) -> RobotShape:
    """A visual mesh's convex hull, centred on its vertices' bounding box, in its link's frame.
    inner is the largest sphere about that centre inside the hull: no thicker than the hull."""
    vertices, triangles = read_stl(visual.mesh, 1.0)
    points = vertices @ visual.rotation.T + visual.offset  # link frame
    centre = (points.max(0) + points.min(0)) / 2
    model = _bvh(points - centre, triangles)
    model.buildConvexHull(True, "Qt")
    hull = model.convex
    offsets = np.asarray(hull.offsets()).ravel()
    radius = float(np.linalg.norm(points - centre, axis=1).max())
    inner = float(max(-offsets.max(), 0.0))  # coal's planes: n . x + offset <= 0 inside
    return RobotShape(name, visual.link, hull, centre, np.eye(3), radius, inner, None)


def read_hold_mesh(path: Path = HOLD_STL) -> tuple[np.ndarray, np.ndarray]:
    """(vertices V x 3 [m], triangles T x 3) of the hold STL, duplicate vertices merged."""
    return read_stl(path, HOLD_STL_SCALE)


def read_stl(path: Path, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """(vertices V x 3 [m], triangles T x 3) of a binary STL in units of `scale` metres,
    duplicate vertices merged."""
    raw = path.read_bytes()
    (n,) = struct.unpack("<I", raw[80:84])
    if len(raw) != 84 + 50 * n:
        raise ValueError(f"{path} is not a binary STL")
    record = np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    corners = np.frombuffer(raw, dtype=record, count=n, offset=84)["v"].reshape(-1, 3)
    keys = np.round(corners.astype(float) * scale * 1e7).astype(np.int64)  # merge to 0.1 µm
    unique, index = np.unique(keys, axis=0, return_inverse=True)
    return unique * 1e-7, index.reshape(-1, 3)


def convex_pieces(vertices: np.ndarray, triangles: np.ndarray) -> list[tuple]:
    """The hold cut into convex pieces (CoACD), as (vertices, triangles), stored in data/holds by
    the STL's sha256 and kept in git, so every machine uses the same pieces. Computed (~20 s) only
    if the organisers change the mesh: then commit the new file, for the same reason."""
    digest = hashlib.sha256(HOLD_STL.read_bytes()).hexdigest()[:16]
    cache = PIECES_CACHE / f"hold_pieces_{digest}_{COACD_THRESHOLD}.npz"
    if cache.exists():
        stored = np.load(cache)
        return [(stored[f"v{i}"], stored[f"t{i}"]) for i in range(len(stored.files) // 2)]
    import coacd

    coacd.set_log_level("error")
    parts = coacd.run_coacd(coacd.Mesh(vertices, triangles), threshold=COACD_THRESHOLD, seed=0)
    pieces = [(np.asarray(v, dtype=float), np.asarray(f, dtype=np.int64)) for v, f in parts]
    cache.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"v{i}": v for i, (v, _) in enumerate(pieces)}
    arrays |= {f"t{i}": f for i, (_, f) in enumerate(pieces)}
    np.savez(cache, **arrays)
    return pieces


def _bvh(vertices: np.ndarray, triangles: np.ndarray) -> coal.BVHModelOBBRSS:
    model = coal.BVHModelOBBRSS()
    model.beginModel(len(triangles), len(vertices))
    model.addVertices(np.asarray(vertices, dtype=float))
    model.addTriangles(np.asarray(triangles, dtype=np.int64))
    model.endModel()
    return model


def _convex(vertices: np.ndarray, triangles: np.ndarray):
    """A convex piece (its triangulated surface) as a coal convex shape."""
    model = _bvh(vertices, triangles)
    model.buildConvexRepresentation(False)
    return model.convex


@lru_cache(maxsize=1)
def hold_geometry() -> HoldGeometry:
    vertices, triangles = read_hold_mesh()
    pieces = convex_pieces(vertices, triangles)
    centres = np.array([(v.max(0) + v.min(0)) / 2 for v, _ in pieces])
    radii = np.array(
        [np.linalg.norm(v - c, axis=1).max() for (v, _), c in zip(pieces, centres, strict=True)]
    )
    centre = (vertices.max(0) + vertices.min(0)) / 2
    planes = [_outward_planes(v, f) for v, f in pieces]
    hull = _bvh(vertices, triangles)
    hull.buildConvexHull(True, "Qt")
    return HoldGeometry(
        mesh=_bvh(vertices, triangles),
        pieces=[_convex(v, f) for v, f in pieces],
        piece_centres=centres,
        piece_radii=radii,
        centre=centre,
        radius=float(np.linalg.norm(vertices - centre, axis=1).max()),
        planes=np.vstack(planes),
        piece_of_plane=np.concatenate([np.full(len(p), i) for i, p in enumerate(planes)]),
        hull=hull.convex,
    )


def _outward_planes(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """(n, d) per face of a convex piece, n pointing out: inside is n.x <= d on every face."""
    a, b, c = (vertices[triangles[:, i]] for i in range(3))
    normals = np.cross(b - a, c - a)
    keep = np.linalg.norm(normals, axis=1) > 1e-14
    normals = normals[keep] / np.linalg.norm(normals[keep], axis=1, keepdims=True)
    offsets = np.einsum("ij,ij->i", normals, a[keep])
    flip = normals @ vertices.mean(axis=0) - offsets > 0  # the centroid must be inside
    normals[flip], offsets[flip] = -normals[flip], -offsets[flip]
    return np.column_stack([normals, offsets])


def hold_origin(hold: Hold) -> np.ndarray:
    """[m] where the hold mesh's origin (the socket centre) is, in the wall frame."""
    return np.array(hold.position) + np.array(HOLD_MESH_OFFSET)


WALL = coal.Halfspace(np.array([0.0, -1.0, 0.0]), 0.0)  # {x : -y <= 0}, i.e. the wall, y >= 0


# --- Distances ----------------------------------------------------------------------------------


@dataclass
class Distance:
    value: float  # [m] signed: negative inside
    point: np.ndarray  # nearest point on the robot shape, wall frame
    away: np.ndarray  # unit: moving `point` along it increases the distance
    other: np.ndarray  # nearest point on the other shape, wall frame


def _query(shape: coal.CollisionGeometry, pose: coal.Transform3s, other, other_pose) -> Distance:
    result = coal.DistanceResult()
    value = coal.distance(shape, pose, other, other_pose, coal.DistanceRequest(), result)
    point1, point2 = result.getNearestPoint1(), result.getNearestPoint2()
    away = -np.array(result.normal, dtype=float)  # coal's normal points from shape to other
    if not np.isfinite(away).all() or np.linalg.norm(away) < 0.5:
        gap = point1 - point2
        away = gap / np.linalg.norm(gap) if np.linalg.norm(gap) > 1e-12 else np.array([0, -1.0, 0])
    return Distance(
        float(value), np.array(point1, dtype=float), away, np.array(point2, dtype=float)
    )


def hold_distance(
    shape: RobotShape, pose: coal.Transform3s, hold: HoldGeometry, origin: np.ndarray
) -> Distance:
    """Signed distance from a robot shape (at `pose`) to one hold (mesh origin at `origin`).

    Not touching the surface, the shape is wholly outside (the exact mesh distance) or wholly
    inside, which its centre tells. Touching or crossing it, the pieces give the depth, unless
    they are within their own swell, where the mesh's (small, negative) value stands.
    """
    hold_pose = coal.Transform3s(np.eye(3), origin)
    exact = _query(shape.geometry, pose, hold.mesh, hold_pose)
    centre = pose.getTranslation() - origin
    if exact.value > TOUCHING:
        if exact.value > HOLD_MAX_DEPTH - shape.inner or not hold.contains(centre):
            return exact  # wholly outside
    gap = np.linalg.norm(hold.piece_centres - centre, axis=1) - hold.piece_radii - shape.radius
    near = np.nonzero(gap <= 0.0)[0]
    if len(near) == 0:
        return exact
    inside = min(
        (_query(shape.geometry, pose, hold.pieces[i], hold_pose) for i in near),
        key=lambda d: d.value,
    )
    if exact.value <= TOUCHING and inside.value >= -SWELL:
        return exact  # touching, or crossing by less than the pieces can tell
    return inside  # inside the hold: the pieces give the depth and the way out


def _tiered_distance(
    shape: RobotShape,
    pose: coal.Transform3s,
    hold: HoldGeometry,
    origin: np.ndarray,
    required: float,
    clearance: Clearance,
) -> Distance | None:
    """hold_distance, or None if it is at least clearance.near past `required`, from the
    cheapest form that can tell (the exact mesh is slow far from the hold: ~1.5 ms at 50 mm):
        the hull    a lower bound: far from it, far from the hold (None);
        the pieces  they cover the mesh (to 3 um on its 30k surface points) and swell past it
                    by at most SWELL, so they understate the distance by less than EXACT_BAND:
                    beyond EXACT_BAND past `required` their distance stands;
        the mesh    within EXACT_BAND, hold_distance decides, exactly."""
    hold_pose = coal.Transform3s(np.eye(3), origin)
    if _query(shape.geometry, pose, hold.hull, hold_pose).value - required >= clearance.near:
        return None
    centre = pose.getTranslation() - origin
    gap = np.linalg.norm(hold.piece_centres - centre, axis=1) - hold.piece_radii - shape.radius
    reach = np.nonzero(gap <= required + clearance.near)[0]
    if len(reach) == 0:
        return None
    nearest = min(
        (_query(shape.geometry, pose, hold.pieces[i], hold_pose) for i in reach),
        key=lambda d: d.value,
    )
    if nearest.value - required >= clearance.near:
        return None
    if nearest.value - required >= EXACT_BAND:
        return nearest
    return hold_distance(shape, pose, hold, origin)


EXACT_BAND = 1e-3  # [m] > SWELL: closer than this to its requirement, a hold distance is exact


def wall_distance(shape: RobotShape, pose: coal.Transform3s) -> Distance:
    return _query(shape.geometry, pose, WALL, coal.Transform3s())


@lru_cache(maxsize=1)
def self_pairs() -> np.ndarray:
    """K x 2 indices into robot_shapes(): the pairs of robot shapes kept apart [Ours]. Left out:
    shapes on one link or on the two links of a joint (they touch by design), and shapes on links
    two joints apart that overlap at the zero pose: the hip and the femur, the tibia and the foot,
    joined by a small block about which they turn, overlap in every pose (sampled: 300 of 300)."""
    shapes, model = robot_shapes(), rm.load_robot_model()
    parent = {j.child: j.parent for j in rm.read_urdf(rm.urdf_path()).joints}
    zero = np.zeros(rm.NQ)
    zero[3] = 1.0
    _, centres, axes = _shape_poses(model, zero, np.zeros(rm.NV))
    pairs = []
    for i in range(len(shapes)):
        for j in range(i + 1, len(shapes)):
            a, b = shapes[i].link, shapes[j].link
            if a == b or parent.get(a) == b or parent.get(b) == a:
                continue
            if parent.get(parent.get(a)) == b or parent.get(parent.get(b)) == a:
                pose_i = coal.Transform3s(axes[i], centres[i])
                pose_j = coal.Transform3s(axes[j], centres[j])
                if _query(shapes[i].geometry, pose_i, shapes[j].geometry, pose_j).value < 0:
                    continue
            pairs.append((i, j))
    return np.array(pairs, dtype=int)


# --- Along a trajectory -------------------------------------------------------------------------


@dataclass(frozen=True)
class Clearance:
    """How clear of the holds and the wall a motion must stay [Ours]."""

    margin: float = 0.002  # [m] every robot part from every hold and the wall, except as below
    step: float = 0.002  # [m] the most any robot point moves between two checks (<= margin)
    near: float = 0.02  # [m] checks within this of their requirement are returned
    # The optimiser aims this much past each requirement when it linearises, so it lands with a
    # little reserve instead of on the boundary, where the measurement is least precise. Only the
    # requirement itself decides whether a motion is clear.
    aim: float = 0.0005  # [m]


@dataclass
class Check:
    """One robot shape against one obstacle at one moment, linearised in the samples' dq."""

    interval: int  # between samples interval and interval + 1
    tau: float  # how far along that interval, 0..1 (1 only for the last sample)
    value: float  # [m] signed distance
    required: float  # [m] the least it may be
    grad_a: np.ndarray  # d value / d dq[interval]  (NV)
    grad_b: np.ndarray  # d value / d dq[interval + 1]  (NV; zero without a next sample)
    label: str  # e.g. "right_foot_finger1 / hold R1"

    @property
    def gap(self) -> float:
        """How much clearance is left: negative means the requirement is violated."""
        return self.value - self.required


def check_trajectory(
    model: rm.RobotModel,
    holds: dict[str, Hold],
    q0: np.ndarray,
    dq: np.ndarray,
    contacts: list[dict[str, str | None]],
    own: dict[str, set[str]],
    clearance: Clearance,
) -> list[Check]:
    """Every robot shape against every hold within reach, the wall and the other robot shapes
    (self_pairs), along the motion
    q(t) = integrate(q0, dq(t)), dq linear between the samples (the columns of dq).

    Checked often enough that no robot point moves more than clearance.step between two checks,
    so with clearance.margin >= step nothing passes through anything in between.
    contacts[k][limb]: the hold that limb's ball rests in at sample k (or None).
    own[limb]: the holds the limb's hook may touch but not enter (its old and new holds).
    Returns the checks with gap < clearance.near.
    """
    moments = _moments(model, q0, dq, clearance.step)
    workers = CHECK_WORKERS if len(moments) >= 4 * CHECK_WORKERS else 1
    if multiprocessing.current_process().daemon:
        workers = 1  # a daemon may not start processes: check serially
    if workers == 1:
        return _check_moments(model, holds, q0, dq, contacts, own, clearance, moments)
    chunks = [moments[w::workers] for w in range(workers)]  # interleaved: even work per worker
    jobs = [(holds, q0, dq, contacts, own, clearance, chunk) for chunk in chunks]
    out = [c for part in _pool().map(_check_chunk, jobs) for c in part]
    return sorted(out, key=lambda c: (c.interval, c.tau))  # stable: the serial order


# The exact check of a motion is split over this many worker processes (its moments are
# independent): ~5 s serial for a hand move on 65 hulls [Ours].
CHECK_WORKERS = max(1, min(8, (os.cpu_count() or 2) - 2))


@lru_cache(maxsize=1)
def _pool() -> ProcessPoolExecutor:
    return ProcessPoolExecutor(
        CHECK_WORKERS,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_exit_with_parent,
    )


def _exit_with_parent() -> None:
    """In each worker: exit when the process that started it is gone, however it ended (an idle
    pool worker would otherwise wait for work forever)."""
    parent = os.getppid()

    def watch() -> None:
        while os.getppid() == parent:
            time.sleep(0.5)
        os._exit(0)

    threading.Thread(target=watch, daemon=True).start()


def _check_chunk(job) -> list[Check]:
    """One worker's share: its own model, shapes and hold geometry, built once per process."""
    return _check_moments(rm.load_robot_model(), *job)


def _check_moments(model, holds, q0, dq, contacts, own, clearance, moments) -> list[Check]:
    """check_trajectory at the given (interval, tau) moments."""
    shapes, hold, pairs = robot_shapes(), hold_geometry(), self_pairs()
    gaps = _self_gaps(shapes, pairs)
    origins = {h: hold_origin(holds[h]) for h in holds}
    centres = {h: origins[h] + hold.centre for h in holds}
    samples = dq.shape[1]
    out = []
    for interval, tau in moments:
        a, b = interval, min(interval + 1, samples - 1)
        dq_t = (1 - tau) * dq[:, a] + tau * dq[:, b]
        frames, positions, rotations = _shape_poses(model, q0, dq_t)
        resting = _resting(contacts, a, b)
        for s, shape in enumerate(shapes):
            pose = coal.Transform3s(rotations[s], positions[s])
            found = []
            if positions[s][1] + shape.radius > -(clearance.margin + clearance.near):
                found.append(("wall", wall_distance(shape, pose), clearance.margin))
            for h, centre in centres.items():
                if (
                    np.linalg.norm(positions[s] - centre)
                    > shape.radius + hold.radius + clearance.near
                ):
                    continue
                if shape.limb is not None and resting.get(shape.limb) == h and "ball" in shape.name:
                    continue  # the contact itself: this ball rests in this socket
                on_hook = shape.limb or _hook_limb(shape.link)
                required = -TOUCHING if on_hook and h in own.get(on_hook, ()) else clearance.margin
                if on_hook and resting.get(on_hook) == h:
                    required = -RESTING_PLAY  # the hook resting on this hold: the contact itself
                distance = _tiered_distance(shape, pose, hold, origins[h], required, clearance)
                if distance is not None:
                    found.append((f"hold {h}", distance, required))
            for what, distance, required in found:
                if distance.value - required >= clearance.near:
                    continue
                grad = _gradient(model, q0, dq_t, frames, shape.link, distance.point, distance.away)
                out.append(_check(interval, tau, a, b, distance.value, required, grad, shape, what))
        for i, j in pairs[gaps(positions) < clearance.margin + clearance.near]:
            pose_i = coal.Transform3s(rotations[i], positions[i])
            distance = _query(
                shapes[i].geometry,
                pose_i,
                shapes[j].geometry,
                coal.Transform3s(rotations[j], positions[j]),
            )
            if distance.value - clearance.margin >= clearance.near:
                continue
            # The distance grows as the nearest point on i moves along `away`, and on j against it
            grad = _gradient(model, q0, dq_t, frames, shapes[i].link, distance.point, distance.away)
            grad -= _gradient(
                model, q0, dq_t, frames, shapes[j].link, distance.other, distance.away
            )
            what = f"{shapes[j].name} (self)"
            out.append(
                _check(interval, tau, a, b, distance.value, clearance.margin, grad, shapes[i], what)
            )
    return out


def _check(interval, tau, a, b, value, required, grad, shape, what) -> Check:
    """A Check at `tau` of `interval` (samples a..b), its gradient shared out to both samples."""
    return Check(
        interval=interval,
        tau=tau,
        value=value,
        required=required,
        grad_a=(1 - tau) * grad if b != a else grad,
        grad_b=tau * grad if b != a else np.zeros_like(grad),
        label=f"{shape.name} / {what}",
    )


def _self_gaps(shapes, pairs):
    """positions -> per pair, a lower bound on its distance (its bounding spheres' gap)."""
    radii = np.array([s.radius for s in shapes])
    i, j = pairs.T
    reach = radii[i] + radii[j]

    def gaps(positions) -> np.ndarray:
        centres = np.asarray(positions)
        return np.linalg.norm(centres[i] - centres[j], axis=1) - reach

    return gaps


def _hook_limb(link: str) -> str | None:
    """The limb whose hook is fixed on this link: its shapes may touch that hook's holds."""
    return next((limb for limb, hook in hooks.HOOKS.items() if hook.link == link), None)


def _resting(contacts, a: int, b: int) -> dict[str, str]:
    """limb -> the hold its ball rests in at both ends of the interval a..b (a == b: at a)."""
    return {limb: h for limb, h in contacts[a].items() if h and contacts[b].get(limb) == h}


def _moments(model, q0, dq, step: float) -> list[tuple[int, float]]:
    """(interval, tau) for every check: each interval cut so that no shape moves more than step."""
    samples = dq.shape[1]
    if samples == 1:
        return [(0, 0.0)]
    shapes = robot_shapes()
    poses = [_shape_poses(model, q0, dq[:, k])[1:] for k in range(samples)]
    moments = []
    for i in range(samples - 1):
        (pa, ra), (pb, rb) = poses[i], poses[i + 1]
        moved = max(
            np.linalg.norm(pb[s] - pa[s]) + np.linalg.norm(rb[s] - ra[s], 2) * shape.radius
            for s, shape in enumerate(shapes)
        )
        cuts = max(1, int(np.ceil(moved / step)))
        moments += [(i, j / cuts) for j in range(cuts)]
    moments.append((samples - 2, 1.0))
    return moments


def _shape_poses(model, q0, dq_t):
    """(link frames, world centres, world rotations) of every robot shape at integrate(q0, dq_t)."""
    q = np.array(model.integrate(q0, dq_t)).ravel()
    positions, rotations = (np.array(m) for m in model.link_frames(q))
    index = {name: i for i, name in enumerate(model.link_names)}
    centres, axes = [], []
    for shape in robot_shapes():
        i = index[shape.link]
        r_link = rotations[:, 3 * i : 3 * i + 3]
        centres.append(positions[:, i] + r_link @ shape.offset)
        axes.append(r_link @ shape.rotation)
    return (positions, rotations), centres, axes


def _gradient(model, q0, dq_t, frames, link: str, point, away) -> np.ndarray:
    """d(distance)/d(dq_t) from a nearest point fixed on `link`: its Jacobian along `away`."""
    positions, rotations = frames
    i = model.link_names.index(link)
    r_link = rotations[:, 3 * i : 3 * i + 3]
    local = r_link.T @ (point - positions[:, i])
    _, jacobian = model.point_jacobian[link](q0, dq_t, local)
    return away @ np.array(jacobian)
