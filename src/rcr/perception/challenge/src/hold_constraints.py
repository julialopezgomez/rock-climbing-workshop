"""Source contact and convex-hull consistency for image-estimated scenes."""

import copy
from functools import lru_cache
from itertools import combinations

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from .hold_geometry import SceneGeometry
from .rcr_methods import _rotation, normalized_mesh
from .rcr_native_geometry import FACES

HULL_GRID = 0.001
HULL_TOLERANCE = np.sqrt(3.0) * HULL_GRID / 2
CONTACT_CLEARANCE = 2 * HULL_TOLERANCE


def shape(points, triangles, tolerance):
    normals, edges = [], []
    for a, b, c in triangles:
        normal = np.cross(points[b] - points[a], points[c] - points[a])
        length = np.linalg.norm(normal)
        if length > 1e-9:
            normals.append(normal / length)
        for i, j in combinations([a, b, c], 2):
            edge = points[j] - points[i]
            length = np.linalg.norm(edge)
            if length > 1e-9:
                edges.append(edge / length)
    return dict(
        points=points,
        normals=np.unique(np.round(normals, 7), axis=0),
        edges=np.unique(np.round(edges, 7), axis=0),
        tolerance=tolerance,
    )


@lru_cache(maxsize=4)
def local_hull(kind):
    points = np.unique(
        np.floor(normalized_mesh(kind).vertices / HULL_GRID + 0.5) * HULL_GRID, axis=0
    )
    hull = ConvexHull(points)
    value = shape(points, hull.simplices, float(HULL_TOLERANCE))
    value["points"] = points[hull.vertices]
    return value


def transformed_hull(kind, rotation, point):
    local = local_hull(kind)
    return dict(
        points=local["points"] @ rotation.T + point,
        normals=local["normals"] @ rotation.T,
        edges=local["edges"] @ rotation.T,
        tolerance=local["tolerance"],
    )


def collision_axes(first, second):
    tolerance = first["tolerance"] + second["tolerance"] + 1e-8
    a, b = first["points"], second["points"]
    if np.any(a.max(0) <= b.min(0) - tolerance) or np.any(b.max(0) <= a.min(0) - tolerance):
        return None
    face_axes = np.concatenate([first["normals"], second["normals"]])
    pa, pb = a @ face_axes.T, b @ face_axes.T
    if np.any(pa.max(0) < pb.min(0) - tolerance) or np.any(pb.max(0) < pa.min(0) - tolerance):
        return None
    crossing = np.cross(first["edges"][:, None], second["edges"][None]).reshape(-1, 3)
    norms = np.linalg.norm(crossing, axis=1)
    crossing = crossing[norms > 1e-6] / norms[norms > 1e-6, None]
    ca, cb = a @ crossing.T, b @ crossing.T
    if np.any(ca.max(0) < cb.min(0) - tolerance) or np.any(cb.max(0) < ca.min(0) - tolerance):
        return None
    axes = np.concatenate([face_axes, crossing])
    pa, pb = np.concatenate([pa, ca], axis=1), np.concatenate([pb, cb], axis=1)
    return axes, pa.min(0), pa.max(0), pb.min(0), pb.max(0), tolerance


def tangent_correction(collision, normal):
    axes, amin, amax, bmin, bmax, tolerance = collision
    tangent = axes - (axes @ normal)[:, None] * normal
    lengths = np.linalg.norm(tangent, axis=1)
    valid = lengths > 1e-6
    direction = tangent[valid] / lengths[valid, None]
    gain = np.sum(axes[valid] * direction, axis=1)
    right = (bmax[valid] - amin[valid] + tolerance + 1e-5) / gain
    left = (amax[valid] - bmin[valid] + tolerance + 1e-5) / gain
    options = np.concatenate([direction * right[:, None], -direction * left[:, None]])
    return options[np.argmin(np.linalg.norm(options, axis=1))]


def recover_scene(holds, diagnostics, calibration):
    """Recover a source-consistent scene using only fitted poses and volumes.

    Volume normals/backplate spacing follow the public geometry. Residual hull
    conflicts move the weaker image hypothesis tangentially to its support.
    Unresolved hypotheses are rejected explicitly. This is not seed recovery.
    """
    recovered = copy.deepcopy(holds)
    geometry = SceneGeometry.from_calibration(calibration)
    objects = [record for record in diagnostics["objects"] if record["accepted"]]
    changes, rejected = [], []
    volumes = [geometry.vertices(v["parameters"]) for v in diagnostics["volumes"]]
    obstacles = [shape(v, [(0, 2, 1), *FACES], 0.0) for v in volumes]
    fitted = []
    for (identifier, hold), record in zip(recovered.items(), objects, strict=True):
        point = np.array([hold["position"][key] for key in "xyz"])
        original_point = point.copy()
        rotation = Rotation.from_euler(
            "XYZ", [hold["orientation"][key] for key in ["roll", "pitch", "yaw"]]
        ).as_matrix()
        original_rotation = rotation.copy()
        normal = np.array([0.0, 0.0, 1.0])
        family = record["family"].split("_")
        if family[0] == "volume":
            volume, face = int(family[1]), int(family[3])
            a, b, c = volumes[volume][list(FACES[face])]
            normal = np.cross(b - a, c - a)
            normal /= np.linalg.norm(normal)
            if normal[2] < 0:
                normal *= -1
            aligned = _rotation(normal, 0.0)
            local = aligned.T @ rotation
            yaw = float(np.arctan2(local[1, 0], local[0, 0]))
            yaw = min(
                [0.0, -np.pi / 4, np.pi / 4],
                key=lambda angle: abs(np.arctan2(np.sin(angle - yaw), np.cos(angle - yaw))),
            )
            rotation = _rotation(normal, yaw)
            center = normalized_mesh(hold["type"]).vertices.mean(0)
            point += (original_rotation - rotation) @ center
            point += normal * (CONTACT_CLEARANCE - float((point - a) @ normal))
        fitted.append(
            dict(
                id=identifier,
                point=point,
                rotation=rotation,
                normal=normal,
                original=original_point,
                original_rotation=original_rotation,
                loss=record["loss"],
            )
        )
    placed = []
    for item in sorted(fitted, key=lambda value: value["loss"]):
        identifier = item["id"]
        hold = recovered[identifier]
        if len(placed) >= 50:
            recovered.pop(identifier)
            rejected.append(dict(id=identifier, reason="Published 50-hold output limit"))
            continue
        clear = False
        for _ in range(8):
            clear = True
            for other in [*obstacles, *placed]:
                candidate = transformed_hull(hold["type"], item["rotation"], item["point"])
                collision = collision_axes(candidate, other)
                if collision is not None:
                    item["point"] += tangent_correction(collision, item["normal"])
                    clear = False
            if clear:
                break
        if not clear:
            candidate = transformed_hull(hold["type"], item["rotation"], item["point"])
            clear = all(collision_axes(candidate, other) is None for other in [*obstacles, *placed])
        if not clear:
            recovered.pop(identifier)
            rejected.append(dict(id=identifier, reason="Unresolved fitted-hull conflicts"))
            continue
        placed.append(transformed_hull(hold["type"], item["rotation"], item["point"]))
        hold["position"] = dict(zip("xyz", map(float, item["point"]), strict=True))
        hold["orientation"] = dict(
            zip(
                ["roll", "pitch", "yaw"],
                map(float, Rotation.from_matrix(item["rotation"]).as_euler("XYZ")),
                strict=True,
            )
        )
        distance = float(np.linalg.norm(item["point"] - item["original"]))
        angle = float(
            Rotation.from_matrix(item["rotation"] @ item["original_rotation"].T).magnitude()
        )
        if distance > 1e-10 or angle > 1e-10:
            changes.append(
                dict(
                    id=identifier,
                    displacement_m=distance,
                    rotation_change_rad=angle,
                    original_position=item["original"].tolist(),
                )
            )
    return recovered, dict(
        method="source support and hull consistency",
        seed_search=False,
        source_exact=False,
        changes=changes,
        rejected=rejected,
        clearance_m=float(CONTACT_CLEARANCE),
    )
