"""Four-mesh stereo fitting with image-estimated support and occlusion geometry.

Input is RGB and declared camera/wall calibration. No sampler seed, true volume
or labelled hold geometry is read. The former PRNG stage is replaced by bounded
support-plane and local-pose recovery against the observed images.
"""

import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

from .hold_fit import (
    fit_plane,
    fit_translation,
    observations,
    plane_initial,
    refine_inclined,
    stereo_photo_error,
)
from .hold_geometry import SceneGeometry, estimate_volumes, refine_volume_faces
from .rcr_contract import TYPES
from .rcr_exact_fit import ExactRenderer
from .rcr_methods import _calibration, _project, _rotation
from .rcr_native_geometry import FACES, paint_masks, wall_point

PAINTS = np.array([[242, 85, 78], [89, 199, 123], [99, 137, 237]], np.uint8)


def proposals(rgb):
    """Keep colored components without the obsolete 100 mm clearance merge."""
    result = []
    values = rgb.astype(float)
    for group, broad in enumerate(paint_masks(rgb)):
        core = broad & (
            (values[..., group] >= 250) | (np.max(np.abs(values - PAINTS[group]), axis=-1) < 12)
        )
        mask = broad & (cv2.dilate(core.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0)
        count, labels, stats, centers = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        for index in range(1, count):
            x, y, width, height, area = stats[index]
            if not (4 <= area <= 5000 and 2 <= width <= 128 and 2 <= height <= 96):
                continue
            component = labels == index
            if (component & core).sum() < max(2, area * 0.08):
                continue
            result.append(
                dict(
                    mask=component,
                    center=centers[index],
                    area=int(area),
                    hue_group=group,
                    bbox=[int(x), int(y), int(width), int(height)],
                    color=PAINTS[group],
                )
            )
    return sorted(result, key=lambda p: (p["hue_group"], p["bbox"][1], p["bbox"][0]))


def match_proposals(left, right, k, t, geometry):
    if not left or not right:
        return []
    relative = t[1] @ np.linalg.inv(t[0])
    x, y, z = relative[:3, 3]
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    fundamental = np.linalg.inv(k[1]).T @ skew @ relative[:3, :3] @ np.linalg.inv(k[0])
    cost = np.full((len(left), len(right)), 1e6)
    for i, a in enumerate(left):
        line = fundamental @ np.r_[a["center"], 1.0]
        for j, b in enumerate(right):
            if a["hue_group"] != b["hue_group"]:
                continue
            epipolar = abs(line @ np.r_[b["center"], 1.0]) / max(np.linalg.norm(line[:2]), 1e-9)
            wall = wall_point(a["center"] + 0.5, k[0], t[0])
            uv, _ = _project(wall, k[1], t[1])
            if epipolar <= 2.5 and np.linalg.norm(b["center"] + 0.5 - uv) < 16:
                cost[i, j] = epipolar + 0.5 * abs(np.log(a["area"] / b["area"]))
    rows, columns = linear_sum_assignment(cost)
    return [(int(i), int(j)) for i, j in zip(rows, columns, strict=True) if cost[i, j] < 1e5]


def support_planes(geometry, volumes):
    result = [(np.array([0.0, 0.0, 1.0]), np.zeros(3), None, "wall")]
    for index, volume in enumerate(volumes):
        vertices = geometry.vertices(volume["parameters"])
        for face, indices in enumerate(FACES):
            triangle = vertices[list(indices)]
            a, b, c = triangle
            normal = np.cross(b - a, c - a)
            normal /= np.linalg.norm(normal)
            if normal[2] < 0:
                normal *= -1
            if normal[2] > 0.25:
                result.append((normal, a, triangle, f"volume_{index}_face_{face}"))
    return result


def on_face(point, triangle, margin=0.08):
    if triangle is None:
        return True
    a, b, c = triangle
    coordinates = np.linalg.lstsq(np.stack([b - a, c - a], axis=1), point - a, rcond=None)[0]
    return bool(min(coordinates) >= -margin and coordinates.sum() <= 1 + margin)


def infer(images, calibration):
    started = time.perf_counter()
    geometry = SceneGeometry.from_calibration(calibration)
    k, t = _calibration(calibration)
    parts = [proposals(image) for image in images]
    volumes = refine_volume_faces(
        images, calibration, geometry, estimate_volumes(images, calibration, geometry)
    )
    planes = support_planes(geometry, volumes)
    pairs = match_proposals(*parts, k, t, geometry)
    maximum_depth = max([v["parameters"][3] for v in volumes] + [0.05]) + 0.08
    renders = {}
    holds, diagnostics = {}, []
    try:
        for kind in sorted(TYPES):
            renders[kind] = ExactRenderer(
                kind=kind,
                wall=True,
                wall_size=(geometry.width, geometry.height),
                volume_height=geometry.volume_height,
                volumes=[v["parameters"] for v in volumes],
            )
        for left, right in pairs:
            selected = [parts[0][left], parts[1][right]]
            obs = observations(
                images, selected, [np.zeros(image.shape[:2], bool) for image in images]
            )
            candidates = []
            for kind, render in renders.items():
                for normal, anchor, triangle, name in planes:
                    for yaw in [0.0, -np.pi / 4, np.pi / 4]:
                        rotation = _rotation(normal, yaw)
                        point = plane_initial(
                            selected, k, t, rotation, normal, anchor, render=render
                        )
                        if not on_face(point, triangle, margin=0.18):
                            continue
                        point, rotation, loss = fit_plane(
                            point, rotation, normal, anchor, obs, k, t, render=render, rounds=2
                        )
                        if on_face(point, triangle):
                            candidates.append((loss, point, rotation, kind, name))
            candidates.sort(key=lambda c: c[0])
            if not candidates:
                continue
            # Retain the support-constrained candidate if a free update worsens the image fit.
            for _loss, point, rotation, kind, name in candidates[:2]:
                if name == "wall":
                    continue
                render = renders[kind]
                p, r, score = fit_translation(
                    point, rotation, obs, k, t, render=render, maximum_depth=maximum_depth, rounds=2
                )
                candidates.append((score, p, r, kind, name + "_translation"))
                p, r, score = refine_inclined(
                    p, r, obs, k, t, render=render, maximum_depth=maximum_depth
                )
                candidates.append((score, p, r, kind, name + "_pose"))
            loss, point, rotation, kind, name = min(candidates, key=lambda c: c[0])
            photo_diagnostics = None
            if kind == "fingerpocket_hold":
                normal = rotation[:, 2]
                alternatives = []
                for yaw in [0.0, -np.pi / 4, np.pi / 4]:
                    candidate = _rotation(normal, yaw)
                    p, r, score = fit_plane(
                        point, candidate, normal, point, obs, k, t, render=renders[kind], rounds=2
                    )
                    photo = stereo_photo_error(p, r, images, obs, k, t, render=renders[kind])
                    alternatives.append((score, photo, p, r))
                minimum = min(candidate[0] for candidate in alternatives)
                supported = sorted(
                    [
                        candidate
                        for candidate in alternatives
                        if candidate[0] <= minimum + 0.02 and candidate[1] is not None
                    ],
                    key=lambda candidate: candidate[1],
                )
                changed = len(supported) > 1 and supported[1][1] - supported[0][1] > 0.01
                if changed:
                    loss, _, point, rotation = supported[0]
                    name += "_stereo_RGB"
                photo_diagnostics = dict(
                    applied=changed,
                    candidates=[dict(silhouette=float(c[0]), RGB_error=c[1]) for c in alternatives],
                    note="Image diagnostic; not calibrated confidence",
                )
            accepted = bool(
                loss < 0.6
                and abs(point[0]) <= geometry.width / 2 + 0.02
                and -0.02 <= point[1] <= geometry.height + 0.02
                and -0.002 <= point[2] <= maximum_depth
            )
            diagnostics.append(
                dict(
                    left_proposal=left,
                    right_proposal=right,
                    kind=kind,
                    family=name,
                    loss=float(loss),
                    accepted=accepted,
                    stereo_RGB=photo_diagnostics,
                )
            )
            if accepted:
                angles = Rotation.from_matrix(rotation).as_euler("XYZ")
                holds[f"hold_{len(holds) + 1}"] = dict(
                    type=kind,
                    color="#" + "".join(f"{c:02X}" for c in selected[0]["color"]),
                    position=dict(zip("xyz", map(float, point), strict=True)),
                    orientation=dict(
                        zip(["roll", "pitch", "yaw"], map(float, angles), strict=True)
                    ),
                )
        return holds, dict(
            method="four_type_scene_CAD",
            runtime_s=time.perf_counter() - started,
            left_proposals=len(parts[0]),
            right_proposals=len(parts[1]),
            objects=diagnostics,
            volumes=[
                dict(
                    parameters=v["parameters"].tolist(),
                    loss=float(v["loss"]),
                    base_height=geometry.volume_height,
                )
                for v in volumes
            ],
            recovery=dict(
                method="RGB support-plane and local-pose refinement",
                seed_search=False,
                source_exact=False,
            ),
            inputs=[
                "stereo RGB",
                "declared intrinsics/extrinsics and wall size",
                "four public meshes",
            ],
        )
    finally:
        for render in renders.values():
            render.close()
