"""RGB cup instances and multiview CAD hypotheses for a calibrated physical board."""

from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.optimize import linear_sum_assignment, minimize

from ._paths import simulator_asset
from .live_pipeline import PixelCentreCupRenderer
from .src.calibration import project_wall

PIXELS_PER_METRE = 1000


def plane_warp(camera, wall, plane_z=0.026):
    transform = np.asarray(camera["T_camera_wall"])
    matrix = np.asarray(camera["K"])
    homography = matrix @ np.column_stack(
        [transform[:3, 0], transform[:3, 1], transform[:3, 3] + plane_z * transform[:3, 2]]
    )
    metric = np.array(
        [
            [PIXELS_PER_METRE, 0, 0],
            [0, -PIXELS_PER_METRE, wall["wall_height_m"] * PIXELS_PER_METRE],
            [0, 0, 1],
        ]
    )
    return metric @ np.linalg.inv(homography)


def cup_instances(image, camera, wall, exclude_mask=None):
    """Split touching cups along the board's horizontal direction before fitting."""
    width = round(wall["wall_width_m"] * PIXELS_PER_METRE)
    height = round(wall["wall_height_m"] * PIXELS_PER_METRE)
    warp = plane_warp(camera, wall)
    rectified = cv2.warpPerspective(image, warp, (width, height))
    hsv = cv2.cvtColor(rectified, cv2.COLOR_RGB2HSV)
    excluded = (
        cv2.warpPerspective(
            exclude_mask.astype(np.uint8), warp, (width, height), flags=cv2.INTER_NEAREST
        )
        if exclude_mask is not None
        else np.zeros((height, width), np.uint8)
    )
    labels = np.zeros((height, width), np.float32)
    groups = [("orange", hsv[..., 0] < 30), ("blue", (hsv[..., 0] >= 90) & (hsv[..., 0] <= 125))]
    entries = []
    for color, hue in groups:
        mask = (hue & (hsv[..., 1] > 110) & (hsv[..., 2] > 50) & (excluded == 0)).astype(np.uint8)
        count, components, stats, _ = cv2.connectedComponentsWithStats(mask)
        widths = [
            w for x, y, w, h, area in stats[1:] if area > 150 and 30 <= w <= 65 and 15 <= h <= 65
        ]
        typical_width = float(np.median(widths)) if widths else 48.0
        for index in range(1, count):
            x, y, w, h, area = map(int, stats[index])
            if not (area >= 100 and w >= 15 and 12 <= h <= 75 and w <= 180):
                continue
            yy, xx = np.nonzero(components[y : y + h, x : x + w] == index)
            number = max(1, min(4, round(w / typical_width)))
            centres = (np.arange(number) + 0.5) * w / number
            for _ in range(15):
                assigned = np.argmin(abs(xx[:, None] - centres), axis=1)
                centres = np.array(
                    [
                        np.mean(xx[assigned == i]) if np.any(assigned == i) else centres[i]
                        for i in range(number)
                    ]
                )
            for part in range(number):
                selected = assigned == part
                if selected.sum() < 100:
                    continue
                px, py = xx[selected] + x, yy[selected] + y
                left, right, top, bottom = px.min(), px.max(), py.min(), py.max()
                if right - left < 18 or bottom - top < 12:
                    continue
                identifier = len(entries) + 1
                labels[py, px] = identifier
                centre = [(left + right) / 2, (top + bottom) / 2]
                point = np.array(
                    [centre[0] / 1000, wall["wall_height_m"] - centre[1] / 1000 - 0.001505, 0.026]
                )
                entries.append(
                    dict(
                        label=identifier,
                        color=color,
                        point=point,
                        rectified_bbox=[
                            int(left),
                            int(top),
                            int(right - left + 1),
                            int(bottom - top + 1),
                        ],
                        split_from=number,
                    )
                )
    native = cv2.warpPerspective(
        labels, np.linalg.inv(warp), (image.shape[1], image.shape[0]), flags=cv2.INTER_NEAREST
    ).astype(int)
    result = []
    for entry in entries:
        mask = (native == entry["label"]).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cv2.drawContours(mask, contours, -1, 1, thickness=cv2.FILLED)
        if mask.sum() < 10:
            continue
        entry["mask"] = mask.astype(bool)
        entry["bbox"] = list(map(int, cv2.boundingRect(mask)))
        result.append(entry)
    return result, rectified


def crop(mask, origin, side):
    x, y = map(int, origin)
    output = np.zeros((side, side), bool)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(mask.shape[1], x + side), min(mask.shape[0], y + side)
    if x0 < x1 and y0 < y1:
        output[y0 - y : y1 - y, x0 - x : x1 - x] = mask[y0:y1, x0:x1]
    return output


def edge(mask):
    value = mask.astype(np.uint8)
    return (value - cv2.erode(value, np.ones((3, 3), np.uint8))).astype(bool)


def observations(entries, cameras, point, side):
    result = []
    for eye, entry in entries.items():
        camera = cameras[eye]
        origin = np.floor(
            project_wall([point], camera["K"], camera["T_camera_wall"])[0] - side / 2
        ).astype(int)
        mask = crop(entry["mask"], origin, side)
        boundary = edge(mask)
        if boundary.sum() < 5:
            continue
        distance = cv2.distanceTransform(
            (~boundary).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )
        result.append(
            dict(camera=camera, origin=origin, mask=mask, boundary=boundary, distance=distance)
        )
    return result


def boundary_cost(point, data, renderer):
    costs = []
    for observation in data:
        camera = observation["camera"]
        model = renderer.render(
            point,
            np.asarray(camera["K"]),
            np.asarray(camera["T_camera_wall"]),
            observation["origin"],
        )
        boundary = edge(model)
        if boundary.sum() < 5:
            return 18.0
        distance = cv2.distanceTransform(
            (~boundary).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )
        costs.append(
            float(
                np.mean(np.minimum(distance[observation["boundary"]], 3.0) ** 2)
                + np.mean(np.minimum(observation["distance"][boundary], 3.0) ** 2)
            )
        )
    return float(np.mean(costs)) if costs else 18.0


def fit_cup(entries, cameras, renderer, scale, free_depth=True):
    point = np.mean([entry["point"] for entry in entries.values()], axis=0)
    point[2] = 0.026 * scale
    data = observations(entries, cameras, point, renderer.side)
    if not data:
        return point, 18.0
    dimensions = 3 if free_depth and len(data) >= 2 else 2
    bounds = [(-18, 18), (-18, 18)] + ([(-point[2] * 1000 + 3, 70)] if dimensions == 3 else [])

    def loss(delta):
        candidate = point.copy()
        candidate[:dimensions] += delta / 1000
        return boundary_cost(candidate, data, renderer)

    fitted = minimize(
        loss,
        np.zeros(dimensions),
        method="Powell",
        bounds=bounds,
        options=dict(maxiter=15, xtol=0.1, ftol=1e-3),
    )
    point[:dimensions] += fitted.x / 1000
    return point, float(fitted.fun)


def renderer_for_scale(scale, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"cup_{scale:.3f}.stl"
    if not target.exists():
        mesh = trimesh.load(simulator_asset("assets/hold/hold_visual.stl"), force="mesh")
        mesh.apply_scale(scale)
        mesh.export(target)
    return PixelCentreCupRenderer(target)


def choose_scale(view, camera, folder):
    samples = [
        entry
        for entry in view
        if entry["split_from"] == 1 and 25 <= entry["rectified_bbox"][2] <= 65
    ][:12]
    records = []
    for scale in [1.0, 1.15, 1.25]:
        renderer = renderer_for_scale(scale, folder)
        try:
            losses = [
                fit_cup({0: entry}, [camera], renderer, scale, free_depth=False)[1]
                for entry in samples
            ]
            records.append(
                dict(
                    scale=scale,
                    median_boundary_loss=float(np.median(losses)) if losses else 18.0,
                    samples=len(losses),
                )
            )
        finally:
            renderer.close()
    chosen = min(records, key=lambda value: value["median_boundary_loss"])["scale"]
    return chosen, records


def associate(first, second, maximum_distance=0.025):
    pairs = []
    if first and second:
        distance = np.linalg.norm(
            np.array([e["point"] for e in first])[:, None, :2]
            - np.array([e["point"] for e in second])[None, :, :2],
            axis=2,
        )
        colors = (
            np.array([e["color"] for e in first])[:, None]
            == np.array([e["color"] for e in second])[None, :]
        )
        valid = (distance <= maximum_distance) & colors
        reward = np.where(
            valid, 1 + (1 - distance / maximum_distance) / (min(distance.shape) + 1), 0
        )
        rows, columns = linear_sum_assignment(-reward)
        pairs = [(int(i), int(j)) for i, j in zip(rows, columns, strict=True) if valid[i, j]]
    used_a, used_b = {i for i, _ in pairs}, {j for _, j in pairs}
    return (
        [{0: first[i], 1: second[j]} for i, j in pairs]
        + [{0: e} for i, e in enumerate(first) if i not in used_a]
        + [{1: e} for i, e in enumerate(second) if i not in used_b]
    )


def map_cups(views, cameras, scale, cache):
    renderer = renderer_for_scale(scale, cache)
    result = []
    try:
        for index, entries in enumerate(associate(*views)):
            point, loss = fit_cup(entries, cameras, renderer, scale)
            result.append(
                dict(
                    id=f"cup_{index:03d}",
                    type="cup",
                    color=next(iter(entries.values()))["color"],
                    socket_hypothesis_wall_m=point.tolist(),
                    mounting_hypothesis_wall_m=(
                        point + [0, 0.014 * scale, -0.026 * scale]
                    ).tolist(),
                    boundary_loss_px2=loss,
                    image_supported=loss <= 6,
                    observed_by=list(entries),
                    depth_source="two-view upright CAD fitting"
                    if len(entries) == 2
                    else "assumed cup socket plane",
                    cad_scale=scale,
                    confidence=None,
                    contact_loading_confirmed=False,
                )
            )
    finally:
        renderer.close()
    return result
