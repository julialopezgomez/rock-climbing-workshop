"""Image-only geometry for the pinned native RCR generator.

The public generator uses three emissive hold paints and flat triangular volume
faces (its triangle meshes have no UV coordinates). These are explicit synthetic
scene priors, not assumptions for camera photographs. No scene labels are read.
"""

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.optimize import least_squares, minimize

from .rcr_methods import _calibration


def paint_masks(rgb):
    x = rgb.astype(float)
    r, g, b = np.moveaxis(x, -1, 0)
    # Emission sets a lower bound before the additive scene illumination.
    return [
        (r >= 235) & (r - g > 40) & (r - b > 40),
        (g >= 195) & (g - r > 30) & (g - b > 10),
        (b >= 230) & (b - r > 30) & (b - g > 5),
    ]


def flat_faces(rgb):
    """Find sizeable, nearly constant non-background material components."""
    kernel = np.ones((3, 3), np.uint8)
    spread = cv2.dilate(rgb, kernel).astype(int) - cv2.erode(rgb, kernel).astype(int)
    flat = spread.max(-1) <= 1
    colors, counts = np.unique(rgb[flat], axis=0, return_counts=True)
    background = np.median(np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]]), axis=0)
    paints = np.logical_or.reduce(paint_masks(rgb))
    used = np.zeros(rgb.shape[:2], bool)
    result = []
    for color in colors[np.argsort(counts)[::-1]][counts[np.argsort(counts)[::-1]] >= 60]:
        if np.linalg.norm(color - background) < 1:
            continue
        mask = (np.max(np.abs(rgb.astype(np.int16) - color), axis=-1) <= 3) & ~paints & ~used
        n, labels, stats, centers = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        for idx in range(1, n):
            x, y, w, h, area = stats[idx]
            if not (70 <= area <= 13000 and 5 <= w <= 180 and 5 <= h <= 150):
                continue
            m = labels == idx
            if (m & flat).sum() < 35:
                continue
            used |= m
            contour = max(
                cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
                key=cv2.contourArea,
            )
            hull = cv2.convexHull(contour)
            poly = cv2.approxPolyDP(hull, 1.5, True).reshape(-1, 2).astype(float) + 0.5
            if len(poly) != 3:
                area_triangle, tri = cv2.minEnclosingTriangle(hull)
                if tri is not None and area_triangle / max(cv2.contourArea(hull), 1) < 1.3:
                    poly = tri.reshape(-1, 2).astype(float) + 0.5
            result.append(
                dict(mask=m, color=color, area=int(area), center=centers[idx], polygon=poly)
            )
    return result, used


def wall_point(pixel, k, t):
    ray = t[:3, :3].T @ np.linalg.solve(k, np.r_[pixel, 1.0])
    origin = -t[:3, :3].T @ t[:3, 3]
    return origin - ray * origin[2] / ray[2]


def vertices(v):
    x, y, w, d = v
    h = 0.08
    return np.array(
        [[x - w / 2, y - h / 2, 0], [x + w / 2, y - h / 2, 0], [x, y + h / 2, 0], [x, y, d]]
    )


FACES = ((0, 1, 3), (1, 2, 3), (2, 0, 3))


def project_many(points, k, t):
    cam = points @ t[:3, :3].T + t[:3, 3]
    q = cam @ k.T
    return q[:, :2] / q[:, 2:]


def volume_candidates(rgb, k, t):
    faces, _ = flat_faces(rgb)
    candidates = []
    for face in faces:
        poly = face["polygon"]
        if len(poly) != 3:
            continue
        for apex in range(3):
            rest = [i for i in range(3) if i != apex]
            for reverse in (False, True):
                order = rest[::-1] if reverse else rest
                observed = poly[[*order, apex]]
                a, b = [wall_point(p, k, t) for p in observed[:2]]
                for fi in range(3):
                    if fi == 0:
                        x = (a[0] + b[0]) / 2
                        y = (a[1] + b[1]) / 2 + 0.04
                        w = b[0] - a[0]
                        error = abs(a[1] - b[1])
                    elif fi == 1:
                        x = b[0]
                        y = (a[1] + b[1]) / 2
                        w = 2 * (a[0] - b[0])
                        error = abs(b[1] - a[1] - 0.08)
                    else:
                        x = a[0]
                        y = (a[1] + b[1]) / 2
                        w = 2 * (a[0] - b[0])
                        error = abs(a[1] - b[1] - 0.08)
                    if not (
                        0.06 < w < 0.23 and error < 0.012 and -0.45 < x < 0.45 and -0.05 < y < 1.45
                    ):
                        continue
                    v = np.array([x, y, w, 0.10])

                    def residual(z, *, fi=fi, observed=observed):
                        return (project_many(vertices(z)[list(FACES[fi])], k, t) - observed).ravel()

                    fit = least_squares(
                        residual,
                        v,
                        bounds=([-0.5, -0.1, 0.06, 0.025], [0.5, 1.5, 0.23, 0.23]),
                        max_nfev=35,
                    )
                    loss = float(np.mean(residual(fit.x) ** 2))
                    if loss < 4:
                        candidates.append(
                            dict(
                                parameters=fit.x,
                                face_index=fi,
                                loss=loss,
                                color=face["color"],
                                observed=face,
                            )
                        )
    return candidates, faces


def triangle_mask(poly, origin, shape, scale=4):
    h, w = shape
    out = np.zeros((h * scale, w * scale), np.uint8)
    cv2.fillConvexPoly(out, np.rint((poly - origin) * scale - 0.5).astype(np.int32), 1)
    return cv2.resize(out.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)


def soft_region(rgb, mask):
    """Estimate antialias coverage from nearby interior/exterior RGB samples."""
    hard = mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    inside = cv2.erode(hard, kernel) > 0
    if not inside.any():
        inside = mask
    outside = cv2.dilate(hard, kernel) == 0
    if not outside.any():
        return hard.astype(float)
    _, ci = distance_transform_edt(~inside, return_indices=True)
    _, bi = distance_transform_edt(~outside, return_indices=True)
    c = rgb[ci[0], ci[1]].astype(float)
    b = rgb[bi[0], bi[1]].astype(float)
    delta = c - b
    contrast = (delta * delta).sum(-1)
    alpha = np.clip(((rgb.astype(float) - b) * delta).sum(-1) / np.maximum(contrast, 1), 0, 1)
    band = (cv2.dilate(hard, kernel) > 0) & ~inside
    return np.where(band & (contrast > 100), alpha, hard).astype(np.float32)


def estimate_volumes(images, calibration):
    k, t = _calibration(calibration)
    candidates, faces = volume_candidates(images[0], k[0], t[0])
    right_faces, _ = flat_faces(images[1])
    accepted = []
    for c in sorted(candidates, key=lambda c: c["loss"]):
        v = c["parameters"]
        fi = c["face_index"]
        obs0 = c["observed"]
        center = project_many(vertices(v)[list(FACES[fi])], k[1], t[1]).mean(0)
        choices = [
            f
            for f in right_faces
            if np.linalg.norm(f["color"].astype(float) - c["color"]) < 15
            and np.linalg.norm(f["center"] - center) < 35
        ]
        if not choices:
            continue
        obs1 = min(choices, key=lambda f: np.linalg.norm(f["center"] - center))
        obs = []
        for f in (obs0, obs1):
            yy, xx = np.where(f["mask"])
            origin = np.array([max(0, xx.min() - 12), max(0, yy.min() - 12)])
            end = np.array(
                [min(f["mask"].shape[1], xx.max() + 13), min(f["mask"].shape[0], yy.max() + 13)]
            )
            hard = f["mask"][origin[1] : end[1], origin[0] : end[0]]
            crop = images[len(obs)][origin[1] : end[1], origin[0] : end[0]]
            mask = soft_region(crop, hard)
            paint = np.logical_or.reduce(paint_masks(crop))
            weight = 1 - cv2.dilate(paint.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(float)
            obs.append((origin, mask, weight))

        def loss(z, *, fi=fi, obs=obs):
            score = 0
            for eye, (origin, mask, weight) in enumerate(obs):
                soft = triangle_mask(
                    project_many(vertices(z)[list(FACES[fi])], k[eye], t[eye]), origin, mask.shape
                )
                score += (np.square(soft - mask) * weight).sum() / max((mask * weight).sum(), 1)
            return float(score)

        scale = np.array([0.01, 0.01, 0.01, 0.05])
        start = v.copy()
        fit = minimize(
            lambda d, loss=loss, start=start, scale=scale: loss(start + d * scale),
            np.zeros(4),
            method="Powell",
            bounds=[(-1, 1), (-1, 1), (-1, 1), (-1, 1)],
            options={"maxiter": 6, "xtol": 0.003, "ftol": 1e-5},
        )
        v = start + fit.x * scale
        score = loss(v)
        if score > 0.32 or not (0.07 < v[2] < 0.21 and 0.03 < v[3] < 0.21):
            continue
        accepted.append(dict(parameters=v, loss=score, face_index=fi))
    result = []
    for candidate in sorted(accepted, key=lambda a: a["loss"]):
        if any(
            np.linalg.norm(candidate["parameters"][:2] - a["parameters"][:2]) < 0.03 for a in result
        ):
            continue
        result.append(candidate)
    return result
