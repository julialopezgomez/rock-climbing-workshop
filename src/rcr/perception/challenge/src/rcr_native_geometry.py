"""Image-only geometry for the pinned native RCR generator.

The public generator uses three emissive hold paints and flat triangular volume
faces (its triangle meshes have no UV coordinates). These are explicit synthetic
scene priors, not assumptions for camera photographs. No scene labels are read.
"""

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


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


FACES = ((0, 1, 3), (1, 2, 3), (2, 0, 3))


def project_many(points, k, t):
    cam = points @ t[:3, :3].T + t[:3, 3]
    q = cam @ k.T
    return q[:, :2] / q[:, 2:]


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
