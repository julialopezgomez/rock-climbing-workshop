#!/usr/bin/env python3
"""CPU colour/epipolar baseline on the included real simulator image pair.

This reads RGB and calibration only. Region centroids are approximate surface
correspondences, not CAD anchors. The output is not a competition submission.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from common import write_results
from scipy.optimize import linear_sum_assignment


def detect_regions(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = ((hsv[:, :, 1] > 85) & (hsv[:, :, 2] > 95)).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    regions = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if 5 <= area <= 400:
            regions.append(
                {
                    "pixel": centroids[index],
                    "area": area,
                    "hue": float(np.median(hsv[:, :, 0][labels == index])),
                }
            )
    return regions


def camera_matrices(calibration):
    cameras = [calibration["cameras"][eye] for eye in ["left", "right"]]
    intrinsics = [np.asarray(c["K"], float).copy() for c in cameras]
    # This source fixture stores raster intrinsics, not OpenCV pixel-centre K.
    for k in intrinsics:
        k[0, 2] -= 0.5
        k[1, 2] -= 0.5
    transforms = [np.asarray(c["T_camera_scene"], float) for c in cameras]
    r = transforms[1][:3, :3] @ transforms[0][:3, :3].T
    t = transforms[1][:3, 3] - r @ transforms[0][:3, 3]
    cross = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    fundamental = np.linalg.inv(intrinsics[1]).T @ cross @ r @ np.linalg.inv(intrinsics[0])
    return (
        [k @ t[:3] for k, t in zip(intrinsics, transforms, strict=False)],
        transforms,
        fundamental,
    )


def match_and_triangulate(regions, projections, transforms, fundamental):
    if not all(regions):
        return []
    costs = np.full((len(regions[0]), len(regions[1])), 1e6)
    for i, left in enumerate(regions[0]):
        line = fundamental @ np.r_[left["pixel"], 1.0]
        for j, right in enumerate(regions[1]):
            distance = abs(line @ np.r_[right["pixel"], 1.0]) / np.linalg.norm(line[:2])
            hue = abs(left["hue"] - right["hue"])
            hue = min(hue, 180 - hue)
            ratio = max(left["area"], right["area"]) / min(left["area"], right["area"])
            if distance < 2.5 and hue < 15 and ratio < 3:
                costs[i, j] = distance + hue / 15 + abs(np.log(ratio))
    matches = []
    rows, columns = linear_sum_assignment(costs)
    for i, j in zip(rows, columns, strict=False):
        if costs[i, j] >= 1e6:
            continue
        pixels = [regions[0][i]["pixel"], regions[1][j]["pixel"]]
        homogeneous = cv2.triangulatePoints(
            *projections, pixels[0][:, None], pixels[1][:, None]
        ).ravel()
        if abs(homogeneous[3]) < 1e-10:
            continue
        point = homogeneous[:3] / homogeneous[3]
        if any((t[:3, :3] @ point + t[:3, 3])[2] <= 0 for t in transforms):
            continue
        matches.append(
            {
                "position_scene_m": point.tolist(),
                "pixels": [p.tolist() for p in pixels],
                "matching_cost": float(costs[i, j]),
            }
        )
    return matches


def run(args):
    bgr = cv2.imread(str(args.image))
    if bgr is None:
        raise FileNotFoundError(args.image)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    calibration = json.loads(args.calibration.read_text())
    width = calibration["image_width"]
    if rgb.shape[:2] != (calibration["image_height"], 2 * width):
        raise ValueError("Input dimensions do not match the stereo calibration")
    images = [rgb[:, :width], rgb[:, width:]]
    regions = [detect_regions(image) for image in images]
    projections, transforms, fundamental = camera_matrices(calibration)
    matches = match_and_triangulate(regions, projections, transforms, fundamental)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for eye, (ax, image) in enumerate(zip(axes, images, strict=False)):
        ax.imshow(image)
        for index, match in enumerate(matches):
            x, y = match["pixels"][eye]
            ax.plot(x, y, "ro", mfc="none", ms=5)
            ax.text(
                x + 5,
                y,
                str(index),
                fontsize=8,
                color="black",
                bbox={"facecolor": "white", "alpha": 0.7, "pad": 0.5},
            )
        ax.set_title(["Left", "Right"][eye])
        ax.axis("off")
    result = {
        "lab": "real_stereo",
        "regions_per_camera": [len(r) for r in regions],
        "matched_regions": len(matches),
        "surface_point_candidates": matches,
        "scope": (
            "One saved simulator scene; unscored colour-centroid baseline; "
            "not hold-anchor pose estimates"
        ),
    }
    write_results(args.out, result, fig)
    return result


if __name__ == "__main__":
    default = Path(__file__).resolve().parents[1] / "examples/hold_scene"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=default / "stereo.png")
    parser.add_argument("--calibration", type=Path, default=default / "calibration.json")
    parser.add_argument("--out", type=Path, default=Path("outputs/real_stereo"))
    run(parser.parse_args())
