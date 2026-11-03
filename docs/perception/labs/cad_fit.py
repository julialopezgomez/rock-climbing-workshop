#!/usr/bin/env python3
"""Multi-start silhouette fitting of an asymmetric planar CAD outline.

This example estimates image-plane translation and yaw, not full 6D hold pose.
The target is generated from a polygon; no detector or research renderer is used.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from common import write_results
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize

POLYGON = np.array([[-27, -20], [12, -20], [27, -5], [8, 2], [8, 24], [-27, 24]], float)


def boundary_points():
    fractions = np.linspace(0, 1, 35, endpoint=False)[:, None]
    return np.concatenate(
        [
            a + fractions * (b - a)
            for a, b in zip(POLYGON, np.roll(POLYGON, -1, axis=0), strict=False)
        ]
    )


def place(points, pose):
    x, y, angle = pose
    c, s = np.cos(angle), np.sin(angle)
    return points @ np.array([[c, s], [-s, c]]) + [x, y]


def fit_outline(observed_mask, starts, keep_fraction=1.0):
    """Select by image residual alone. The reference pose is not an input."""
    edge = observed_mask - cv2.erode(observed_mask, np.ones((3, 3), np.uint8))
    distance = cv2.distanceTransform(1 - edge, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    points = boundary_points()

    def objective(pose):
        uv = place(points, pose)
        residual = map_coordinates(
            distance, [uv[:, 1], uv[:, 0]], order=1, mode="constant", cval=100.0
        )
        keep = max(1, int(np.ceil(keep_fraction * len(residual))))
        return float(np.mean(np.sort(residual**2)[:keep]))

    candidates = []
    for angle in np.linspace(0, 2 * np.pi, starts, endpoint=False):
        initial = [80.0, 80.0, angle]
        fitted = minimize(
            objective,
            initial,
            method="Powell",
            bounds=[(30, 130), (30, 130), (angle - np.pi / 3, angle + np.pi / 3)],
            options={"maxiter": 120, "xtol": 1e-4, "ftol": 1e-5},
        )
        candidates.append(
            {
                "pose": fitted.x.tolist(),
                "cost": float(fitted.fun),
                "converged": bool(fitted.success),
            }
        )
    return min(candidates, key=lambda item: item["cost"]), candidates


def run(args):
    if args.starts < 1 or not 0 <= args.occlusion < 1 or not 0 < args.keep <= 1:
        raise ValueError("Need starts>=1, 0<=occlusion<1 and 0<keep<=1")
    rng = np.random.default_rng(args.seed)
    truth = np.array([86.0, 78.0, np.deg2rad(args.yaw)]) + rng.uniform(
        [-5, -5, -0.08], [5, 5, 0.08]
    )
    mask = np.zeros((160, 160), np.uint8)
    cv2.fillPoly(mask, [np.round(place(POLYGON, truth)).astype(np.int32)], 1)
    if args.occlusion:
        mask[:, : int(55 + 60 * args.occlusion)] = 0
    chosen, candidates = fit_outline(mask, args.starts, args.keep)
    pose = np.asarray(chosen["pose"])
    angle_error = abs(
        np.rad2deg(np.arctan2(np.sin(pose[2] - truth[2]), np.cos(pose[2] - truth[2])))
    )
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax in axes:
        ax.imshow(mask, cmap="gray", vmin=0, vmax=1)
        ax.set(xlim=(20, 140), ylim=(140, 20), xlabel="u (pixels)", ylabel="v (pixels)")
    for ax, candidate in zip(axes, [candidates[0], chosen], strict=False):
        vertices = place(np.vstack([POLYGON, POLYGON[0]]), candidate["pose"])
        ax.plot(vertices[:, 0], vertices[:, 1], color="tab:red", lw=1)
    axes[0].set_title("First initialisation")
    axes[1].set_title("Lowest observed residual")
    result = {
        "lab": "cad_fit",
        "seed": args.seed,
        "starts": args.starts,
        "occlusion": args.occlusion,
        "keep_fraction": args.keep,
        "translation_error_px": float(np.linalg.norm(pose[:2] - truth[:2])),
        "yaw_error_deg": float(angle_error),
        "selected": chosen,
        "candidates": candidates,
        "scope": "Planar synthetic outline, 3 parameters; no 6D or benchmark accuracy claim",
    }
    write_results(args.out, result, fig)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--starts", type=int, default=4)
    parser.add_argument("--yaw", type=float, default=37.0, help="Synthetic target yaw, degrees")
    parser.add_argument("--occlusion", type=float, default=0.0)
    parser.add_argument(
        "--keep",
        type=float,
        default=1.0,
        help="Fraction of boundary residuals retained",
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/cad_fit"))
    run(parser.parse_args())
