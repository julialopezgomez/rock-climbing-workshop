#!/usr/bin/env python3
"""Rectified stereo: study how pixel noise becomes depth error."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from common import project, write_results


def triangulate(left_pixels, right_pixels, intrinsics, baseline):
    """Return camera-0 points and a validity mask; reject nonpositive disparity."""
    disparity = left_pixels[:, 0] - right_pixels[:, 0]
    valid = disparity > 0
    points = np.full((len(disparity), 3), np.nan)
    z = intrinsics[0, 0] * baseline / disparity[valid]
    x = (left_pixels[valid, 0] - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (left_pixels[valid, 1] - intrinsics[1, 2]) * z / intrinsics[1, 1]
    points[valid] = np.column_stack((x, y, z))
    return points, valid


def run(args):
    if args.baseline <= 0 or args.noise < 0:
        raise ValueError("Baseline must be positive; noise must be nonnegative")
    rng = np.random.default_rng(args.seed)
    k = np.array([[700.0, 0.0, 320.0], [0.0, 700.0, 240.0], [0.0, 0.0, 1.0]])
    truth = np.column_stack(
        (
            rng.uniform(-0.12, 0.12, 500),
            rng.uniform(-0.1, 0.1, 500),
            rng.uniform(0.5, 2.0, 500),
        )
    )
    left = project(truth, k)
    right = project(truth - [args.baseline, 0, 0], k)
    left += rng.normal(0, args.noise, left.shape)
    right += rng.normal(0, args.noise, right.shape)
    estimated, valid = triangulate(left, right, k, args.baseline)
    depth_error = np.abs(estimated[valid, 2] - truth[valid, 2]) * 1000
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(truth[valid, 2], depth_error, s=9, color="0.3", alpha=0.5)
    depth = np.linspace(0.5, 2.0, 100)
    sigma_z = depth**2 / (k[0, 0] * args.baseline) * np.sqrt(2) * args.noise
    ax.plot(depth, 1000 * sigma_z, "k--", label="First-order depth standard deviation")
    ax.set(xlabel="Reference depth (m)", ylabel="Absolute depth error (mm)")
    ax.legend(fontsize=8)
    result = {
        "lab": "stereo",
        "seed": args.seed,
        "noise_px": args.noise,
        "baseline_m": args.baseline,
        "points": len(truth),
        "valid_points": int(valid.sum()),
        "median_depth_error_mm": float(np.median(depth_error)),
        "p90_depth_error_mm": float(np.percentile(depth_error, 90)),
    }
    write_results(args.out, result, fig)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--noise",
        type=float,
        default=0.5,
        help="Independent image noise standard deviation, pixels",
    )
    parser.add_argument("--baseline", type=float, default=0.064, help="Camera separation, metres")
    parser.add_argument("--out", type=Path, default=Path("outputs/stereo"))
    run(parser.parse_args())
