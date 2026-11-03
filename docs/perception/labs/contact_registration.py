#!/usr/bin/env python3
"""Rigid registration with tentative contact correspondences and sample consensus.

Synthetic correspondences are given. The algorithm does not detect contacts,
associate visual sockets, or establish seating/loading.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from common import pose_errors, transform_points, write_results
from scipy.spatial.transform import Rotation


def rigid_registration(model_points, observed_points):
    """Least-squares rigid alignment; reject collinear and coincident points."""
    if len(model_points) < 3:
        raise ValueError("Three non-collinear correspondences are required")
    a, b = (
        model_points - model_points.mean(0),
        observed_points - observed_points.mean(0),
    )
    if np.linalg.matrix_rank(a, tol=1e-9) < 2 or np.linalg.matrix_rank(b, tol=1e-9) < 2:
        raise ValueError("Collinear points leave rotation about their line unobservable")
    u, _, vt = np.linalg.svd(a.T @ b)
    correction = np.diag([1.0, 1.0, np.linalg.det(vt.T @ u.T)])
    rotation = vt.T @ correction @ u.T
    translation = observed_points.mean(0) - rotation @ model_points.mean(0)
    return rotation, translation


def consensus_registration(model_points, observed_points, gate=0.003, seed=7):
    rng = np.random.default_rng(seed)
    best_mask, best_score = None, (-1, -np.inf)
    for _ in range(150):
        sample = rng.choice(len(model_points), 3, replace=False)
        try:
            rotation, translation = rigid_registration(
                model_points[sample], observed_points[sample]
            )
        except ValueError:
            continue
        errors = np.linalg.norm(
            transform_points(model_points, rotation, translation) - observed_points,
            axis=1,
        )
        good = errors < gate
        if good.sum() < 3:
            continue
        score = (int(good.sum()), -float(np.median(errors[good])))
        if score > best_score:
            best_mask, best_score = good, score
    if best_mask is None:
        raise ValueError("No consistent non-collinear subset found")
    rotation, translation = rigid_registration(model_points[best_mask], observed_points[best_mask])
    return rotation, translation, best_mask


def run(args):
    if args.noise < 0 or args.gate <= 0 or not 0 <= args.outliers <= 3:
        raise ValueError("Need noise>=0 and between zero and three outliers")
    rng = np.random.default_rng(args.seed)
    points = np.array(
        [
            [-0.1, -0.1, 0],
            [0.1, -0.1, 0],
            [0.1, 0.1, 0],
            [-0.1, 0.1, 0],
            [-0.07, -0.02, 0.04],
            [0.08, -0.03, 0.02],
            [0.03, 0.06, -0.01],
            [-0.04, 0.08, 0.03],
        ]
    )
    reference_r = Rotation.from_euler("xyz", [0.1, -0.2, 0.3]).as_matrix()
    reference_t = np.array([0.3, 0.1, 0.5])
    clean = transform_points(points, reference_r, reference_t)
    observations = clean + rng.normal(0, args.noise / 1000, clean.shape)
    if args.outliers:
        observations[: args.outliers] += [0.04, -0.03, 0.02]
    all_r, all_t = rigid_registration(points, observations)
    rotation, translation, good = consensus_registration(
        points, observations, gate=args.gate / 1000, seed=args.seed
    )
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(*observations[good].T, label="Retained observations", color="black")
    ax.scatter(*observations[~good].T, label="Rejected observations", color="0.7", marker="x")
    ax.scatter(
        *transform_points(points, rotation, translation).T,
        label="Fitted geometry",
        marker="+",
        color="tab:red",
    )
    ax.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)")
    ax.legend(fontsize=8)
    result = {
        "lab": "contact_registration",
        "seed": args.seed,
        "noise_mm": args.noise,
        "injected_outliers": args.outliers,
        "gate_mm": args.gate,
        "retained": good.tolist(),
        "least_squares_all": pose_errors(all_r, all_t, reference_r, reference_t),
        "sample_consensus": pose_errors(rotation, translation, reference_r, reference_t),
        "scope": "Given synthetic point correspondences; does not confirm contact loading",
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
        help="Observation noise standard deviation, mm",
    )
    parser.add_argument("--outliers", type=int, default=2)
    parser.add_argument("--gate", type=float, default=3.0, help="Inlier gate, mm")
    parser.add_argument("--out", type=Path, default=Path("outputs/contact_registration"))
    run(parser.parse_args())
