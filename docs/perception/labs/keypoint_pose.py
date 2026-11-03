#!/usr/bin/env python3
"""Known-joint toy robot: PnP followed by robust two-camera refinement.

Landmarks are generated from geometry and corrupted in image space. This lab
supplies correspondences; it does not train or run a landmark detector.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from common import pose_errors, project, transform_points, write_results
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def robot_landmarks(joints):
    trunk = np.array(
        [[x, y, z] for x in [-0.05, 0.05] for y in [-0.07, 0.07] for z in [-0.025, 0.025]]
    )
    limbs = []
    for sign, angle in zip([-1, 1], joints, strict=False):
        shoulder = np.array([sign * 0.05, -0.05, 0.0])
        for length in [0.05, 0.10, 0.15, 0.20]:
            limbs.append(shoulder + [sign * length * np.cos(angle), length * np.sin(angle), 0.015])
    return np.vstack([trunk, limbs])


def estimate_pose(model_points, pixels, intrinsics, camera_centres, seed=7):
    """Return model-to-camera-0 rotation/translation and retained observation counts."""
    cv2.setRNGSeed(seed % (2**31))
    ok, rvec, tvec, _ = cv2.solvePnPRansac(
        model_points,
        pixels[0],
        intrinsics,
        None,
        iterationsCount=300,
        reprojectionError=3.0,
        confidence=0.999,
        flags=cv2.SOLVEPNP_SQPNP,
    )
    if not ok:
        raise ValueError("PnP failed: need nondegenerate identified correspondences")
    initial = np.r_[rvec.ravel(), tvec.ravel()]
    rotation = Rotation.from_rotvec(initial[:3]).as_matrix()
    retained = [
        np.linalg.norm(
            project(transform_points(model_points, rotation, initial[3:]) - c, intrinsics) - uv,
            axis=1,
        )
        < 4.0
        for c, uv in zip(camera_centres, pixels, strict=False)
    ]
    if sum(int(g.sum()) for g in retained) < 6:
        raise ValueError("Too few geometrically consistent image observations")

    def residual(pose):
        camera0 = transform_points(
            model_points, Rotation.from_rotvec(pose[:3]).as_matrix(), pose[3:]
        )
        return np.concatenate(
            [
                (project(camera0 - c, intrinsics) - uv)[good].ravel()
                for c, uv, good in zip(camera_centres, pixels, retained, strict=False)
            ]
        )

    fit = least_squares(residual, initial, loss="soft_l1", f_scale=1.0, max_nfev=300)
    if not fit.success:
        raise ValueError(f"Refinement failed: {fit.message}")
    return (
        Rotation.from_rotvec(fit.x[:3]).as_matrix(),
        fit.x[3:],
        [int(g.sum()) for g in retained],
    )


def run(args):
    if args.noise < 0 or not 0 <= args.outliers < 0.5:
        raise ValueError("Need noise>=0 and 0<=outliers<0.5")
    rng = np.random.default_rng(args.seed)
    k = np.array([[700.0, 0.0, 320.0], [0.0, 700.0, 240.0], [0.0, 0.0, 1.0]])
    centres = np.array([[0.0, 0.0, 0.0], [0.10, 0.0, 0.0]])
    true_joints = np.deg2rad([35.0, -20.0])
    truth_points = robot_landmarks(true_joints)
    measured_points = robot_landmarks(true_joints + np.deg2rad(args.joint_error))
    reference_r = Rotation.from_euler("xyz", [0.2, -0.15, 0.3]).as_matrix()
    reference_t = np.array([0.025, -0.015, 0.85])
    reference_camera = transform_points(truth_points, reference_r, reference_t)
    pixels = [project(reference_camera - c, k) for c in centres]
    for uv in pixels:
        uv += rng.normal(0, args.noise, uv.shape)
        count = int(args.outliers * len(uv))
        wrong = rng.choice(len(uv), count, replace=False)
        uv[wrong] += rng.uniform(25, 70, (count, 2))
    rotation, translation, retained = estimate_pose(measured_points, pixels, k, centres, args.seed)
    estimated_camera = transform_points(measured_points, rotation, translation)
    endpoint_error = 1000 * np.linalg.norm(
        estimated_camera[[11, 15]] - reference_camera[[11, 15]], axis=1
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(*pixels[0].T, color="0.5", label="Noisy observations")
    ax.scatter(
        *project(estimated_camera, k).T,
        marker="+",
        color="black",
        label="Fitted projection",
    )
    ax.set(
        xlabel="u (pixels)",
        ylabel="v (pixels)",
        title="Camera 0; the fit uses both views",
    )
    ax.invert_yaxis()
    ax.legend()
    result = dict(
        lab="keypoint_pose",
        seed=args.seed,
        noise_px=args.noise,
        outlier_fraction=args.outliers,
        joint_error_deg=args.joint_error,
        retained_per_camera=retained,
        **pose_errors(rotation, translation, reference_r, reference_t),
        worst_endpoint_error_mm=float(endpoint_error.max()),
        rotation_model_to_camera0=rotation.tolist(),
        translation_model_to_camera0_m=translation.tolist(),
        scope="Synthetic identified landmarks; no learned detector or Microban benchmark",
    )
    write_results(args.out, result, fig)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--noise", type=float, default=0.5)
    parser.add_argument("--outliers", type=float, default=0.15)
    parser.add_argument(
        "--joint-error",
        type=float,
        default=0.0,
        help="Offset in each supplied joint angle, degrees",
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/keypoint_pose"))
    run(parser.parse_args())
