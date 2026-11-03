"""Known-q robot landmark PnP frontend. Does not fabricate detected landmarks."""

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .frame_conventions import apply, project, transform
from .refine import refine_pose


def estimate_keypoints(observations, orientation_prior=None):
    candidates = []
    for obs in observations:
        p, uv = np.asarray(obs["points_trunk"], float), np.asarray(obs["pixels"], float)
        if len(p) < 4:
            continue
        c = obs["camera"]
        k = np.asarray(c["K"], float)
        dist = np.asarray(c.get("distortion", [0] * 5), float)
        ok, rv, tv, inliers = cv2.solvePnPRansac(
            p, uv, k, dist, iterationsCount=300, reprojectionError=3.0, flags=cv2.SOLVEPNP_SQPNP
        )
        if ok:
            candidates.append(
                np.linalg.inv(c["T_camera_wall"])
                @ transform(Rotation.from_rotvec(rv.ravel()).as_matrix(), tv.ravel())
            )
    if not candidates:
        raise ValueError(
            "Need four identified nondegenerate robot landmarks in a"
            "t least one view for this initializer"
        )
    ranked = []
    for t in candidates:
        fitted, info = refine_pose(t, observations, orientation_prior=orientation_prior)
        ranked.append((info["cost"], fitted, info))
    _, t, info = min(ranked, key=lambda x: x[0])
    # Remove gross correspondence outliers, then jointly refit all views.
    clean = []
    for o in observations:
        residual = np.linalg.norm(
            project(apply(t, o["points_trunk"]), o["camera"]) - o["pixels"], axis=1
        )
        good = residual < 4.0
        if good.any():
            clean.append(
                {
                    **o,
                    "points_trunk": np.asarray(o["points_trunk"])[good],
                    "pixels": np.asarray(o["pixels"])[good],
                }
            )
    if sum(len(o["pixels"]) for o in clean) >= 4:
        t, info = refine_pose(t, clean, orientation_prior=orientation_prior)
    return t, {
        **info,
        "initializer_count": len(candidates),
        "retained_keypoints": sum(len(o["pixels"]) for o in clean),
    }


def estimate_generalized(observations):
    """PoseLib GPnP: landmarks need not be shared between the cameras.

    Optional dependency lives in development/envs/robot-pose, not the base env.
    This consumes detected correspondences; it is not itself a keypoint network.
    """
    import poselib

    from .frame_conventions import to_wxyz

    cameras = []
    extrinsics = []
    points = []
    pixels = []
    for o in observations:
        c = o["camera"]
        k = np.asarray(c["K"])
        tc = np.asarray(c["T_camera_wall"])
        if np.any(c.get("distortion", [0] * 5)):
            raise ValueError("Generalized adapter expects undistorted keypoint pixels")
        cameras.append(
            dict(
                model="PINHOLE",
                width=int(c.get("width", 1280)),
                height=int(c.get("height", 720)),
                params=[k[0, 0], k[1, 1], k[0, 2], k[1, 2]],
            )
        )
        p = poselib.CameraPose()
        p.q = to_wxyz(tc[:3, :3])
        p.t = tc[:3, 3]
        extrinsics.append(p)
        points.append(np.asarray(o["points_trunk"], float))
        pixels.append(np.asarray(o["pixels"], float))
    pose, info = poselib.estimate_generalized_absolute_pose(
        pixels,
        points,
        extrinsics,
        cameras,
        {"max_reproj_error": 3.0, "min_iterations": 100, "max_iterations": 2000, "seed": 0},
        {},
    )
    if info["num_inliers"] < 4:
        raise ValueError("Generalized pose has insufficient support")
    t = transform(pose.R, pose.t)
    t, refined = refine_pose(t, observations)
    return t, dict(
        **refined, initializer="PoseLib generalized absolute pose", num_inliers=info["num_inliers"]
    )
