"""Common robust multi-camera reprojection fitting, with optional point contacts."""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .frame_conventions import apply, project, transform


def pack(t):
    return np.r_[t[:3, 3], Rotation.from_matrix(t[:3, :3]).as_rotvec()]


def unpack(x):
    return transform(Rotation.from_rotvec(x[3:]).as_matrix(), x[:3])


def refine_pose(initial, observations, contacts=None, orientation_prior=None, max_nfev=120):
    """Each observation has points_trunk Nx3, pixels Nx2, camera, optional sigma_px."""

    def residual(x):
        t = unpack(x)
        parts = []
        for obs in observations:
            r = project(apply(t, obs["points_trunk"]), obs["camera"]) - np.asarray(obs["pixels"])
            parts.extend((r / obs.get("sigma_px", 1.0)).ravel())
        if contacts is not None:
            parts.extend(
                (
                    (apply(t, contacts["points_trunk"]) - contacts["points_wall"])
                    / contacts.get("sigma_m", 0.003)
                ).ravel()
            )
        if orientation_prior is not None:
            r, sigma = orientation_prior
            parts.extend(Rotation.from_matrix(t[:3, :3] @ np.asarray(r).T).as_rotvec() / sigma)
        return np.asarray(parts)

    fit = least_squares(
        residual, pack(initial), loss="soft_l1", f_scale=2.0, max_nfev=max_nfev, x_scale="jac"
    )
    t = unpack(fit.x)
    errors = [
        np.linalg.norm(project(apply(t, o["points_trunk"]), o["camera"]) - o["pixels"], axis=1)
        for o in observations
    ]
    return t, dict(
        cost=float(fit.cost),
        nfev=fit.nfev,
        reprojection_median_px=float(np.median(np.concatenate(errors))) if errors else None,
    )
