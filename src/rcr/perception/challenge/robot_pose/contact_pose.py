"""Robust point-contact registration. Planned contacts are not measurements."""

import itertools

import numpy as np

from .frame_conventions import apply, transform


class Unobservable(ValueError):
    pass


def rigid_registration(local, world, weights=None, rotation=None):
    p, h = np.asarray(local, float), np.asarray(world, float)
    if (
        p.shape != h.shape
        or p.ndim != 2
        or p.shape[1] != 3
        or not len(p)
        or not np.isfinite([p, h]).all()
    ):
        raise ValueError("Matching finite Nx3 contact points required")
    w = np.ones(len(p)) if weights is None else np.asarray(weights, float)
    if w.shape != (len(p),) or not np.isfinite(w).all() or np.any(w < 0) or not w.sum():
        raise ValueError("Nonnegative contact weights with positive sum required")
    w = w / w.sum()
    cp = w @ p
    ch = w @ h
    if rotation is None:
        if np.linalg.matrix_rank((p - cp) * np.sqrt(w[:, None]), tol=1e-5) < 2:
            raise Unobservable(
                "Need three non-collinear point contacts, or a separately measured orientation"
            )
        u, _, vt = np.linalg.svd((p - cp).T @ (w[:, None] * (h - ch)))
        rotation = vt.T @ np.diag([1, 1, np.sign(np.linalg.det(vt.T @ u.T))]) @ u.T
    return transform(rotation, ch - rotation @ cp)


def estimate_contacts(local, world, weights=None, rotation=None, gate_m=0.006):
    p, h = np.asarray(local, float), np.asarray(world, float)
    w = np.ones(len(p)) if weights is None else np.asarray(weights, float)
    minimum = 1 if rotation is not None else 3
    if w.shape != (len(p),) or not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("Finite nonnegative contact weights required")
    if np.count_nonzero(w) > 0 and len(p) < minimum:
        raise Unobservable("Insufficient point contacts for the available orientation measurements")
    options = []
    for indices in itertools.combinations(np.flatnonzero(w > 0), minimum):
        ix = np.array(indices)
        try:
            t = rigid_registration(p[ix], h[ix], w[ix], rotation)
        except Unobservable:
            continue
        residual = np.linalg.norm(apply(t, p) - h, axis=1)
        good = residual < gate_m
        if good.sum() >= minimum:
            options.append((float(w[good].sum()), float(np.median(residual[good])), t, good))
    if not options:
        raise Unobservable("No geometrically consistent contact hypothesis")
    options.sort(key=lambda v: (-v[0], v[1]))
    _, _, t, good = options[0]
    for _ in range(4):
        residual = np.linalg.norm(apply(t, p[good]) - h[good], axis=1)
        robust = w[good] / np.maximum(1.0, residual / 0.002)
        t = rigid_registration(p[good], h[good], robust, rotation)
    residual = np.linalg.norm(apply(t, p) - h, axis=1)
    return t, dict(
        inliers=good.tolist(),
        residual_mm=(1000 * residual).tolist(),
        orientation_supplied=rotation is not None,
        constraints="endpoint positions only",
    )
