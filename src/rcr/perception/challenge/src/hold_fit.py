"""Explicit-renderer CAD refinement; supports arbitrary declared wall sizes."""

import cv2
import numpy as np
from scipy.optimize import differential_evolution, minimize

from .rcr_exact_fit import crop_mask
from .rcr_methods import _project, _rotation
from .rcr_native_geometry import soft_region


def loss_images(softs, obs):
    return sum(
        (
            float((np.square(s - o["mask"]) * o["weight"]).sum() / max(o["mask"].sum(), 1))
            for s, o in zip(softs, obs, strict=False)
        )
    )


def render_pair(p, r, obs, k, t, *, render):
    return [
        render.render_soft(p, r, ki, ti, o["origin"]) for o, ki, ti in zip(obs, k, t, strict=False)
    ]


def fit_plane(initial, r, normal, plane_point, obs, k, t, rounds=3, *, render):
    """Fast exact rerender/translation alternation; fixed supporting plane."""
    p = initial.copy()
    n = np.asarray(normal)
    p[2] = plane_point[2] - (n[0] * (p[0] - plane_point[0]) + n[1] * (p[1] - plane_point[1])) / n[2]
    center = r @ render.center
    yy, xx = np.mgrid[:64, :64].astype(np.float32)
    for it in range(rounds):
        softs = render_pair(p, r, obs, k, t, render=render)
        uv0 = [_project(p + center, ki, ti)[0] for ki, ti in zip(k, t, strict=False)]

        def unpack(x, *, p=p):
            return p + np.r_[x * 0.01, -0.01 * (n[:2] @ x) / n[2]]

        def loss(x, *, softs=softs, uv0=uv0):
            q = unpack(x)
            shift = [
                _project(q + center, ki, ti)[0] - u for ki, ti, u in zip(k, t, uv0, strict=False)
            ]
            rendered = [
                cv2.remap(
                    s,
                    xx - float(d[0]),
                    yy - float(d[1]),
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                )
                for s, d in zip(softs, shift, strict=False)
            ]
            return loss_images(rendered, obs)

        bound = 3.5 if it == 0 else 0.25
        initial_shift = np.zeros(2)
        if it == 0:
            grid = [
                np.array([a, b])
                for a in np.linspace(-2.5, 2.5, 13)
                for b in np.linspace(-2.5, 2.5, 13)
            ]
            initial_shift = min(grid, key=loss)
        fit = minimize(
            loss,
            initial_shift,
            method="Powell",
            bounds=[(-bound, bound)] * 2,
            options={"maxiter": 6, "xtol": 0.001, "ftol": 1e-06},
        )
        q = unpack(fit.x)
        if (
            loss_images(render_pair(q, r, obs, k, t, render=render), obs) > loss_images(softs, obs)
            and it > 0
        ):
            break
        p = q
    return (p, r, loss_images(render_pair(p, r, obs, k, t, render=render), obs))


def plane_initial(props, k, t, r, n, a, *, render):
    center = r @ render.center
    points = []
    for prop, ki, ti in zip(props, k, t, strict=False):
        origin = -ti[:3, :3].T @ ti[:3, 3]
        ray = ti[:3, :3].T @ np.linalg.solve(ki, np.r_[prop["center"] + 0.5, 1.0])
        den = n @ ray
        if abs(den) > 1e-08:
            points.append(origin + ray * (n @ (a + center - origin)) / den - center)
    return np.mean(points, axis=0)


def fit_translation(initial, r, obs, k, t, rounds=2, *, render, maximum_depth=0.4):
    p = initial.copy()
    p[2] = np.clip(p[2], 0, maximum_depth)
    center = r @ render.center
    yy, xx = np.mgrid[:64, :64].astype(np.float32)
    for _it in range(rounds):
        softs = render_pair(p, r, obs, k, t, render=render)
        proj = [_project(p + center, ki, ti) for ki, ti in zip(k, t, strict=False)]

        def unpack(x, *, p=p):
            return p + x * np.array([0.01, 0.01, 0.05])

        def loss(x, *, proj=proj, softs=softs):
            q = unpack(x)
            rendered = []
            for s, (u, z), ki, ti, o in zip(softs, proj, k, t, obs, strict=False):
                uv, zz = _project(q + center, ki, ti)
                scale = z / zz
                d = uv - u
                cx, cy = u - o["origin"]
                rendered.append(
                    cv2.remap(
                        s,
                        ((xx - cx - float(d[0])) / scale + cx).astype(np.float32),
                        ((yy - cy - float(d[1])) / scale + cy).astype(np.float32),
                        cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                    )
                )
            return loss_images(rendered, obs)

        fit = minimize(
            loss,
            np.zeros(3),
            method="Powell",
            bounds=[(-3, 3), (-3, 3), (-p[2] / 0.05, (maximum_depth - p[2]) / 0.05)],
            options={"maxiter": 6, "xtol": 0.002, "ftol": 1e-05},
        )
        p = unpack(fit.x)
    return (p, r, loss_images(render_pair(p, r, obs, k, t, render=render), obs))


def refine_inclined(p, r, obs, k, t, global_search=False, *, render, maximum_depth=0.4):
    n = r[:, 2]
    theta = float(np.arccos(np.clip(n[2], -1, 1)))
    phi = float(np.arctan2(n[1], n[0]))
    local = _rotation(n, 0).T @ r
    raw = np.arctan2(local[1, 0], local[0, 0])
    yaw = min(
        [0.0, -np.pi / 4, np.pi / 4],
        key=lambda y: abs(np.arctan2(np.sin(y - raw), np.cos(y - raw))),
    )
    bottom = phi < 0
    x0 = np.r_[p * 100, theta] if bottom else np.r_[p * 100, theta, phi]
    bounds = [
        (p[0] * 100 - 1, p[0] * 100 + 1),
        (p[1] * 100 - 1, p[1] * 100 + 1),
        (max(0, p[2] * 100 - 3), min(maximum_depth * 100, p[2] * 100 + 3)),
        (max(0.5, theta - 0.2), min(np.arccos(0.25), theta + 0.2)),
    ]
    if not bottom:
        bounds.append((max(0.4, phi - 0.15), min(2.75, phi + 0.15)))

    def unpack(x):
        ph = -np.pi / 2 if bottom else x[4]
        nn = np.array([np.sin(x[3]) * np.cos(ph), np.sin(x[3]) * np.sin(ph), np.cos(x[3])])
        return (x[:3] / 100, _rotation(nn, yaw))

    def loss(x):
        return loss_images(render_pair(*unpack(x), obs, k, t, render=render), obs)

    x0 = np.clip(x0, np.array(bounds)[:, 0] + 1e-09, np.array(bounds)[:, 1] - 1e-09)
    if global_search:
        global_fit = differential_evolution(
            loss,
            bounds,
            x0=x0,
            rng=np.random.default_rng(0),
            popsize=8,
            maxiter=22,
            polish=False,
            tol=0.005,
        )
        if global_fit.fun < loss(x0):
            x0 = global_fit.x
    fit = minimize(
        loss,
        x0,
        method="Powell",
        bounds=bounds,
        options={"maxiter": 8, "xtol": 0.001, "ftol": 1e-06},
    )
    if loss(x0) < fit.fun:
        fit.x = x0
    pp, rr = unpack(fit.x)
    return (pp, rr, float(loss(fit.x)))


def stereo_photo_error(
    point, rotation, images, observations, k, t, *, render, return_samples=False
):
    """Compare RGB under the candidate CAD's visible stereo correspondence."""
    depth, mask = render.depth_patch(point, rotation, k[0], t[0], observations[0]["origin"])
    right_depth, right_mask = render.depth_patch(
        point, rotation, k[1], t[1], observations[1]["origin"]
    )
    yy, xx = np.mgrid[: render.side, : render.side]
    origin = observations[0]["origin"]
    pixels = np.stack([xx + origin[0] + 0.5, yy + origin[1] + 0.5, np.ones(xx.shape)], -1)
    camera = (pixels @ np.linalg.inv(k[0]).T) * depth[..., None]
    world = (camera - t[0][:3, 3]) @ t[0][:3, :3]
    second = world @ t[1][:3, :3].T + t[1][:3, 3]
    projected = second @ k[1].T
    uv = projected[..., :2] / projected[..., 2:]
    rx = (uv[..., 0] - observations[1]["origin"][0] - 0.5).astype(np.float32)
    ry = (uv[..., 1] - observations[1]["origin"][1] - 0.5).astype(np.float32)
    target_depth = cv2.remap(right_depth, rx, ry, cv2.INTER_LINEAR)
    target_mask = cv2.remap(right_mask, rx, ry, cv2.INTER_LINEAR)
    source = cv2.remap(
        images[0].astype(np.float32),
        (pixels[..., 0] - 0.5).astype(np.float32),
        (pixels[..., 1] - 0.5).astype(np.float32),
        cv2.INTER_LINEAR,
    )
    target = cv2.remap(
        images[1].astype(np.float32),
        (uv[..., 0] - 0.5).astype(np.float32),
        (uv[..., 1] - 0.5).astype(np.float32),
        cv2.INTER_LINEAR,
    )
    valid = (mask > 0.95) & (target_mask > 0.95) & (np.abs(target_depth - second[..., 2]) < 0.004)
    valid &= observations[0]["mask"] > 0.7
    error = np.minimum(np.abs(source - target), 20).mean(axis=-1)
    if return_samples:
        return error, valid
    if valid.sum() < 10:
        return None
    return float(error[valid].mean())


def observations(images, props, occluders, soft=True):
    out = []
    for rgb, p, occ in zip(images, props, occluders, strict=False):
        origin = np.floor(p["center"] - 32).astype(int)
        hard = crop_mask(p["mask"], origin) > 0
        image = np.stack([crop_mask(rgb[..., c], origin) for c in range(3)], axis=-1).astype(
            np.uint8
        )
        mask = soft_region(image, hard) if soft else hard.astype(float)
        ignore = crop_mask(occ, origin) > 0
        # Volume boundary antialiasing belongs to the foreground occluder too.
        ignore = cv2.dilate(ignore.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        weight = np.where(ignore & (mask < 0.05), 0.0, 1.0)
        out.append(dict(origin=origin, mask=mask, weight=weight))
    return out
