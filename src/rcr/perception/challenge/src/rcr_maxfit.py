"""Native-generator inverse graphics using RGB, calibration and public CAD.

No seed, saved prediction, scene directory, label, mask label or depth is an
inference input. The generator-specific profile knows the emissive paint family,
three local yaw values, wall plane, triangle-volume family and 100mm clearance.
"""

import time

import cv2
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.transform import Rotation

from .rcr_exact_fit import ExactRenderer, crop_mask
from .rcr_methods import _calibration, _project, _rotation, match_proposals
from .rcr_native_geometry import (
    FACES,
    estimate_volumes,
    flat_faces,
    paint_masks,
    soft_region,
    vertices,
    wall_point,
)

_RENDERER = None


def renderer():
    global _RENDERER
    if _RENDERER is None:
        _RENDERER = ExactRenderer(wall=True)
    return _RENDERER


def close_renderer():
    global _RENDERER
    if _RENDERER is not None:
        _RENDERER.close()
        _RENDERER = None


def proposals(rgb):
    result = []
    x = rgb.astype(float)
    rr, gg, bb = np.moveaxis(x, -1, 0)
    broad = paint_masks(rgb)
    # Emissive holds reach the display ceiling under direct light; shaded
    # pixels stay near their public base paint. Grow these reliable cores only
    # locally, preventing a similarly colored wall from joining the object.
    palette = np.array([[242, 85, 78], [89, 199, 123], [99, 137, 237]])
    cores = [
        broad[g] & ((x[..., g] >= 250) | (np.max(np.abs(x - palette[g]), axis=-1) < 12))
        for g in range(3)
    ]
    for group, wide in enumerate(broad):
        mask = wide & (cv2.dilate(cores[group].astype(np.uint8), np.ones((7, 7), np.uint8)) > 0)
        n, labels, stats, centers = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if not (8 <= area <= 1300 and 2 <= w <= 40 and 2 <= h <= 40):
                continue
            m = labels == i
            if (m & cores[group]).sum() < max(3, 0.08 * area):
                continue
            result.append(
                dict(
                    mask=m,
                    center=centers[i],
                    area=int(area),
                    hue_group=group,
                    bbox=[int(x), int(y), int(w), int(h)],
                    color=np.median(rgb[m], axis=0),
                )
            )
    # The native sampler enforces 100mm anchor clearance. Disconnected same-
    # paint fragments within 15 image pixels can be reunited without filling
    # their cavity/occluded pixels as morphological closing would.
    merged = []
    for p in sorted(result, key=lambda a: -a["area"]):
        other = next(
            (
                a
                for a in merged
                if a["hue_group"] == p["hue_group"]
                and np.linalg.norm(a["center"] - p["center"]) < 15
            ),
            None,
        )
        if other is None:
            merged.append(p)
            continue
        mask = other["mask"] | p["mask"]
        y, x = np.where(mask)
        if x.max() - x.min() > 38 or y.max() - y.min() > 38:
            merged.append(p)
            continue
        other.update(
            mask=mask,
            center=np.array([x.mean(), y.mean()]),
            area=len(x),
            bbox=[
                int(x.min()),
                int(y.min()),
                int(x.max() - x.min() + 1),
                int(y.max() - y.min() + 1),
            ],
        )
    return sorted(merged, key=lambda a: (a["hue_group"], a["bbox"][1], a["bbox"][0]))


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


def loss_images(softs, obs):
    return sum(
        float((np.square(s - o["mask"]) * o["weight"]).sum() / max(o["mask"].sum(), 1))
        for s, o in zip(softs, obs, strict=False)
    )


def render_pair(p, r, obs, k, t):
    return [
        renderer().render_soft(p, r, ki, ti, o["origin"])
        for o, ki, ti in zip(obs, k, t, strict=False)
    ]


def fit_plane(initial, r, normal, plane_point, obs, k, t, rounds=3):
    """Fast exact rerender/translation alternation; fixed supporting plane."""
    p = initial.copy()
    n = np.asarray(normal)
    p[2] = plane_point[2] - (n[0] * (p[0] - plane_point[0]) + n[1] * (p[1] - plane_point[1])) / n[2]
    center = r @ renderer().center
    yy, xx = np.mgrid[:64, :64].astype(np.float32)
    for it in range(rounds):
        softs = render_pair(p, r, obs, k, t)
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
            options={"maxiter": 6, "xtol": 0.001, "ftol": 1e-6},
        )
        q = unpack(fit.x)
        if loss_images(render_pair(q, r, obs, k, t), obs) > loss_images(softs, obs) and it > 0:
            break
        p = q
    return p, r, loss_images(render_pair(p, r, obs, k, t), obs)


def plane_initial(props, k, t, r, n, a):
    center = r @ renderer().center
    points = []
    for prop, ki, ti in zip(props, k, t, strict=False):
        origin = -ti[:3, :3].T @ ti[:3, 3]
        ray = ti[:3, :3].T @ np.linalg.solve(ki, np.r_[prop["center"] + 0.5, 1.0])
        den = n @ ray
        if abs(den) > 1e-8:
            points.append(origin + ray * (n @ (a + center - origin)) / den - center)
    return np.mean(points, axis=0)


def fit_translation(initial, r, obs, k, t, rounds=2):
    p = initial.copy()
    p[2] = np.clip(p[2], 0, 0.2)
    center = r @ renderer().center
    yy, xx = np.mgrid[:64, :64].astype(np.float32)
    for _it in range(rounds):
        softs = render_pair(p, r, obs, k, t)
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
            bounds=[(-3, 3), (-3, 3), (-p[2] / 0.05, (0.2 - p[2]) / 0.05)],
            options={"maxiter": 6, "xtol": 0.002, "ftol": 1e-5},
        )
        p = unpack(fit.x)
    return p, r, loss_images(render_pair(p, r, obs, k, t), obs)


def refine_inclined(p, r, obs, k, t, global_search=False):
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
        (max(0, p[2] * 100 - 3), min(20, p[2] * 100 + 3)),
        (max(0.5, theta - 0.2), min(np.arccos(0.25), theta + 0.2)),
    ]
    if not bottom:
        bounds.append((max(0.4, phi - 0.15), min(2.75, phi + 0.15)))

    def unpack(x):
        ph = -np.pi / 2 if bottom else x[4]
        nn = np.array([np.sin(x[3]) * np.cos(ph), np.sin(x[3]) * np.sin(ph), np.cos(x[3])])
        return x[:3] / 100, _rotation(nn, yaw)

    def loss(x):
        return loss_images(render_pair(*unpack(x), obs, k, t), obs)

    x0 = np.clip(x0, np.array(bounds)[:, 0] + 1e-9, np.array(bounds)[:, 1] - 1e-9)
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
        options={"maxiter": 8, "xtol": 0.001, "ftol": 1e-6},
    )
    if loss(x0) < fit.fun:
        fit.x = x0
    pp, rr = unpack(fit.x)
    return pp, rr, float(loss(fit.x))


def search_free(initial, obs, k, t):
    candidates = []
    for phi in [-np.pi / 2, np.radians(30), np.radians(45), np.radians(135), np.radians(150)]:
        for theta in np.radians([45, 65, 75]):
            n = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
            for yaw in [0.0, -np.pi / 4, np.pi / 4]:
                r = _rotation(n, yaw)
                q = initial.copy()
                q[2] = 0.025
                q -= r @ renderer().center
                p, r, loss = fit_translation(q, r, obs, k, t)
                candidates.append((loss, p, r, "free_source", n, yaw))
    result = []
    for _, p, r, _, n, yaw in sorted(candidates, key=lambda c: c[0])[:3]:
        # The public bottom face has n.x=0; side-face azimuth is constrained
        # by the released 80mm base height and 80..200mm base width.
        theta = np.arccos(n[2])
        phi = np.arctan2(n[1], n[0])
        bottom = phi < 0
        x0 = np.array(
            [p[0] * 100, p[1] * 100, p[2] * 100, theta]
            if bottom
            else [p[0] * 100, p[1] * 100, p[2] * 100, theta, phi]
        )
        bounds = [
            (p[0] * 100 - 2, p[0] * 100 + 2),
            (p[1] * 100 - 2, p[1] * 100 + 2),
            (0, 20),
            (0.5, np.arccos(0.25)),
        ]
        if not bottom:
            bounds.append(
                (np.arctan(0.08 / 0.16), np.arctan(0.2 / 0.16))
                if phi < np.pi / 2
                else (np.pi - np.arctan(0.2 / 0.16), np.pi - np.arctan(0.08 / 0.16))
            )

        def unpack(x, *, bottom=bottom, yaw=yaw):
            ph = -np.pi / 2 if bottom else x[4]
            nn = np.array([np.sin(x[3]) * np.cos(ph), np.sin(x[3]) * np.sin(ph), np.cos(x[3])])
            return x[:3] / 100, _rotation(nn, yaw)

        def loss(x):
            return loss_images(render_pair(*unpack(x), obs, k, t), obs)

        fit = minimize(
            loss,
            x0,
            method="Powell",
            bounds=bounds,
            options={"maxiter": 8, "xtol": 0.003, "ftol": 1e-5},
        )
        p, r = unpack(fit.x)
        result.append((float(fit.fun), p, r, "free_source_refined"))
    return result


def encode(p, r, color):
    e = Rotation.from_matrix(r).as_euler("XYZ")
    return dict(
        type="microban_hold",
        color="#" + "".join(f"{int(c):02X}" for c in color),
        position=dict(zip("xyz", map(float, p), strict=False)),
        orientation=dict(zip(("roll", "pitch", "yaw"), map(float, e), strict=False)),
    )


def infer(
    images,
    calibration,
    soft=True,
    use_volumes=True,
    occlusion=True,
    use_search=True,
    render_volumes=True,
    deep_search=False,
):
    global _RENDERER
    start = time.perf_counter()
    k, t = _calibration(calibration)
    ps = [proposals(rgb) for rgb in images]
    occ = [flat_faces(rgb)[1] if occlusion else np.zeros(rgb.shape[:2], bool) for rgb in images]
    volumes = estimate_volumes(images, calibration) if use_volumes else []
    close_renderer()
    _RENDERER = ExactRenderer(
        wall=True, volumes=[v["parameters"] for v in volumes] if render_volumes else None
    )
    if occlusion:
        # Estimated volume outlines complete shadowed/nonuniform face patches.
        from .rcr_native_geometry import project_many

        for eye in range(2):
            for v in volumes:
                for face in FACES:
                    poly = project_many(vertices(v["parameters"])[list(face)], k[eye], t[eye])
                    cv2.fillConvexPoly(
                        occ[eye].view(np.uint8), np.rint(poly - 0.5).astype(np.int32), 1
                    )
    pairs = match_proposals(*ps, k, t)
    # Tiny partial instances need no physical-width or area-ratio gate. Pair
    # unused same-paint fragments by their wall-plane image locations.
    used = [{p[eye] for p in pairs} for eye in range(2)]
    for i, a in enumerate(ps[0]):
        if i in used[0]:
            continue
        p = wall_point(a["center"] + 0.5, k[0], t[0])
        uv, _ = _project(p, k[1], t[1])
        choices = [
            (np.linalg.norm(b["center"] + 0.5 - uv), j)
            for j, b in enumerate(ps[1])
            if j not in used[1] and a["hue_group"] == b["hue_group"]
        ]
        if choices and min(choices)[0] < 10:
            error, j = min(choices)
            pairs.append((i, j, p, error))
            used[1].add(j)
    holds = {}
    details = []
    for i, j, _centroid, _epi in pairs:
        props = [ps[0][i], ps[1][j]]
        obs = observations(images, props, occ, soft)
        # Wall-plane initialization is stable even when centroid disparity is
        # biased by asymmetric visibility in the two eyes.
        initial = np.mean(
            [wall_point(p["center"] + 0.5, ki, ti) for p, ki, ti in zip(props, k, t, strict=False)],
            axis=0,
        )
        candidates = []
        for yaw in [0.0, -np.pi / 4, np.pi / 4]:
            r = _rotation([0, 0, 1], yaw)
            p0 = plane_initial(props, k, t, r, np.array([0.0, 0, 1]), np.zeros(3))
            p, r, loss = fit_plane(p0, r, [0, 0, 1], [0, 0, 0], obs, k, t)
            candidates.append((loss, p, r, "wall"))
        for v in volumes:
            verts = vertices(v["parameters"])
            if np.linalg.norm(initial[:2] - v["parameters"][:2]) > 0.18:
                continue
            for fi, face in enumerate(FACES):
                a, b, c = verts[list(face)]
                n = np.cross(b - a, c - a)
                n /= np.linalg.norm(n)
                if n[2] < 0:
                    n = -n
                if n[2] < 0.25:
                    continue
                # A hold on this volume is in front of its own faces. Use its
                # observed full mask to rank support-plane alternatives.
                plain = observations(images, props, [np.zeros_like(o) for o in occ], soft)
                for yaw in [0.0, -np.pi / 4, np.pi / 4]:
                    r = _rotation(n, yaw)
                    p0 = plane_initial(props, k, t, r, n, a)
                    coeff = np.linalg.lstsq(np.stack([b - a, c - a], axis=1), p0 - a, rcond=None)[0]
                    weights = np.maximum(np.r_[1 - coeff.sum(), coeff], 0.02)
                    weights /= weights.sum()
                    p0 = weights @ np.array([a, b, c])
                    p, r, loss = fit_plane(p0, r, n, a, plain, k, t)
                    bary = np.linalg.lstsq(np.stack([b - a, c - a], axis=1), p - a, rcond=None)[0]
                    if min(bary) < -0.05 or bary.sum() > 1.05:
                        continue
                    candidates.append((loss, p, r, f"volume_face_{fi}"))
        # The RGB triangle estimate is an initializer, not exact measured CAD.
        # Release its plane constraint for the two best inclined candidates.
        inclined = sorted([c for c in candidates if c[3].startswith("volume")], key=lambda c: c[0])
        for _, p, r, family in inclined[:3]:
            plain = observations(images, props, [np.zeros_like(o) for o in occ], soft)
            pp, rr, ll = fit_translation(p, r, plain, k, t, rounds=3)
            candidates.append((ll, pp, rr, family + "_translation_refined"))
            pp, rr, ll = refine_inclined(pp, rr, plain, k, t)
            candidates.append((ll, pp, rr, family + "_pose_refined"))
        if use_search and min(c[0] for c in candidates) > 0.10:
            plain = observations(images, props, [np.zeros_like(o) for o in occ], soft)
            candidates.extend(search_free(initial, plain, k, t))
        if deep_search:
            inclined = sorted([c for c in candidates if c[3] != "wall"], key=lambda c: c[0])
            if inclined and inclined[0][0] < 0.35:
                _, pp, rr, _ = inclined[0]
                plain = observations(images, props, [np.zeros_like(o) for o in occ], soft)
                pp, rr, ll = refine_inclined(pp, rr, plain, k, t, global_search=True)
                candidates.append((ll, pp, rr, "inclined_global_refined"))
        loss, p, r, family = min(candidates, key=lambda c: c[0])
        valid = bool(
            -0.405 <= p[0] <= 0.405
            and -0.005 <= p[1] <= 1.405
            and -0.001 <= p[2] <= 0.23
            and loss < 0.4
        )
        entry = dict(
            left_proposal=i,
            right_proposal=j,
            loss=float(loss),
            family=family,
            accepted=valid,
            candidate_losses=[dict(loss=float(c[0]), family=c[3]) for c in candidates],
        )
        details.append(entry)
        if valid:
            key = f"hold_{len(holds) + 1}"
            entry["id"] = key
            paint = np.array([[242, 85, 78], [89, 199, 123], [99, 137, 237]])[props[0]["hue_group"]]
            holds[key] = encode(p, r, paint)
    return holds, dict(
        method="native_inverse_graphics",
        objects=details,
        volumes=[dict(parameters=v["parameters"].tolist(), loss=v["loss"]) for v in volumes],
        runtime_s=time.perf_counter() - start,
        inputs=(
            "stereo RGB, public calibration/CAD and source geometry "
            "priors; no labels or cached poses"
        ),
    )
