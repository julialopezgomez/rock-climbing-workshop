"""Actual RGB masks + configured climbing CAD; no ground-truth masks or poses."""

import time

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares, minimize
from scipy.spatial.transform import Rotation

from rcr.perception.challenge.src.calibration import (
    calibrate_wall,
    calibrate_wall_boundary,
    intersect_wall_plane,
)

from .frame_conventions import R_WALL_WORLD, project, transform


def calibrate(images, packets, method="aruco", allow_partial=False):
    out = []
    for image, p in zip(images, packets, strict=False):
        k = np.asarray(p["K"], float)
        if np.any(p.get("distortion", [0] * 5)):
            raise ValueError("Undistort RGB and update intrinsics before this raster fitting path")
        try:
            if method == "aruco":
                result = calibrate_wall(image, k)
            else:
                # The floor reflects the plywood. Suppress its darker lobe
                # before fitting the board outline, using only RGB statistics.
                hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                wood = (
                    (hsv[..., 0] >= 15)
                    & (hsv[..., 0] <= 50)
                    & (hsv[..., 1] >= 25)
                    & (hsv[..., 1] < 125)
                    & (hsv[..., 2] > 45)
                )
                filtered = image.copy()
                if wood.any():
                    filtered[hsv[..., 2] < 0.6 * np.percentile(hsv[..., 2][wood], 80)] = 0
                result = calibrate_wall_boundary(filtered, k)
            out.append(
                dict(
                    K=k,
                    T_camera_wall=result.T_camera_wall,
                    distortion=np.zeros(5),
                    calibration=result.to_dict(),
                )
            )
        except ValueError:
            if not allow_partial:
                raise
            out.append(None)
    return out


def segment_robot(image, camera, return_region=False):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    polygon = project(
        np.array(
            [[0.025, 0.04, 0.04], [0.775, 0.04, 0.04], [0.775, 1.37, 0.04], [0.025, 1.37, 0.04]]
        ),
        camera,
    )
    region = np.zeros(image.shape[:2], np.uint8)
    cv2.fillConvexPoly(region, np.rint(polygon).astype(np.int32), 1)
    # Exclude calibrated corner plates; their white borders are not robot pixels.
    for x in [0.0, 0.664]:
        for y in [0.0, 1.264]:
            pts = np.array(
                [
                    [x - 0.01, y - 0.01, 0.001],
                    [x + 0.146, y - 0.01, 0.001],
                    [x + 0.146, y + 0.146, 0.001],
                    [x - 0.01, y + 0.146, 0.001],
                ]
            )
            cv2.fillConvexPoly(region, np.rint(project(pts, camera)).astype(np.int32), 0)
    raw = (hsv[..., 1] < 55) & (hsv[..., 2] > 18) & (region > 0)
    grown = cv2.dilate(raw.astype(np.uint8), np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(grown)
    if n < 2:
        raise ValueError("No RGB robot component")
    index = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    core = (labels == index) & raw
    if core.sum() < 40:
        raise ValueError("Robot mask too small")
    y, x = np.nonzero(core)
    bbox = [x.min(), y.min(), x.max(), y.max()]
    return (core, bbox, region) if return_region else (core, bbox)


def crop(array, origin, side):
    result = np.zeros((side, side), array.dtype)
    x, y = origin
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(array.shape[1], x + side), min(array.shape[0], y + side)
    if x1 > x0 and y1 > y0:
        result[y0 - y : y1 - y, x0 - x : x1 - x] = array[y0:y1, x0:x1]
    return result


def model_points(render, q):
    render.set_joints(q)
    points = []
    for gid in range(render.model.ngeom):
        if render.model.geom_group[gid] not in [0, 1, 2]:
            continue
        mid = render.model.geom_dataid[gid]
        if mid < 0:
            continue
        addr = render.model.mesh_vertadr[mid]
        count = render.model.mesh_vertnum[mid]
        p = render.model.mesh_vert[addr : addr + count : max(1, count // 25)]
        r = render.data.geom_xmat[gid].reshape(3, 3)
        t = render.data.geom_xpos[gid]
        points.append(p @ r.T + t)
    points = np.concatenate(points)
    return points[:: max(1, len(points) // 1500)]


def estimate(images, cameras, q, render, orientation=None, budget="normal", initial=None):
    started = time.perf_counter()
    side = render.side
    masks = []
    boxes = []
    observations = []
    for image, camera in zip(images, cameras, strict=False):
        mask, box, region = segment_robot(image, camera, return_region=True)
        masks.append(mask)
        boxes.append(np.asarray(box))
        origin = np.floor((np.array(box[:2]) + box[2:]) / 2 - side / 2).astype(int)
        observed = crop(mask.astype(np.float32), origin, side)
        # Colored holds can occlude the robot. Their pixels supply image-only
        # unknown-occlusion exclusions, without reading a hold segmentation label.
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        occluder = (hsv[..., 1] > 100) & (hsv[..., 0] < 30) & (hsv[..., 2] > 70)
        # Pixels deliberately excluded from segmentation (notably reflective
        # floor below the board) are unknown, not negative robot observations.
        valid = (~crop(occluder, origin, side)) & (crop(region, origin, side) > 0)
        observations.append(
            dict(origin=origin, mask=observed, soft=gaussian_filter(observed, 1.0), valid=valid)
        )
    points = model_points(render, q)
    nominal = R_WALL_WORLD @ Rotation.from_euler("z", 90, degrees=True).as_matrix()
    rotations = (
        [orientation]
        if orientation is not None
        else (
            [initial[:3, :3]]
            if initial is not None
            else [
                Rotation.from_rotvec(d).as_matrix() @ nominal
                for d in [
                    [0, 0, 0],
                    [0.3, 0, 0],
                    [-0.3, 0, 0],
                    [0, 0.3, 0],
                    [0, -0.3, 0],
                    [0, 0, 0.3],
                    [0, 0, -0.3],
                ]
            ]
        )
    )
    candidates = []
    for rotation in rotations:

        def bb_residual(pos, *, rotation=rotation):
            world = points @ rotation.T + pos
            residual = []
            for camera, box in zip(cameras, boxes, strict=False):
                uv = project(world, camera)
                predicted = np.r_[uv.min(0), uv.max(0)]
                residual.extend(predicted - box)
            return residual

        centers = [(b[:2] + b[2:]) / 2 for b in boxes]
        guesses = [
            intersect_wall_plane(np.array([uv]), c["K"], c["T_camera_wall"], 0.13)[0]
            for uv, c in zip(centers, cameras, strict=False)
        ]
        guess = np.mean(guesses, 0) - rotation @ ((points.min(0) + points.max(0)) / 2)
        bb = least_squares(
            bb_residual,
            np.clip(guess, [0.05, 0.1, 0.04], [0.75, 1.3, 0.3]),
            bounds=([0.05, 0.1, 0.04], [0.75, 1.3, 0.3]),
            max_nfev=60,
        )
        candidates.append(transform(rotation, bb.x))
    if initial is not None:
        candidates.insert(0, initial)
    render.set_joints(q)
    count = 0

    def score(t):
        nonlocal count
        total = 0.0
        for c, o in zip(cameras, observations, strict=False):
            pred = render.render(t, c, o["origin"])
            soft = gaussian_filter(pred, 1.0)
            total += float((((soft - o["soft"]) ** 2) * o["valid"]).sum() / max(o["mask"].sum(), 1))
        count += 1
        return total

    ranked = sorted([(score(t), t) for t in candidates], key=lambda x: x[0])
    winners = []
    for _, base in ranked[
        : 1 if orientation is not None or initial is not None else (2 if budget == "normal" else 4)
    ]:
        dimensions = 3 if orientation is not None else 6
        scale = np.array([0.025] * 3 + [0.25] * 3)[:dimensions]

        def unpack_delta(x, *, base=base, dimensions=dimensions, scale=scale):
            r = (
                base[:3, :3]
                if dimensions == 3
                else Rotation.from_rotvec(x[3:] * scale[3:]).as_matrix() @ base[:3, :3]
            )
            return transform(r, base[:3, 3] + x[:3] * scale[:3])

        fit = minimize(
            lambda x: score(unpack_delta(x)),
            np.zeros(dimensions),
            method="Powell",
            bounds=[(-2, 2)] * dimensions,
            options=dict(maxfev=350 if dimensions == 3 else 750, xtol=0.0015, ftol=0.0001),
        )
        winners.append((float(fit.fun), unpack_delta(fit.x)))
    loss, t = min(winners, key=lambda x: x[0])
    if loss > 0.25:
        raise ValueError(f"RGB/CAD disagreement too large ({loss:.3f}); pose rejected")
    return t, dict(
        loss=loss,
        render_evaluations=count,
        seconds=time.perf_counter() - started,
        orientation_source="measured IMU" if orientation is not None else "RGB multi-start",
        mask_pixels=[int(m.sum()) for m in masks],
        boxes=[b.tolist() for b in boxes],
        hypothesis_count=len(candidates),
        q_source="named measured encoders",
        image_only_segmentation=True,
    )
