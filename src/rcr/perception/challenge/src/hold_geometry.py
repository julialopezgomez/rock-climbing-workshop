"""RGB volume fitting for an explicitly declared RCR wall geometry."""

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares, minimize

from .rcr_methods import _calibration
from .rcr_native_geometry import (
    FACES,
    flat_faces,
    paint_masks,
    project_many,
    soft_region,
    triangle_mask,
    wall_point,
)


@dataclass(frozen=True)
class SceneGeometry:
    width: float
    height: float
    volume_height: float

    @classmethod
    def from_calibration(cls, calibration):
        wall = calibration.get("wall", {})
        width = float(wall.get("width", calibration.get("wall_width_m", 0.8)))
        height = float(wall.get("height", calibration.get("wall_height_m", 1.4)))
        volume_height = float(calibration.get("volume_base_height_m", min(width * 0.1, height)))
        if (
            not np.isfinite([width, height, volume_height]).all()
            or min(width, height, volume_height) <= 0
        ):
            raise ValueError("Wall and volume dimensions must be finite positive metres")
        return cls(width, height, volume_height)

    def vertices(self, parameters):
        x, y, width, depth = parameters
        half = self.volume_height / 2
        return np.array(
            [
                [x - width / 2, y - half, 0],
                [x + width / 2, y - half, 0],
                [x, y + half, 0],
                [x, y, depth],
            ]
        )


def volume_candidates(rgb, k, t, geometry):
    faces, _ = flat_faces(rgb)
    candidates = []
    for face in faces:
        poly = face["polygon"]
        if len(poly) != 3:
            continue
        for apex in range(3):
            rest = [i for i in range(3) if i != apex]
            for reverse in (False, True):
                order = rest[::-1] if reverse else rest
                observed = poly[[*order, apex]]
                a, b = [wall_point(p, k, t) for p in observed[:2]]
                for fi in range(3):
                    if fi == 0:
                        x = (a[0] + b[0]) / 2
                        y = (a[1] + b[1]) / 2 + geometry.volume_height / 2
                        w = b[0] - a[0]
                        error = abs(a[1] - b[1])
                    elif fi == 1:
                        x = b[0]
                        y = (a[1] + b[1]) / 2
                        w = 2 * (a[0] - b[0])
                        error = abs(b[1] - a[1] - geometry.volume_height)
                    else:
                        x = a[0]
                        y = (a[1] + b[1]) / 2
                        w = 2 * (a[0] - b[0])
                        error = abs(a[1] - b[1] - geometry.volume_height)
                    if not (
                        0.04 * geometry.width < w < 0.28 * geometry.width
                        and error < 0.012
                        and -geometry.width / 2 - 0.05 < x < geometry.width / 2 + 0.05
                        and -0.05 < y < geometry.height + 0.05
                    ):
                        continue
                    v = np.array([x, y, w, 0.10])

                    def residual(z, *, fi=fi, observed=observed):
                        return (
                            project_many(geometry.vertices(z)[list(FACES[fi])], k, t) - observed
                        ).ravel()

                    fit = least_squares(
                        residual,
                        v,
                        bounds=(
                            [-geometry.width / 2 - 0.1, -0.1, geometry.width * 0.04, 0.005],
                            [
                                geometry.width / 2 + 0.1,
                                geometry.height + 0.1,
                                geometry.width * 0.28,
                                geometry.width * 0.28,
                            ],
                        ),
                        max_nfev=35,
                    )
                    loss = float(np.mean(residual(fit.x) ** 2))
                    if loss < 4:
                        candidates.append(
                            dict(
                                parameters=fit.x,
                                face_index=fi,
                                loss=loss,
                                color=face["color"],
                                observed=face,
                            )
                        )
    return candidates, faces


def estimate_volumes(images, calibration, geometry):
    k, t = _calibration(calibration)
    candidates, faces = volume_candidates(images[0], k[0], t[0], geometry)
    right_faces, _ = flat_faces(images[1])
    accepted = []
    for c in sorted(candidates, key=lambda c: c["loss"]):
        v = c["parameters"]
        fi = c["face_index"]
        obs0 = c["observed"]
        center = project_many(geometry.vertices(v)[list(FACES[fi])], k[1], t[1]).mean(0)
        choices = [
            f
            for f in right_faces
            if np.linalg.norm(f["color"].astype(float) - c["color"]) < 15
            and np.linalg.norm(f["center"] - center) < 35
        ]
        if not choices:
            continue
        obs1 = min(choices, key=lambda f: np.linalg.norm(f["center"] - center))
        obs = []
        for f in (obs0, obs1):
            yy, xx = np.where(f["mask"])
            origin = np.array([max(0, xx.min() - 12), max(0, yy.min() - 12)])
            end = np.array(
                [min(f["mask"].shape[1], xx.max() + 13), min(f["mask"].shape[0], yy.max() + 13)]
            )
            hard = f["mask"][origin[1] : end[1], origin[0] : end[0]]
            crop = images[len(obs)][origin[1] : end[1], origin[0] : end[0]]
            mask = soft_region(crop, hard)
            paint = np.logical_or.reduce(paint_masks(crop))
            weight = 1 - cv2.dilate(paint.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(float)
            obs.append((origin, mask, weight))

        def loss(z, *, fi=fi, obs=obs):
            score = 0
            for eye, (origin, mask, weight) in enumerate(obs):
                soft = triangle_mask(
                    project_many(geometry.vertices(z)[list(FACES[fi])], k[eye], t[eye]),
                    origin,
                    mask.shape,
                )
                score += (np.square(soft - mask) * weight).sum() / max((mask * weight).sum(), 1)
            return float(score)

        scale = np.array([0.01, 0.01, 0.01, 0.05])
        start = v.copy()
        fit = minimize(
            lambda d, loss=loss, start=start, scale=scale: loss(start + d * scale),
            np.zeros(4),
            method="Powell",
            bounds=[(-1, 1), (-1, 1), (-1, 1), (-1, 1)],
            options={"maxiter": 6, "xtol": 0.003, "ftol": 1e-5},
        )
        v = start + fit.x * scale
        score = loss(v)
        if score > 0.32 or not (
            geometry.width * 0.04 < v[2] < geometry.width * 0.28
            and 0.005 < v[3] < geometry.width * 0.28
        ):
            continue
        accepted.append(dict(parameters=v, loss=score, face_index=fi))
    result = []
    for candidate in sorted(accepted, key=lambda a: a["loss"]):
        if any(
            np.linalg.norm(candidate["parameters"][:2] - a["parameters"][:2]) < 0.03 for a in result
        ):
            continue
        result.append(candidate)
    return result


def refine_volume_faces(images, calibration, geometry, volumes):
    """Fit all identifiable flat faces jointly; hold paint is excluded from loss."""
    k, t = _calibration(calibration)
    faces = [flat_faces(image)[0] for image in images]
    for volume in volumes:
        parameters = volume["parameters"].copy()
        observations = []
        for eye, choices in enumerate(faces):
            for index, indices in enumerate(FACES):
                projected = project_many(
                    geometry.vertices(parameters)[list(indices)], k[eye], t[eye]
                )
                scores = []
                for face in choices:
                    yy, xx = np.where(face["mask"])
                    lo = np.maximum(
                        0,
                        np.floor(np.minimum(projected.min(0), [xx.min(), yy.min()]) - 5).astype(
                            int
                        ),
                    )
                    hi = np.minimum(
                        [images[eye].shape[1], images[eye].shape[0]],
                        np.ceil(np.maximum(projected.max(0), [xx.max(), yy.max()]) + 6).astype(int),
                    )
                    if np.any(hi <= lo):
                        continue
                    target = face["mask"][lo[1] : hi[1], lo[0] : hi[0]]
                    predicted = triangle_mask(projected, lo, target.shape) > 0.5
                    union = (target | predicted).sum()
                    iou = float((target & predicted).sum() / max(union, 1))
                    scores.append((iou, face, lo, hi))
                if not scores:
                    continue
                iou, face, lo, hi = max(scores, key=lambda item: item[0])
                if iou < 0.3:
                    continue
                image = images[eye][lo[1] : hi[1], lo[0] : hi[0]]
                target = face["mask"][lo[1] : hi[1], lo[0] : hi[0]]
                soft = soft_region(image, target)
                paint = np.logical_or.reduce(paint_masks(image)).astype(np.uint8)
                weight = 1.0 - cv2.dilate(paint, np.ones((3, 3), np.uint8))
                observations.append((eye, index, lo, soft, weight))
        if len(observations) < 2:
            continue

        def loss(parameters, *, observations=observations):
            if parameters[2] <= 0.01 or parameters[3] <= 0:
                return 1000.0
            total = 0.0
            for eye, index, origin, mask, weight in observations:
                projected = project_many(
                    geometry.vertices(parameters)[list(FACES[index])], k[eye], t[eye]
                )
                predicted = triangle_mask(projected, origin, mask.shape)
                total += float(
                    (np.square(predicted - mask) * weight).sum() / max((mask * weight).sum(), 1)
                )
            return total / len(observations)

        scale = np.array([0.01, 0.01, 0.01, 0.02])
        start = parameters.copy()
        fitted = minimize(
            lambda delta, loss=loss, start=start, scale=scale: loss(start + delta * scale),
            np.zeros(4),
            method="Powell",
            bounds=[(-3, 3)] * 4,
            options={"maxiter": 12, "xtol": 0.001, "ftol": 1e-6},
        )
        updated = start + fitted.x * scale
        if loss(updated) <= loss(start):
            volume.update(
                parameters=updated, joint_face_loss=loss(updated), observed_faces=len(observations)
            )
    return volumes
