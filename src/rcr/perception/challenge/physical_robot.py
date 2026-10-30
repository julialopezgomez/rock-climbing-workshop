"""Static RGB robot foreground and partially occluded printed-label hypotheses."""

import itertools

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .physical_mapping import plane_warp


def foreground_box(image, background, camera, background_camera, wall):
    size = (round(wall["wall_width_m"] * 1000), round(wall["wall_height_m"] * 1000))
    first = cv2.warpPerspective(image, plane_warp(camera, wall, 0), size)
    second = cv2.warpPerspective(background, plane_warp(background_camera, wall, 0), size)
    gray_a, gray_b = [cv2.cvtColor(v, cv2.COLOR_RGB2GRAY) for v in [first, second]]
    hsv = cv2.cvtColor(first, cv2.COLOR_RGB2HSV)
    mask = (
        (cv2.absdiff(gray_a, gray_b) > 35) & (gray_a > 0) & (gray_b > 0) & (hsv[..., 1] < 120)
    ).astype(np.uint8)
    mask[:120] = 0
    mask[max(0, size[1] - 100) :] = 0
    mask[:, :50] = 0
    mask[:, max(0, size[0] - 50) :] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    candidates = [s for s in stats[1:] if s[4] > 300 and 120 < s[2] < 450 and s[3] > 150]
    if count < 2 or not candidates:
        raise ValueError("No compact robot foreground against the empty board")
    x, y, width, height, _ = max(candidates, key=lambda value: value[4])
    corners = np.array([[x, y], [x + width, y], [x + width, y + height], [x, y + height]], float)
    pixels = cv2.perspectiveTransform(corners[:, None], np.linalg.inv(plane_warp(camera, wall, 0)))[
        :, 0
    ]
    low = np.clip(np.floor(pixels.min(0) - 30), 0, np.array(image.shape[1::-1]) - 1).astype(int)
    high = np.clip(np.ceil(pixels.max(0) + 30), 0, np.array(image.shape[1::-1])).astype(int)
    return [int(low[0]), int(low[1]), int(high[0]), int(high[1])]


def ordered_quad(points):
    upper = points[np.argsort(points[:, 1])[:2]]
    lower = points[np.argsort(points[:, 1])[2:]]
    upper, lower = upper[np.argsort(upper[:, 0])], lower[np.argsort(lower[:, 0])]
    return np.array([upper[0], upper[1], lower[1], lower[0]], float)


def quadrilateral(hull):
    for epsilon in [0.02, 0.03, 0.04, 0.05, 0.06, 0.08]:
        candidate = cv2.approxPolyDP(hull, epsilon * cv2.arcLength(hull, True), True)[:, 0]
        if len(candidate) == 4:
            return ordered_quad(candidate)
    raise ValueError("The visible label does not supply a four-corner outline")


def label_candidates(patch, origin):
    hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    white = ((hsv[..., 1] < 90) & (hsv[..., 2] > 160)).astype(np.uint8)
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if (
            300 < area < 3000
            and 0.25 < width / height < 1.1
            and 0.25 < (y + height / 2) / patch.shape[0] < 0.7
        ):
            try:
                quad = quadrilateral(cv2.convexHull(contour))
            except ValueError:
                continue
            candidates.append((area, contour, quad, (x, y, width, height)))
    if not candidates:
        raise ValueError("No white torso-label candidate in the robot foreground")
    _, contour, outer, box = max(candidates, key=lambda value: value[0])
    region = np.zeros(white.shape, np.uint8)
    cv2.fillConvexPoly(region, cv2.convexHull(contour), 1)
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    x, y, width, height = box
    squares = []
    for threshold, left, end in itertools.product(
        [80, 100, 120], [0.05, 0.1, 0.15], [0.80, 0.84, 0.88]
    ):
        mask = (region & (gray < threshold)).astype(np.uint8)
        mask[: y + round(height * 0.30)] = 0
        mask[y + round(height * end) :] = 0
        mask[:, : x + round(width * left)] = 0
        mask[:, x + round(width * 0.95) :] = 0
        yy, xx = np.nonzero(mask)
        if len(xx) < 60:
            continue
        try:
            quad = (
                quadrilateral(cv2.convexHull(np.column_stack([xx, yy]).astype(np.int32))) + origin
            )
        except ValueError:
            continue
        if any(np.max(np.abs(quad - old)) < 0.1 for old in squares):
            continue
        squares.append(quad)
    return outer + origin, squares


def projection_matrix(camera):
    return np.asarray(camera["K"]) @ np.asarray(camera["T_camera_wall"])[:3]


def triangulate(corners, cameras):
    matrices = [projection_matrix(c) for c in cameras]
    homogeneous = cv2.triangulatePoints(*matrices, *[p.T.astype(float) for p in corners])
    points = (homogeneous[:3] / homogeneous[3]).T

    def residual(value):
        world = value.reshape(-1, 3)
        return np.concatenate(
            [
                (projection[:, :2] / projection[:, 2:] - observed).ravel()
                for matrix, observed in zip(matrices, corners, strict=True)
                for projection in [np.column_stack([world, np.ones(len(world))]) @ matrix.T]
            ]
        )

    fitted = least_squares(residual, points.ravel(), loss="soft_l1", f_scale=1)
    error = np.linalg.norm(residual(fitted.x).reshape(-1, 2), axis=1)
    return fitted.x.reshape(-1, 3), float(np.sqrt(np.mean(error**2)))


def fit_square(corners, cameras):
    points, _ = triangulate(corners, cameras)
    x = (points[1] - points[0] + points[2] - points[3]) / 2
    y = (points[0] - points[3] + points[1] - points[2]) / 2
    normal = np.cross(x, y)
    basis = np.column_stack(
        [x / np.linalg.norm(x), y / np.linalg.norm(y), normal / np.linalg.norm(normal)]
    )
    u, _, v = np.linalg.svd(basis)
    rotation = u @ v
    if np.linalg.det(rotation) < 0:
        raise ValueError("Label corner correspondence has a reflected orientation")
    size = (np.linalg.norm(x) + np.linalg.norm(y)) / 2
    initial = np.r_[Rotation.from_matrix(rotation).as_rotvec(), points.mean(0), np.log(size)]
    local = np.array([[-0.5, 0.5, 0], [0.5, 0.5, 0], [0.5, -0.5, 0], [-0.5, -0.5, 0]])

    def residual(value):
        world = (
            local * np.exp(value[-1]) @ Rotation.from_rotvec(value[:3]).as_matrix().T + value[3:6]
        )
        return np.concatenate(
            [
                (projected[:, :2] / projected[:, 2:] - observed).ravel()
                for camera, observed in zip(cameras, corners, strict=True)
                for projected in [
                    np.column_stack([world, np.ones(4)]) @ projection_matrix(camera).T
                ]
            ]
        )

    fitted = least_squares(residual, initial, max_nfev=80)
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_rotvec(fitted.x[:3]).as_matrix()
    transform[:3, 3] = fitted.x[3:6]
    error = np.linalg.norm(residual(fitted.x).reshape(-1, 2), axis=1)
    return dict(
        T_wall_marker=transform.tolist(),
        estimated_black_square_size_m=float(np.exp(fitted.x[-1])),
        reprojection_rms_px=float(np.sqrt(np.mean(error**2))),
    )


def label_hypothesis(images, boxes, cameras):
    outer, squares = [], []
    for image, (x0, y0, x1, y1) in zip(images, boxes, strict=True):
        border, variants = label_candidates(image[y0:y1, x0:x1], np.array([x0, y0]))
        outer.append(border)
        squares.append(variants)
    points, rms = triangulate(outer, cameras)
    fits = []
    for first, second in itertools.product(*squares):
        try:
            result = fit_square([first, second], cameras)
        except (ValueError, cv2.error):
            continue
        if 0.02 < result["estimated_black_square_size_m"] < 0.15:
            fits.append(dict(**result, corners=[first.tolist(), second.tolist()]))
    fits.sort(key=lambda value: value["reprojection_rms_px"])
    return dict(
        outer_corners=[c.tolist() for c in outer],
        outer_corners_wall_m=points.tolist(),
        outer_triangulation_rms_px=rms,
        centre_wall_m=points.mean(0).tolist(),
        square=fits[0] if fits else None,
        square_alternatives=fits[:5],
        marker_id=None,
        marker_id_decoded=False,
        orientation_verified=False,
        valid_for_control=False,
        assumptions=[
            "Independent recordings treated as one static robot pose",
            "Known hole pitch and board/grid origin",
            "Visible dark border belongs to the same square label",
        ],
        scope="Rope-occluded label geometry; no measured marker-to-trunk transform",
    )
