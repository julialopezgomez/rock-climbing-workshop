"""Camera pose from a cropped perforated board, with explicit metric assumptions."""

import cv2
import numpy as np
from scipy.spatial import cKDTree


def hole_centres(rgb):
    """Find locally dark holes, then retain neighborhoods with repeated dots."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    background = cv2.GaussianBlur(gray, (0, 0), 2.0)
    contrast = background.astype(float) - gray
    mask = ((contrast > 16) & (hsv[..., 1] < 120) & (background > 60)).astype(np.uint8)
    count, _, stats, centres = cv2.connectedComponentsWithStats(mask)
    points = []
    for index in range(1, count):
        _, _, width, height, area = stats[index]
        if (
            2 <= area <= 32
            and 1 <= width <= 10
            and 1 <= height <= 10
            and max(width, height) <= 4 * min(width, height)
        ):
            points.append(centres[index])
    points = np.asarray(points, float).reshape(-1, 2)
    if len(points) < 100:
        raise ValueError("Too few dark boreholes for lattice calibration")
    density = cKDTree(points).query_ball_point(points, 25, return_length=True)
    return points[density >= 10]


def index_lattice(points, image_shape):
    """Grow an integer grid from a local patch; reject isolated edge indices."""
    mask = np.zeros(image_shape, np.uint8)
    for point in points:
        cv2.circle(mask, tuple(np.rint(point).astype(int)), 7, 1, -1)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    pixels = np.rint(points).astype(int)
    points = points[labels[pixels[:, 1], pixels[:, 0]] == component]
    low, high = points.min(0), points.max(0)
    target = low + (high - low) * [0.12, 0.12]
    seed = points[np.argmin(np.linalg.norm(points - target, axis=1))]
    delta = points - seed
    length = np.linalg.norm(delta, axis=1)
    horizontal = np.flatnonzero(
        (delta[:, 0] > 3) & (abs(delta[:, 1]) < 0.5 * delta[:, 0]) & (length < 14)
    )
    vertical = np.flatnonzero(
        (delta[:, 1] > 3) & (abs(delta[:, 0]) < 0.5 * delta[:, 1]) & (length < 14)
    )
    if not len(horizontal) or not len(vertical):
        raise ValueError("A local horizontal/vertical lattice basis is unavailable")
    a = delta[horizontal[np.argmin(length[horizontal])]]
    b = delta[vertical[np.argmin(length[vertical])]]
    homography = np.array([[a[0], b[0], seed[0]], [a[1], b[1], seed[1]], [0, 0, 1.0]])
    history = []
    for radius in [25, 35, 50, 70, 100, 150, 220, 330, 500, 1000]:
        coordinates = cv2.perspectiveTransform(points[:, None], np.linalg.inv(homography))[:, 0]
        indices = np.round(coordinates)
        good = (np.linalg.norm(coordinates - indices, axis=1) < 0.23) & (
            np.linalg.norm(points - seed, axis=1) < radius
        )
        if good.sum() < 15:
            raise ValueError("Insufficient local lattice support")
        homography, support = cv2.findHomography(
            indices[good], points[good], cv2.RANSAC, 0.8, maxIters=3000
        )
        if homography is None:
            raise ValueError("Lattice homography failed")
        history.append(dict(radius_px=radius, inliers=int(support.sum())))
    coordinates = cv2.perspectiveTransform(points[:, None], np.linalg.inv(homography))[:, 0]
    indices = np.round(coordinates)
    good = np.linalg.norm(coordinates - indices, axis=1) < 0.2
    points, indices = points[good], indices[good]
    _, support = cv2.findHomography(indices, points, cv2.RANSAC, 0.8, maxIters=4000)
    supported = indices[support.ravel() > 0]
    origin = []
    for axis in [0, 1]:
        values, counts = np.unique(supported[:, axis], return_counts=True)
        origin.append(values[counts >= max(8, 0.2 * np.median(counts))].min())
    indices -= origin
    return points, indices, history


def fit_grid_pose(pixels, indices, camera_matrix, geometry):
    """Fit a metric planar pose; image residuals do not validate the dimensions."""
    pitch = geometry["hole_pitch_m"]
    points = np.column_stack(
        [
            geometry["first_hole_left_m"] + indices[:, 0] * pitch,
            geometry["wall_height_m"] - geometry["first_hole_top_m"] - indices[:, 1] * pitch,
            np.zeros(len(pixels)),
        ]
    )
    _, inliers = cv2.findHomography(points[:, :2], pixels, cv2.RANSAC, 0.8, maxIters=4000)
    if inliers is None:
        raise ValueError("Grid correspondences do not define one plane")
    good = np.flatnonzero(inliers)
    if len(good) < 100 or np.ptp(indices[good], axis=0).min() < 20:
        raise ValueError("Grid support does not span enough of the board")
    success, rvec, tvec = cv2.solvePnP(
        points[good], pixels[good], camera_matrix, None, flags=cv2.SOLVEPNP_IPPE
    )
    if not success:
        raise ValueError("Grid camera pose failed")
    rvec, tvec = cv2.solvePnPRefineLM(points[good], pixels[good], camera_matrix, None, rvec, tvec)
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = cv2.Rodrigues(rvec)[0], tvec.ravel()
    position = -transform[:3, :3].T @ transform[:3, 3]
    if position[2] <= 0:
        raise ValueError("Grid pose places the camera behind the wall")
    projected = cv2.projectPoints(points, rvec, tvec, camera_matrix, None)[0][:, 0]
    error = np.linalg.norm(projected - pixels, axis=1)
    rms = float(np.sqrt(np.mean(error[good] ** 2)))
    if rms > 1.5:
        raise ValueError(f"Grid reprojection RMS {rms:.2f}px exceeds 1.5px")
    return (
        dict(
            K=camera_matrix.tolist(),
            T_camera_wall=transform.tolist(),
            camera_position_wall_m=position.tolist(),
            reprojection_rms_px=rms,
            reprojection_median_px=float(np.median(error[good])),
            inliers=len(good),
            grid_origin_verified=False,
            metric_geometry=geometry,
            scope="Image-estimated pose conditional on declared hole pitch and board/grid origin",
        ),
        points,
        good,
    )


def calibrate_perforated_board(rgb, camera_matrix, geometry):
    """Use the largest dense hole array; images are normalized to 720px height."""
    if min(geometry.values()) <= 0 or not np.isfinite(list(geometry.values())).all():
        raise ValueError("Board/grid dimensions must be finite positive metres")
    scale = 720 / rgb.shape[0]
    image = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    pixels, indices, history = index_lattice(hole_centres(image), image.shape[:2])
    pixels /= scale
    camera, points, good = fit_grid_pose(
        pixels, indices, np.asarray(camera_matrix, float), geometry
    )
    camera.update(hole_count=len(pixels), growth=history)
    return camera, dict(pixels=pixels, indices=indices, object_points=points, inliers=good)
