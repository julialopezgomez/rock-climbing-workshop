"""Image-only wall calibration, in metres and OpenCV camera axes.

The returned transform maps wall points (X right, Y up, Z outward) to
camera points (X right, Y down, Z forward). Intrinsics must come from a
camera calibration. No nominal camera pose or simulator state is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


class CalibrationError(ValueError):
    """An image does not supply a usable, geometrically consistent board pose."""


@dataclass
class WallCalibration:
    T_camera_wall: np.ndarray
    marker_ids: list[int]
    reprojection_rms_px: float
    corners_px: dict[int, np.ndarray]
    inlier_count: int

    @property
    def camera_position_wall_m(self):
        T = self.T_camera_wall
        return -T[:3, :3].T @ T[:3, 3]

    def to_dict(self):
        return {
            "T_camera_wall": self.T_camera_wall.tolist(),
            "camera_position_wall_m": self.camera_position_wall_m.tolist(),
            "marker_ids": self.marker_ids,
            "reprojection_rms_px": self.reprojection_rms_px,
            "inlier_count": self.inlier_count,
        }


def default_marker_corners(
    wall_width_m=0.8, wall_height_m=1.4, code_size_m=0.1, white_border_m=0.018, face_z_m=0.001
):
    """Known black-code corners, TL/TR/BR/BL, for the released board.

    This is measured board geometry, not a camera-pose prior. Use a measured
    face_z_m and code size on hardware, never the outer white plate size.
    """
    plate = code_size_m + 2 * white_border_m
    result = {}
    for mid in range(1, 5):
        x0 = 0.0 if mid in (1, 3) else wall_width_m - plate
        y0 = wall_height_m - plate if mid in (1, 2) else 0.0
        x0 += white_border_m
        y0 += white_border_m
        result[mid] = np.array(
            [
                [x0, y0 + code_size_m, face_z_m],
                [x0 + code_size_m, y0 + code_size_m, face_z_m],
                [x0 + code_size_m, y0, face_z_m],
                [x0, y0, face_z_m],
            ],
            dtype=np.float64,
        )
    return result


def detect_markers(rgb: np.ndarray):
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError("RGB image must have shape HxWx3 and dtype uint8")
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 4
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), params
    )
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    if ids is None:
        return {}
    # Duplicate IDs (e.g. floor reflections) are ambiguous; prefer the larger
    # image quadrilateral, which is the actual board in the released scenes.
    found = {}
    for corners_i, mid in zip(corners, ids.ravel(), strict=False):
        p = corners_i.reshape(4, 2).astype(np.float64)
        if int(mid) not in found or cv2.contourArea(p.astype(np.float32)) > cv2.contourArea(
            found[int(mid)].astype(np.float32)
        ):
            found[int(mid)] = p
    return found


def calibrate_wall(
    rgb, K, distortion=None, marker_corners_wall_m=None, min_markers=2, max_reprojection_rms_px=2.5
):
    """Detect wall markers and jointly solve their camera pose.

    A minimum of two known markers avoids a weak single-marker planar pose.
    Missing/mismatched markers cause a reported failure, not a fallback to a
    cached ground-truth pose. Distortion follows OpenCV's coefficient order.
    """
    K = np.asarray(K, dtype=np.float64)
    if K.shape != (3, 3) or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0:
        raise ValueError("K must be a finite 3x3 camera matrix with positive focal lengths")
    distortion = np.zeros(5) if distortion is None else np.asarray(distortion, np.float64)
    board = (
        default_marker_corners()
        if marker_corners_wall_m is None
        else {
            int(mid): np.asarray(corners, np.float64)
            for mid, corners in marker_corners_wall_m.items()
        }
    )
    corners = detect_markers(rgb)
    ids = sorted(set(corners) & set(board))
    if len(ids) < min_markers:
        raise CalibrationError(f"only {len(ids)} known wall markers detected; need {min_markers}")
    obj = np.concatenate([board[mid] for mid in ids])
    img = np.concatenate([corners[mid] for mid in ids])
    # IPPE uses all board corners to initialize this planar pose. Small
    # same-marker minimal RANSAC subsets are poorly conditioned at distance;
    # use RANSAC only when the initial all-marker fit contains outliers.
    success, rvec, tvec = cv2.solvePnP(obj, img, K, distortion, flags=cv2.SOLVEPNP_IPPE)
    initial_projection = cv2.projectPoints(obj, rvec, tvec, K, distortion)[0].reshape(-1, 2)
    residuals = np.linalg.norm(initial_projection - img, axis=1)
    inliers = np.flatnonzero(residuals <= max_reprojection_rms_px)[:, None]
    if not success or len(inliers) < len(obj) * 0.75:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj,
            img,
            K,
            distortion,
            iterationsCount=300,
            reprojectionError=max_reprojection_rms_px,
            confidence=0.999,
            flags=cv2.SOLVEPNP_IPPE,
        )
    if not success or inliers is None or len(inliers) < 6:
        raise CalibrationError("joint marker PnP failed")
    ix = inliers.ravel()
    rvec, tvec = cv2.solvePnPRefineLM(obj[ix], img[ix], K, distortion, rvec, tvec)
    R = cv2.Rodrigues(rvec)[0]
    if np.any((obj @ R.T + tvec.ravel())[:, 2] <= 0):
        raise CalibrationError("PnP places board behind camera")
    projected = cv2.projectPoints(obj[ix], rvec, tvec, K, distortion)[0].reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum((projected - img[ix]) ** 2, axis=1))))
    if rms > max_reprojection_rms_px:
        raise CalibrationError(f"marker reprojection RMS {rms:.3f}px exceeds limit")
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, tvec.ravel()
    used_ids = sorted({ids[i // 4] for i in ix})
    if len(used_ids) < min_markers:
        raise CalibrationError(f"only {len(used_ids)} wall markers remain after outlier rejection")
    return WallCalibration(T, used_ids, rms, corners, len(ix))


def intersect_wall_plane(pixels, K, T_camera_wall, plane_z_m=0.0, distortion=None):
    """Intersect image rays with a known wall-parallel plane.

    A cup socket plane is at Z=.026m; a surface mask centroid is only an
    approximate ray to a socket. This function never supplies object depth.
    """
    pixels = np.asarray(pixels, np.float64).reshape(-1, 1, 2)
    K, T = np.asarray(K, np.float64), np.asarray(T_camera_wall, np.float64)
    rays_camera = np.c_[
        cv2.undistortPoints(pixels, K, distortion).reshape(-1, 2), np.ones(len(pixels))
    ]
    R, t = T[:3, :3], T[:3, 3]
    origin = -R.T @ t
    rays_wall = rays_camera @ R
    with np.errstate(divide="ignore", invalid="ignore"):
        distance = (plane_z_m - origin[2]) / rays_wall[:, 2]
    xyz = origin + distance[:, None] * rays_wall
    xyz[~np.isfinite(distance) | (distance <= 0)] = np.nan
    return xyz


def project_wall(points, K, T_camera_wall, distortion=None):
    T = np.asarray(T_camera_wall, np.float64)
    rvec = cv2.Rodrigues(T[:3, :3])[0]
    return cv2.projectPoints(
        np.asarray(points, np.float64), rvec, T[:3, 3], np.asarray(K, np.float64), distortion
    )[0].reshape(-1, 2)


def detect_wall_texture_core(rgb):
    """Find the large perforated board region, without reading ArUco patterns.

    This identifies the visible hole-rich interior, not its physical corners.
    It is tuned to the released board's dense small-hole texture; a second,
    narrower perforated board may also be present behind it.
    """
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError("RGB image must have shape HxWx3 and dtype uint8")
    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    dark_spots = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 5
    )
    _, _, stats, centroids = cv2.connectedComponentsWithStats(dark_spots)
    small = (
        (stats[:, cv2.CC_STAT_AREA] >= 1)
        & (stats[:, cv2.CC_STAT_AREA] <= 30)
        & (stats[:, cv2.CC_STAT_WIDTH] <= 7)
        & (stats[:, cv2.CC_STAT_HEIGHT] <= 7)
    )
    small[0] = False
    density = np.zeros((height, width), dtype=np.float32)
    centers = centroids[small].astype(int)
    density[centers[:, 1], centers[:, 0]] = 1.0
    density = cv2.GaussianBlur(density, (0, 0), width / 100)
    size = max(3, int(width * 0.025) | 1)
    kernel = np.ones((size, size), np.uint8)
    for threshold in (0.025, 0.020, 0.016):
        dense = (density > threshold).astype(np.uint8) * 255
        dense = cv2.morphologyEx(dense, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(dense, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            x, y, box_width, box_height = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if (
                area > width * height * 0.06
                and box_height > height * 0.55
                and 0.28 < box_width / box_height < 0.85
            ):
                candidates.append((area, contour, (x, y, box_width, box_height)))
        if candidates:
            _, contour, bbox = max(candidates, key=lambda item: item[0])
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
            return mask, bbox
    raise CalibrationError("no large perforated board region found")


def detect_wall_corner_squares(rgb):
    """Locate four unlabeled 100 mm black squares using the board texture as context.

    This uses only each square's outer silhouette and its position relative to
    the board. It never reads the internal ArUco pattern or decodes an ID.
    The supplied photos show all four squares; a hidden corner is rejected.
    """
    _, (bx, by, bw, bh) = detect_wall_texture_core(rgb)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    black = (gray < 100).astype(np.uint8) * 255
    contours, _ = cv2.findContours(black, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    groups = {name: [] for name in ("TL", "TR", "BR", "BL")}
    for contour in contours:
        area = cv2.contourArea(contour)
        if not width * height * 0.0003 < area < width * height * 0.02:
            continue
        polygon = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue
        x, y, box_width, box_height = cv2.boundingRect(polygon)
        cx, cy = x + box_width / 2, y + box_height / 2
        if not 0.6 < box_width / box_height < 1.5:
            continue
        if not bx - bw * 0.25 < cx < bx + bw * 1.25:
            continue
        if cy < by + bh * 0.28:
            role = "TL" if cx < bx + bw * 0.5 else "TR"
        elif cy > by + bh * 0.7:
            role = "BL" if cx < bx + bw * 0.5 else "BR"
        else:
            continue
        groups[role].append((area, polygon.reshape(4, 2).astype(np.float64)))
    if any(not candidates for candidates in groups.values()):
        missing = [role for role, candidates in groups.items() if not candidates]
        raise CalibrationError(f"missing visible wall corner squares: {missing}")
    squares = {}
    for role, candidates in groups.items():
        polygon = max(candidates, key=lambda candidate: candidate[0])[1]
        upper = polygon[np.argsort(polygon[:, 1])[:2]]
        lower = polygon[np.argsort(polygon[:, 1])[2:]]
        squares[role] = np.array(
            [
                upper[np.argmin(upper[:, 0])],
                upper[np.argmax(upper[:, 0])],
                lower[np.argmax(lower[:, 0])],
                lower[np.argmin(lower[:, 0])],
            ]
        )
    return squares


def calibrate_wall_corner_squares(
    rgb,
    K,
    distortion=None,
    wall_width_m=0.8,
    wall_height_m=1.4,
    code_size_m=0.1,
    white_border_m=0.018,
    code_face_z_m=0.0,
    max_reprojection_rms_px=3.0,
):
    """Fit the wall using four anonymous black-square silhouettes.

    The 136 mm plates touch the wall's outer corners. The known 18 mm white
    border locates each 100 mm black square relative to those corners. This
    is geometric use of visible print, not ArUco detection or ID decoding.
    Returns the calibration and the four observed square corner arrays.
    """
    K = np.asarray(K, dtype=np.float64)
    if K.shape != (3, 3) or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0:
        raise ValueError("K must be a finite 3x3 camera matrix with positive focal lengths")
    if (
        wall_width_m <= 0
        or wall_height_m <= 0
        or code_size_m <= 0
        or white_border_m < 0
        or code_size_m + 2 * white_border_m >= min(wall_width_m, wall_height_m)
    ):
        raise ValueError("invalid wall or corner-square dimensions")
    distortion = np.zeros(5) if distortion is None else np.asarray(distortion, np.float64)
    squares = detect_wall_corner_squares(rgb)
    plate = code_size_m + 2 * white_border_m
    object_points, image_points = [], []
    for role in ("TL", "TR", "BR", "BL"):
        x0 = 0.0 if role in ("TL", "BL") else wall_width_m - plate
        y0 = wall_height_m - plate if role in ("TL", "TR") else 0.0
        x0 += white_border_m
        y0 += white_border_m
        object_points.extend(
            [
                [x0, y0 + code_size_m, code_face_z_m],
                [x0 + code_size_m, y0 + code_size_m, code_face_z_m],
                [x0 + code_size_m, y0, code_face_z_m],
                [x0, y0, code_face_z_m],
            ]
        )
        image_points.extend(squares[role])
    obj = np.asarray(object_points, np.float64)
    img = np.asarray(image_points, np.float64)
    success, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        obj, img, K, distortion, flags=cv2.SOLVEPNP_IPPE
    )
    if not success:
        raise CalibrationError("corner-square PnP failed")
    candidates = []
    for rvec, tvec in zip(rvecs, tvecs, strict=False):
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, distortion, rvec, tvec)
        R = cv2.Rodrigues(rvec)[0]
        if np.any((obj @ R.T + tvec.ravel())[:, 2] <= 0):
            continue
        if (-R.T @ tvec.ravel())[2] <= 0:
            continue
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, tvec.ravel()
        residual = project_wall(obj, K, T, distortion) - img
        rms = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
        candidates.append((rms, T))
    if not candidates:
        raise CalibrationError("no valid front-facing wall pose from corner squares")
    rms, T = min(candidates, key=lambda candidate: candidate[0])
    if rms > max_reprojection_rms_px:
        raise CalibrationError(f"corner-square reprojection RMS {rms:.2f}px exceeds limit")
    wall = np.array(
        [
            [0, wall_height_m, 0],
            [wall_width_m, wall_height_m, 0],
            [wall_width_m, 0, 0],
            [0, 0, 0],
        ],
        dtype=np.float64,
    )
    board_corners = project_wall(wall, K, T, distortion)
    return WallCalibration(T, [], rms, {0: board_corners}, len(obj)), squares


def calibrate_wall_corners(
    corners_px, K, image_size, distortion=None, wall_width_m=0.8, wall_height_m=1.4
):
    """Estimate wall-to-camera pose from measured outer board corners, without markers.

    Corners are TL, TR, BR, BL in pixels. The challenge wall frame has its
    origin at bottom-left, X right, Y up, and Z outward. Four fitted points
    cannot independently establish localization accuracy; check held-out
    grid holes or a second view before using this pose for a robot.
    """
    corners = np.asarray(corners_px, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    width, height = image_size
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        raise ValueError("corners_px must be four finite TL, TR, BR, BL pixel pairs")
    if K.shape != (3, 3) or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0:
        raise ValueError("K must be a finite 3x3 camera matrix with positive focal lengths")
    if width <= 0 or height <= 0 or wall_width_m <= 0 or wall_height_m <= 0:
        raise ValueError("image and wall dimensions must be positive")
    if np.any(corners < 0) or np.any(corners[:, 0] >= width) or np.any(corners[:, 1] >= height):
        raise CalibrationError("board corners must lie inside the image")
    if not cv2.isContourConvex(corners.astype(np.float32)) or cv2.contourArea(
        corners.astype(np.float32)
    ) < width * height * 0.01:
        raise CalibrationError("board corners must form a sizable convex quadrilateral")
    distortion = np.zeros(5) if distortion is None else np.asarray(distortion, np.float64)
    obj = np.array(
        [
            [0.0, wall_height_m, 0.0],
            [wall_width_m, wall_height_m, 0.0],
            [wall_width_m, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    success, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        obj, corners, K, distortion, flags=cv2.SOLVEPNP_IPPE
    )
    if not success:
        raise CalibrationError("board-corner PnP failed")
    candidates = []
    for rvec, tvec in zip(rvecs, tvecs, strict=False):
        rvec, tvec = cv2.solvePnPRefineLM(obj, corners, K, distortion, rvec, tvec)
        R = cv2.Rodrigues(rvec)[0]
        if np.any((obj @ R.T + tvec.ravel())[:, 2] <= 0):
            continue
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, tvec.ravel()
        if (-R.T @ tvec.ravel())[2] <= 0:
            continue
        residual = project_wall(obj, K, T, distortion) - corners
        rms = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
        candidates.append((rms, T))
    if not candidates:
        raise CalibrationError("no valid camera pose in front of the board")
    rms, T = min(candidates, key=lambda candidate: candidate[0])
    return WallCalibration(T, [], rms, {0: corners}, 4)


def calibrate_wall_boundary(rgb, K, wall_width_m=0.8, wall_height_m=1.4, board_face_z_m=0.0004):
    """Markerless challenger: plywood segmentation and four fitted outer edges.

    Requires the known rectangular board to be substantially visible, upright
    in the image, and color-separated from its surroundings. This is specific
    to the released plywood board, not a general markerless solution. Corner
    marker plates may remove the ends of an edge; long-edge intersections
    extrapolate those corners without reading their IDs or pixel patterns.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    wood = (
        (hsv[..., 0] >= 15)
        & (hsv[..., 0] <= 50)
        & (hsv[..., 1] >= 25)
        & (hsv[..., 1] < 125)
        & (hsv[..., 2] >= 45)
    ).astype(np.uint8)
    contours, _ = cv2.findContours(wood, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise CalibrationError("no plywood board color region")
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < rgb.shape[0] * rgb.shape[1] * 0.03:
        raise CalibrationError("board region is too small")
    hull = cv2.convexHull(contour)
    poly = (
        cv2.approxPolyDP(hull, cv2.arcLength(hull, True) * 0.004, True).reshape(-1, 2).astype(float)
    )
    if len(poly) < 4:
        raise CalibrationError("board does not supply four edges")
    lengths = np.linalg.norm(np.roll(poly, -1, axis=0) - poly, axis=1)
    selected = sorted(np.argsort(lengths)[-4:])
    if min(lengths[selected]) < 35:
        raise CalibrationError("four board edges are not resolved")
    points = contour.reshape(-1, 2).astype(float)
    lines = []
    for i in selected:
        a, b = poly[i], poly[(i + 1) % len(poly)]
        direction = (b - a) / lengths[i]
        along = (points - a) @ direction
        delta = points - a
        perpendicular = np.abs(direction[0] * delta[:, 1] - direction[1] * delta[:, 0])
        near = points[(perpendicular < 2.0) & (along > 2.0) & (along < lengths[i] - 2.0)]
        if len(near) < 20:
            raise CalibrationError("board edge fit has insufficient support")
        vx, vy, x, y = cv2.fitLine(near.astype(np.float32), cv2.DIST_HUBER, 0.0, 0.01, 0.01).ravel()
        normal = np.array([-vy, vx], float)
        lines.append((normal, float(normal @ [x, y])))
    corners = []
    for i in range(4):
        n1, d1 = lines[i - 1]
        n2, d2 = lines[i]
        M = np.array([n1, n2])
        if abs(np.linalg.det(M)) < 0.15:
            raise CalibrationError("board edge directions are degenerate")
        corners.append(np.linalg.solve(M, [d1, d2]))
    corners = np.asarray(corners)
    if (
        np.any(corners[:, 0] < 0)
        or np.any(corners[:, 0] >= rgb.shape[1])
        or np.any(corners[:, 1] < 0)
        or np.any(corners[:, 1] >= rgb.shape[0])
    ):
        raise CalibrationError("board boundary extends beyond image")
    # The physical board is upright: sort upper/lower pairs by image y.
    upper = corners[np.argsort(corners[:, 1])[:2]]
    lower = corners[np.argsort(corners[:, 1])[2:]]
    upper, lower = upper[np.argsort(upper[:, 0])], lower[np.argsort(lower[:, 0])]
    corners = np.array([upper[0], upper[1], lower[1], lower[0]])
    obj = np.array(
        [
            [0.0, wall_height_m, board_face_z_m],
            [wall_width_m, wall_height_m, board_face_z_m],
            [wall_width_m, 0.0, board_face_z_m],
            [0.0, 0.0, board_face_z_m],
        ]
    )
    K = np.asarray(K, float)
    success, rvec, tvec = cv2.solvePnP(obj, corners, K, None, flags=cv2.SOLVEPNP_IPPE)
    if not success:
        raise CalibrationError("boundary PnP failed")
    rvec, tvec = cv2.solvePnPRefineLM(obj, corners, K, None, rvec, tvec)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = cv2.Rodrigues(rvec)[0], tvec.ravel()
    if (-T[:3, :3].T @ T[:3, 3])[2] <= 0:
        raise CalibrationError("boundary camera is behind the board")
    residual = np.linalg.norm(project_wall(obj, K, T) - corners, axis=1)
    rms = float(np.sqrt(np.mean(residual**2)))
    if rms > 3.0:
        raise CalibrationError(f"board rectangle reprojection RMS {rms:.2f}px exceeds 3px")
    return WallCalibration(T, [], rms, {0: corners}, 4)
