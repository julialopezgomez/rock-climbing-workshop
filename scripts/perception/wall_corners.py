"""Wall pose in markerless or no_aruco_with_marker mode.

Input JSON has an ``images`` list. Each entry needs ``path`` and ``K``.
Set ``mode`` explicitly to distinguish the methods. ``markerless`` uses
supplied physical ``corners_px`` in TL, TR,
BR, BL order; texture detection is diagnostic, not automatic calibration.
``no_aruco_with_marker`` detects the printed black-square silhouettes but
never decodes their ArUco patterns or IDs. Optional ``holes`` entries have
zero-based ``col``, ``row`` (row 0 is bottom), and measured ``pixel``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from rcr.perception.challenge.src.calibration import (
    CalibrationError,
    calibrate_wall_corners,
    calibrate_wall_corner_squares,
    detect_wall_texture_core,
    project_wall,
)
from rcr.types import GRID_MARGIN_X, GRID_MARGIN_Z, GRID_PITCH, WALL_HEIGHT, WALL_WIDTH


def grid_offset_diagnostic(rectified: np.ndarray) -> dict:
    """Compare known hole sites with nearby pixels in unobstructed side strips.

    Offsets are rectified image pixels, equal to millimetres here. A darker
    high-pass response is evidence of a hole, but is not a metric accuracy
    guarantee when holds or lighting obscure the sampled strips.
    """
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY).astype(np.float32)
    highpass = gray - cv2.GaussianBlur(gray, (0, 0), 5)
    bands = {}
    for name, rows in (
        ("bottom", range(8, 30)),
        ("middle", range(31, 60)),
        ("top", range(61, 85)),
    ):
        sites = np.array(
            [
                (
                    round((GRID_MARGIN_X + col * GRID_PITCH) * 1000),
                    round((WALL_HEIGHT - GRID_MARGIN_Z - row * GRID_PITCH) * 1000),
                )
                for row in rows
                for col in (*range(5, 16), *range(39, 49))
            ]
        )
        scores = [
            (
                float(highpass[sites[:, 1] + dy, sites[:, 0] + dx].mean()),
                dx,
                dy,
            )
            for dy in range(-7, 8)
            for dx in range(-7, 8)
        ]
        best_response, dx, dy = min(scores)
        zero_response = next(score for score, x, y in scores if x == 0 and y == 0)
        bands[name] = {
            "best_offset_px_rectified_xy": [dx, dy],
            "best_offset_mm_wall_x_up": [dx, -dy],
            "response_at_measured_grid": zero_response,
            "response_at_best_offset": best_response,
        }
    return {
        "method": "dark-hole response in side strips; diagnostic, not ground truth",
        "bands": bands,
    }


def process_image(entry: dict, output_dir: Path) -> dict:
    path = Path(entry["path"])
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise ValueError(f"cannot read image: {path}")
    height, width = bgr.shape[:2]
    K = np.asarray(entry["K"], dtype=float)
    distortion = np.asarray(entry.get("distortion", [0, 0, 0, 0, 0]), dtype=float)
    mode = entry.get("mode", "markerless" if "corners_px" in entry else "no_aruco_with_marker")
    if mode == "markerless":
        if "corners_px" not in entry:
            raise ValueError("markerless mode requires measured outer corners_px")
        calibration = calibrate_wall_corners(
            entry["corners_px"], K, (width, height), distortion=distortion
        )
        squares = None
        corner_source = "supplied physical board corners; not automatically detected"
    elif mode == "no_aruco_with_marker":
        if "corners_px" in entry:
            raise ValueError("no_aruco_with_marker mode detects corners; remove corners_px")
        calibration, squares = calibrate_wall_corner_squares(
            cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
            K,
            distortion=distortion,
            code_size_m=float(entry.get("code_size_m", 0.1)),
            white_border_m=float(entry.get("white_border_m", 0.018)),
            code_face_z_m=float(entry.get("code_face_z_m", 0.0)),
        )
        corner_source = "detected printed square geometry; no ArUco ID decoding"
    else:
        raise ValueError(f"unknown wall calibration mode: {mode}")
    T = calibration.T_camera_wall
    corners = calibration.corners_px[0]
    overlay = bgr.copy()
    texture_bbox = None
    try:
        texture_mask, texture_bbox = detect_wall_texture_core(
            cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        )
        contours, _ = cv2.findContours(
            texture_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, (255, 255, 0), 2)
    except CalibrationError:
        pass
    if squares is not None:
        for role, points in squares.items():
            cv2.polylines(overlay, [np.rint(points).astype(np.int32)], True, (255, 0, 0), 2)
            x, y = np.rint(points[0]).astype(int)
            cv2.putText(overlay, role, (x, y - 5), 0, 0.5, (255, 0, 0), 1)
    cv2.polylines(overlay, [np.rint(corners).astype(np.int32)], True, (0, 255, 255), 2)
    wall_corners = np.array(
        [[0, WALL_HEIGHT, 0], [WALL_WIDTH, WALL_HEIGHT, 0], [WALL_WIDTH, 0, 0], [0, 0, 0]]
    )
    for x, y in project_wall(wall_corners, K, T, distortion):
        cv2.circle(overlay, (round(x), round(y)), 4, (255, 0, 255), 2)

    # Project the measured 52 x 93 hole lattice from bottom-left, row 0.
    full_grid = np.array(
        [
            [GRID_MARGIN_X + col * GRID_PITCH, GRID_MARGIN_Z + row * GRID_PITCH, 0]
            for row in range(93)
            for col in range(52)
        ],
        dtype=float,
    )
    projected_grid = project_wall(full_grid, K, T, distortion)
    for x, y in projected_grid.reshape(93, 52, 2)[::5, ::5].reshape(-1, 2):
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(overlay, (round(x), round(y)), 1, (0, 180, 0), -1)
    origin = tuple(np.rint(corners[3]).astype(int))
    cv2.circle(overlay, origin, 6, (0, 0, 255), -1)
    cv2.putText(overlay, "(0,0)", (origin[0] + 8, origin[1] - 8), 0, 0.6, (0, 0, 255), 2)
    for label, point in zip(("TL", "TR", "BR", "BL"), corners, strict=True):
        x, y = np.rint(point).astype(int)
        cv2.putText(overlay, label, (x + 5, y - 5), 0, 0.5, (0, 255, 255), 1)

    checks = []
    for hole in entry.get("holes", []):
        col, row = int(hole["col"]), int(hole["row"])
        if not (0 <= col < 52 and 0 <= row < 93):
            raise ValueError(f"hole index outside 52 x 93 grid: ({col}, {row})")
        xyz = np.array(
            [[GRID_MARGIN_X + col * GRID_PITCH, GRID_MARGIN_Z + row * GRID_PITCH, 0]]
        )
        expected = project_wall(xyz, K, T, distortion)[0]
        measured = np.asarray(hole["pixel"], dtype=float)
        error = float(np.linalg.norm(expected - measured))
        checks.append(
            {
                "col": col,
                "row": row,
                "measured_px": measured.tolist(),
                "projected_px": expected.tolist(),
                "error_px": error,
            }
        )
        cv2.circle(overlay, tuple(np.rint(measured).astype(int)), 4, (255, 0, 0), 1)
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = output_dir / f"{path.stem}_wall_overlay.png"
    if not cv2.imwrite(str(overlay_path), overlay):
        raise OSError(f"could not write {overlay_path}")
    full_overlay = overlay.copy()
    for x, y in projected_grid:
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(full_overlay, (round(x), round(y)), 0, (0, 0, 255), -1)
    full_grid_path = output_dir / f"{path.stem}_full_grid.png"
    if not cv2.imwrite(str(full_grid_path), full_overlay):
        raise OSError(f"could not write {full_grid_path}")

    rectified_size = (round(WALL_WIDTH * 1000), round(WALL_HEIGHT * 1000))
    rectified_corners = np.array(
        [
            [0, 0],
            [rectified_size[0] - 1, 0],
            [rectified_size[0] - 1, rectified_size[1] - 1],
            [0, rectified_size[1] - 1],
        ],
        dtype=np.float32,
    )
    if np.any(distortion):
        rectification_source = cv2.undistort(bgr, K, distortion)
        rectification_corners = cv2.undistortPoints(
            corners.reshape(-1, 1, 2), K, distortion, P=K
        ).reshape(4, 2)
    else:
        rectification_source = bgr
        rectification_corners = corners
    homography = cv2.getPerspectiveTransform(
        rectification_corners.astype(np.float32), rectified_corners
    )
    rectified = cv2.warpPerspective(rectification_source, homography, rectified_size)
    rectified_path = output_dir / f"{path.stem}_rectified_wall.png"
    if not cv2.imwrite(str(rectified_path), rectified):
        raise OSError(f"could not write {rectified_path}")
    grid_diagnostic = grid_offset_diagnostic(rectified)
    rectified_grid = rectified.copy()
    for row in range(93):
        y = round((WALL_HEIGHT - GRID_MARGIN_Z - row * GRID_PITCH) * 1000)
        for col in range(52):
            x = round((GRID_MARGIN_X + col * GRID_PITCH) * 1000)
            cv2.circle(rectified_grid, (x, y), 1, (0, 0, 255), -1)
    rectified_grid_path = output_dir / f"{path.stem}_rectified_grid.png"
    if not cv2.imwrite(str(rectified_grid_path), rectified_grid):
        raise OSError(f"could not write {rectified_grid_path}")

    camera_challenge = calibration.camera_position_wall_m
    camera_robot = [
        float(camera_challenge[0]),
        float(-camera_challenge[2]),
        float(camera_challenge[1]),
    ]
    hole_rms = (
        float(np.sqrt(np.mean([check["error_px"] ** 2 for check in checks])))
        if checks
        else None
    )
    return {
        "image": str(path),
        "image_size": [width, height],
        "K": K.tolist(),
        "distortion": distortion.tolist(),
        "intrinsics_source": entry.get("intrinsics_source", "unspecified"),
        "printed_square_geometry_m": (
            {
                "code_size": float(entry.get("code_size_m", 0.1)),
                "white_border": float(entry.get("white_border_m", 0.018)),
                "code_face_z": float(entry.get("code_face_z_m", 0.0)),
            }
            if squares is not None
            else None
        ),
        "corners_px_tl_tr_br_bl": corners.tolist(),
        "mode": mode,
        "corner_source": corner_source,
        "detected_square_corners_px": (
            {role: points.tolist() for role, points in squares.items()}
            if squares is not None
            else None
        ),
        "detected_texture_core_bbox_px_xywh": texture_bbox,
        "wall_size_m": [WALL_WIDTH, WALL_HEIGHT],
        "hole_grid": {
            "columns": 52,
            "rows": 93,
            "pitch_m": GRID_PITCH,
            "first_hole_bottom_left_m": [GRID_MARGIN_X, GRID_MARGIN_Z],
        },
        "hole_grid_alignment": grid_diagnostic,
        "T_camera_wall_challenge": T.tolist(),
        "camera_position_challenge_wall_m": camera_challenge.tolist(),
        "camera_position_robot_wall_m": camera_robot,
        "corner_fit_rms_px_not_accuracy": calibration.reprojection_rms_px,
        "held_out_hole_checks": checks,
        "held_out_hole_rms_px": hole_rms,
        "passes_image_geometry_checks": (
            calibration.reprojection_rms_px <= 3.0
            and len(checks) >= 6
            and hole_rms is not None
            and hole_rms <= 3.0
        ),
        "overlay": str(overlay_path),
        "full_grid_overlay": str(full_grid_path),
        "rectified_wall": str(rectified_path),
        "rectified_grid": str(rectified_grid_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="JSON file with an images list")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    results = [process_image(entry, args.out) for entry in config["images"]]
    output = args.out / "wall_poses.json"
    output.write_text(json.dumps({"results": results}, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
