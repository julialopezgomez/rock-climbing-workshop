"""Project the real wall's 15 mm hole grid onto RGB images using wall ArUco markers.

Run from the workshop root:
  python scripts/overlay_wall_grid.py /path/to/AruCo1_Color.png --output-dir /tmp/wall-grid
  python scripts/overlay_wall_grid.py image1280.png image640.png --pose-pairs \
      --serials 243222071545 233722072758 --output-dir /tmp/wall-grid
  python scripts/overlay_wall_grid.py image.png --pose \
      --serials 233722072758 --output-dir /tmp/wall-grid

The grid is a projection of the documented wall geometry, not a measurement of
physical hole placement. The source images are never modified.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from capture_robot_pose import DEFAULT_CALIBRATION, load_calibration
from robot_pose_aruco import (
    MARKER_CENTRE,
    MARKER_SIZE,
    _wall_corners,
    detect_markers,
    estimate_robot_pose,
)

from rcr.types import GRID_MARGIN_X, GRID_MARGIN_Z, GRID_PITCH, WALL_HEIGHT, WALL_WIDTH


def wall_homography(detections):
    ids = sorted(set(detections) & {1, 2, 3, 4})
    if len(ids) < 3:
        raise ValueError(f"Need at least three wall markers (IDs 1-4); found {ids}")
    world = np.concatenate([_wall_corners(mid)[:, [0, 2]] for mid in ids]).astype(np.float32)
    pixels = np.concatenate([detections[mid] for mid in ids]).astype(np.float32)
    homography, _ = cv2.findHomography(world, pixels, method=0)
    if homography is None:
        raise ValueError("Could not fit the wall plane to detected markers")
    projected = cv2.perspectiveTransform(world[None], homography)[0]
    rms = float(np.sqrt(np.mean(np.sum((projected - pixels) ** 2, axis=1))))
    if rms > 3.0:
        raise ValueError(f"Wall-marker fit is {rms:.1f} px; refusing a misleading grid")
    return homography, ids, rms


def project(homography, points):
    return cv2.perspectiveTransform(np.asarray(points, dtype=np.float32)[None], homography)[0]


def hole_coordinates():
    columns = GRID_MARGIN_X + GRID_PITCH * np.arange(52)
    rows = GRID_MARGIN_Z + GRID_PITCH * np.arange(93)
    return columns, rows


def label(image, text, point, scale=0.55):
    x, y = np.rint(point).astype(int)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4,
                cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1,
                cv2.LINE_AA)


def annotate(image, homography):
    output = image.copy()
    columns, rows = hole_coordinates()
    fine = output.copy()
    major = output.copy()
    grid_start, grid_end = rows[0], rows[-1]
    left, right = columns[0], columns[-1]
    for index, x in enumerate(columns):
        points = np.rint(project(homography, [[x, grid_start], [x, grid_end]])).astype(int)
        layer = major if index % 10 == 0 else fine
        cv2.line(layer, tuple(points[0]), tuple(points[1]), (255, 220, 0), 1, cv2.LINE_AA)
    for index, z in enumerate(rows):
        points = np.rint(project(homography, [[left, z], [right, z]])).astype(int)
        layer = major if index % 10 == 0 else fine
        cv2.line(layer, tuple(points[0]), tuple(points[1]), (255, 220, 0), 1, cv2.LINE_AA)
    output = cv2.addWeighted(fine, 0.33, output, 0.67, 0)
    output = cv2.addWeighted(major, 0.62, output, 0.38, 0)

    board = np.rint(
        project(homography, [[0, 0], [WALL_WIDTH, 0],
                             [WALL_WIDTH, WALL_HEIGHT], [0, WALL_HEIGHT]])
    ).astype(int)
    cv2.polylines(output, [board], True, (0, 180, 255), 2, cv2.LINE_AA)
    origin = board[0]
    x_axis, z_axis = np.rint(project(homography, [[0.20, 0], [0, 0.20]])).astype(int)
    cv2.circle(output, tuple(origin), 6, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.arrowedLine(output, tuple(origin), tuple(x_axis), (0, 0, 255), 2, cv2.LINE_AA,
                    tipLength=0.17)
    cv2.arrowedLine(output, tuple(origin), tuple(z_axis), (0, 0, 255), 2, cv2.LINE_AA,
                    tipLength=0.17)
    label(output, "x", x_axis + [5, -5])
    label(output, "z", z_axis + [-14, -5])
    label(output, "(0, 0) mm", origin + [-125, -16])

    marker_three = np.rint(project(homography, [[0.068, 0.068]])[0]).astype(int)
    cv2.circle(output, tuple(marker_three), 4, (0, 0, 255), -1, cv2.LINE_AA)
    label(output, "ID 3: (68, 68) mm", marker_three + [12, -10], 0.44)
    banner = output.copy()
    cv2.rectangle(banner, (7, 5), (365, 38), (0, 0, 0), -1)
    output = cv2.addWeighted(banner, 0.65, output, 0.35, 0)
    label(output, "15 mm grid | first hole (17.5, 10) mm", [14, 28], 0.48)
    return output


def calibration_path_for(size):
    if size == (640, 480):
        return DEFAULT_CALIBRATION
    if size == (1280, 720):
        return DEFAULT_CALIBRATION.with_name("realsense-1280x720.json")
    raise ValueError(f"No Color calibration for {size[0]}x{size[1]}")


def calibration_for(serial, size):
    path = calibration_path_for(size)
    matrices, distortion, _, _, serials = load_calibration(path)
    if serial not in serials:
        raise ValueError(f"No calibration for serial {serial}")
    index = serials.index(serial)
    return matrices[index], distortion[index]


def pose_report(paths, images, serials, calibrations, result, fits):
    rotation = np.asarray(result["rotation_matrix"])
    marker = np.asarray(result["position_m"]) + rotation @ MARKER_CENTRE
    return {
        "pose": result,
        "marker_center_wall_m": marker.tolist(),
        "wall_frame": {
            "origin": "bottom-left outer corner, viewed from the front",
            "axes": "x right, y into wall, z up",
            "board_width_m": WALL_WIDTH,
            "board_height_m": WALL_HEIGHT,
            "quaternion_order": "wxyz, trunk-to-wall",
        },
        "cameras": [
            {
                "image": str(path.resolve()),
                "assumed_serial": serial,
                "resolution": [image.shape[1], image.shape[0]],
                "calibration_file": str(calibration_path_for((image.shape[1], image.shape[0]))),
                "camera_matrix": matrix.tolist(),
                "distortion_coeffs": distortion.tolist(),
                "wall_marker_ids": ids,
                "wall_homography_rms_px": rms,
            }
            for path, image, serial, (matrix, distortion), (_, ids, rms) in zip(
                paths, images, serials, calibrations, fits, strict=True
            )
        ],
        "assumptions": [
            "Camera serials are supplied by the caller; PNG files do not establish their serials.",
            f"Robot ID 5 black code is {MARKER_SIZE * 1000:g} mm square.",
            "Robot marker-to-trunk transform is the simulator mount in robot_pose_aruco.py.",
            "Wall marker geometry and board dimensions are those in robot_pose_aruco.py.",
            "If two images are provided, they must show the same static robot pose; "
            "no synchronization is checked.",
            "Reprojection residuals measure model consistency, not ground-truth pose error.",
        ],
    }


def add_pose_annotation(image, detections, matrix, distortion, result, serial):
    ids = sorted(set(detections) & {1, 2, 3, 4})
    world = np.concatenate([_wall_corners(mid) for mid in ids])
    pixels = np.concatenate([detections[mid] for mid in ids])
    ok, rvec, tvec = cv2.solvePnP(world, pixels, matrix, distortion)
    if not ok:
        raise ValueError("Cannot project trunk origin into image")

    trunk = np.asarray(result["position_m"])
    rotation = np.asarray(result["rotation_matrix"])
    marker = trunk + rotation @ MARKER_CENTRE
    trunk_pixel = cv2.projectPoints(trunk[None], rvec, tvec, matrix, distortion)[0][0, 0]
    marker_pixel = detections[5].mean(axis=0)
    trunk_pixel, marker_pixel = [tuple(np.rint(point).astype(int))
                                 for point in (trunk_pixel, marker_pixel)]

    output = np.full((image.shape[0] + 168, image.shape[1], 3), (26, 29, 32), dtype=np.uint8)
    output[: image.shape[0]] = image
    cv2.line(output, marker_pixel, trunk_pixel, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(output, marker_pixel, 7, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.circle(output, trunk_pixel, 7, (255, 70, 210), 2, cv2.LINE_AA)
    label(output, "ID 5", np.asarray(marker_pixel) + [9, -9], 0.45)
    label(output, "trunk", np.asarray(trunk_pixel) + [9, -9], 0.45)

    def position_text(values):
        return "  ".join(f"{value:+.3f}" for value in values)

    lines = [
        f"ID 5 (x, y, z) m:   {position_text(marker)}",
        f"Trunk (x, y, z) m: {position_text(trunk)}",
        f"Trunk q (w,x,y,z): {position_text(result['quaternion_wxyz'])}",
        f"Assumed serial {serial}; simulated 42 mm marker and mount",
        "y points into wall; larger y is closer to wall",
        ("SINGLE CAMERA: depth and orientation ambiguous" if result["method"] == "single_camera"
         else "STEREO: ID 5 fitted jointly in both images"),
    ]
    for index, line in enumerate(lines):
        cv2.putText(output, line, (12, image.shape[0] + 25 + index * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pose", action="store_true",
                        help="Estimate pose from one image, or prefer stereo with two images")
    parser.add_argument("--pose-pairs", action="store_true",
                        help="Estimate stereo pose from consecutive image pairs")
    parser.add_argument("--serials", nargs="+", metavar="SERIAL",
                        help="Assumed camera serials in image order")
    args = parser.parse_args()
    if args.pose and args.pose_pairs:
        parser.error("Choose --pose or --pose-pairs")
    if args.pose and (len(args.images) > 2 or args.serials is None
                      or len(args.serials) != len(args.images)):
        parser.error("--pose needs one or two images and one --serials entry per image")
    if args.pose_pairs and (len(args.images) % 2 or args.serials is None
                            or len(args.serials) != 2):
        parser.error("--pose-pairs needs an even number of images and --serials FIRST SECOND")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    step = 2 if args.pose_pairs else len(args.images) if args.pose else 1
    for offset in range(0, len(args.images), step):
        paths = args.images[offset : offset + step]
        images = [cv2.imread(str(path)) for path in paths]
        for path, image in zip(paths, images, strict=True):
            if image is None:
                parser.exit(1, f"Cannot read image: {path}\n")
        detections = [detect_markers(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                      for image in images]
        try:
            fits = [wall_homography(found) for found in detections]
            result = None
            if args.pose or args.pose_pairs:
                calibrations = [calibration_for(serial, (image.shape[1], image.shape[0]))
                                for serial, image in zip(args.serials, images, strict=True)]
                matrices = [calibration[0] for calibration in calibrations]
                distortion = [calibration[1] for calibration in calibrations]
                rgb = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
                result = estimate_robot_pose(rgb, matrices, distortion)
                if args.pose_pairs and result["method"] != "stereo":
                    raise ValueError("Both cameras must see robot marker ID 5 for stereo pose")
        except ValueError as error:
            parser.exit(1, f"{paths[0].name}: {error}\n")
        for index, (path, image, found, fit) in enumerate(zip(paths, images, detections, fits,
                                                                strict=True)):
            homography, ids, rms = fit
            annotated = annotate(image, homography)
            if result is not None and index in result["cameras_used"]:
                matrix, distortion_coeffs = calibrations[index]
                annotated = add_pose_annotation(annotated, found, matrix, distortion_coeffs,
                                                 result, args.serials[index])
            output = args.output_dir / f"{path.stem}_grid.png"
            if not cv2.imwrite(str(output), annotated):
                parser.exit(1, f"Cannot write image: {output}\n")
            print(f"{output}: wall IDs {ids}, fit {rms:.2f} px")
        if result is not None:
            report = pose_report(paths, images, args.serials, calibrations, result, fits)
            stem = "__".join(path.stem for path in paths)
            output = args.output_dir / f"{stem}_pose.json"
            output.write_text(json.dumps(report, indent=2) + "\n")
            print(f"{output}: {result['method']} trunk pose and calibration details")


if __name__ == "__main__":
    main()
