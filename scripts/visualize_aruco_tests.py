"""Save pose overlays and synthetic corner-test illustrations next to camera images.

Run from the workshop root using third_party/simulator/.venv/bin/python.
The original camera PNGs are read only. All enlargements use nearest-neighbour
sampling so they show existing pixels rather than inventing image detail.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from robot_pose_aruco import (
    MARKER_CENTRE,
    R_TRUNK_MARKER,
    detect_markers,
    estimate_robot_pose,
    mujoco_pixel_intrinsics,
    refine_corners,
)

MAGENTA = (255, 70, 210)
GREEN = (70, 255, 100)
CYAN = (30, 220, 255)


def font(size):
    path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    return ImageFont.truetype(path, size) if Path(path).exists() else ImageFont.load_default()


def outline(image, corners, colour):
    points = np.round(corners).astype(np.int32)
    cv2.polylines(image, [points], True, colour, 1, cv2.LINE_AA)
    for point in points:
        cv2.circle(image, tuple(point), 3, colour, 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path.home() / "robot-pose-debug-terminal")
    args = parser.parse_args()
    folder = args.directory
    cv2.setNumThreads(2)
    from climbing_sim.cameras import default_cameras

    cameras = default_cameras()
    images = []
    for camera in cameras:
        path = folder / f"{camera.name}.png"
        bgr = cv2.imread(str(path))
        if bgr is None:
            parser.error(f"Cannot read camera image: {path}")
        images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    matrices = [mujoco_pixel_intrinsics(c.K()) for c in cameras]
    world_to_calibration_frame = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])
    extrinsics = [
        np.column_stack([c.R_wall_to_cam() @ world_to_calibration_frame, c.tvec]) for c in cameras
    ]
    detections = [detect_markers(image) for image in images]
    result = estimate_robot_pose(images, matrices, camera_extrinsics=extrinsics)
    (folder / "pose_latest.json").write_text(json.dumps(result, indent=2) + "\n")
    rotation = np.asarray(result["rotation_matrix"])
    position = np.asarray(result["position_m"])
    local = np.array(
        [[-0.021, 0.021, 0], [0.021, 0.021, 0], [0.021, -0.021, 0], [-0.021, -0.021, 0]]
    )
    world = (local @ R_TRUNK_MARKER.T + MARKER_CENTRE) @ rotation.T + position

    board = Image.new("RGB", (1280, 1040), (20, 24, 30))
    draw = ImageDraw.Draw(board)
    draw.text((20, 12), "Your saved simulator images: calibrated stereo pose", font=font(25))
    draw.text(
        (20, 50), "Magenta: detected border    Green: projected fitted 42 mm square", font=font(20)
    )
    draw.text(
        (20, 82),
        "The overlay compares image measurements; it is not ground truth.",
        font=font(18),
        fill=(190, 195, 205),
    )
    for index, (image, detected, matrix, extrinsic) in enumerate(
        zip(images, detections, matrices, extrinsics, strict=True)
    ):
        projected = cv2.projectPoints(
            world, cv2.Rodrigues(extrinsic[:, :3])[0], extrinsic[:, 3], matrix, None
        )[0][:, 0]
        annotated = image.copy()
        for mid, corners in detected.items():
            outline(annotated, corners, MAGENTA)
            cv2.putText(
                annotated,
                str(mid),
                tuple(corners[0].astype(int) + [0, -5]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                MAGENTA,
                1,
                cv2.LINE_AA,
            )
        outline(annotated, projected, GREEN)
        Image.fromarray(annotated).save(folder / f"{cameras[index].name}_pose_overlay.png")
        board.paste(Image.fromarray(annotated).resize((640, 360)), (index * 640, 145))
        draw.text((index * 640 + 20, 115), cameras[index].name, font=font(20))
        centre = detected[5].mean(axis=0)
        x, y = np.floor(centre - 27).astype(int)
        crop = image[y : y + 54, x : x + 54]
        zoom = cv2.resize(crop, (432, 432), interpolation=cv2.INTER_NEAREST)
        for corners, colour in ((detected[5], MAGENTA), (projected, GREEN)):
            outline(zoom, (corners - [x, y] + 0.5) * 8 - 0.5, colour)
        board.paste(Image.fromarray(zoom), (index * 640 + 104, 555))
        error = result["marker_reprojection_rms_px"][index]
        draw.text(
            (index * 640 + 55, 515),
            f"8x pixel enlargement; image-fit RMS {error:.3f} px",
            font=font(18),
        )
    draw.text(
        (20, 1005),
        "Original image dimensions: 1280 x 720. Cameras and robot unchanged.",
        font=font(18),
    )
    board.save(folder / "pose_test_images.png")

    board = Image.new("RGB", (1200, 570), (20, 24, 30))
    draw = ImageDraw.Draw(board)
    draw.text((20, 10), "Synthetic image tests: an 18-pixel ArUco ID 5", font=font(24))
    draw.text(
        (20, 48),
        "Magenta: coarse corners    Cyan: known ideal border    Green: refined",
        font=font(19),
    )
    for index, damaged in enumerate((False, True)):
        gray = np.full((100, 100), 180, np.uint8)
        gray[40:58, 40:58] = cv2.aruco.generateImageMarker(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 5, 18
        )
        if damaged:
            gray[40:42, 43:47] = 180
        coarse = np.array([[40.0, 40.0], [57.0, 40.0], [57.0, 57.0], [40.0, 57.0]])
        ideal = np.array([[39.5, 39.5], [57.5, 39.5], [57.5, 57.5], [39.5, 57.5]])
        refined = refine_corners(gray, coarse)
        zoom = cv2.resize(
            cv2.cvtColor(gray[32:66, 32:66], cv2.COLOR_GRAY2RGB),
            (408, 408),
            interpolation=cv2.INTER_NEAREST,
        )
        for corners, colour in ((coarse, MAGENTA), (ideal, CYAN), (refined, GREEN)):
            outline(zoom, (corners - 32 + 0.5) * 12 - 0.5, colour)
        board.paste(Image.fromarray(zoom), (index * 600 + 96, 120))
        draw.text(
            (index * 600 + 96, 86),
            "Weak border section" if damaged else "Clean border",
            font=font(20),
        )
    draw.text(
        (20, 540),
        "Illustrations are synthetic tests, not additional simulator observations.",
        font=font(18),
    )
    board.save(folder / "synthetic_corner_tests.png")
    print(
        f"Saved pose_test_images.png, synthetic_corner_tests.png and pose_latest.json in {folder}"
    )


if __name__ == "__main__":
    main()
