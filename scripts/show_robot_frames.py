"""Draw wall, camera and estimated trunk axes on saved simulator camera images.

Run with third_party/simulator/.venv/bin/python scripts/show_robot_frames.py.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from robot_pose_aruco import mujoco_pixel_intrinsics
from visualize_aruco_tests import font

COLOURS = [(245, 65, 65), (50, 220, 80), (70, 140, 255)]


def main():
    root = Path.home() / "robot-pose-debug-terminal"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=root / "upright_climb_render_fix_terminal" / "height_00"
    )
    parser.add_argument("--output", type=Path, default=root)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = json.loads((args.input / "result.json").read_text())["estimate"]
    position = np.array(result["position_m"])
    rotation = np.array(result["rotation_matrix"])
    from climbing_sim.cameras import default_cameras

    cameras = default_cameras()
    conversion = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])
    extrinsics = [np.column_stack([c.R_wall_to_cam() @ conversion, c.tvec]) for c in cameras]
    board = Image.new("RGB", (1280, 1120), (20, 24, 30))
    draw = ImageDraw.Draw(board)
    draw.text(
        (20, 12), "Wall coordinates and fitted robot axes in both camera views", font=font(25)
    )
    draw.text((20, 52), "X = red    Y = green    Z = blue", font=font(21))
    draw.text(
        (20, 85),
        "Wall: X right, Y into wall, Z up. Robot: X forward, Y left, Z toward head.",
        font=font(18),
    )
    for index, (camera, extrinsic) in enumerate(zip(cameras, extrinsics, strict=True)):
        image = np.asarray(Image.open(args.input / f"{camera.name}.png").convert("RGB")).copy()
        matrix = mujoco_pixel_intrinsics(camera.K())

        def project(points, matrix=matrix, extrinsic=extrinsic):
            return cv2.projectPoints(
                np.asarray(points, dtype=float).reshape(-1, 3),
                cv2.Rodrigues(extrinsic[:, :3])[0],
                extrinsic[:, 3],
                matrix,
                None,
            )[0][:, 0]

        def axes(image, origin, orientation, length, prefix):
            points = np.vstack([origin, origin + length * orientation.T])
            pixels = np.round(project(points)).astype(int)
            for axis, colour in enumerate(COLOURS):
                cv2.arrowedLine(
                    image,
                    tuple(pixels[0]),
                    tuple(pixels[axis + 1]),
                    colour,
                    2,
                    cv2.LINE_AA,
                    tipLength=0.2,
                )
                cv2.putText(
                    image,
                    prefix + "XYZ"[axis],
                    tuple(pixels[axis + 1] + [4, -3]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    colour,
                    1,
                    cv2.LINE_AA,
                )
            return pixels[0]

        grid = image.copy()
        for height in np.arange(0.0, 1.401, 0.2):
            points = np.round(project([[0, 0, height], [0.8, 0, height]])).astype(int)
            cv2.line(grid, tuple(points[0]), tuple(points[1]), (80, 170, 255), 1)
            cv2.putText(
                grid,
                f"z={height:.1f}m",
                tuple(points[0] + [-60, 0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (80, 170, 255),
                1,
                cv2.LINE_AA,
            )
        for x in np.arange(0.0, 0.801, 0.2):
            points = np.round(project([[x, 0, 0], [x, 0, 1.4]])).astype(int)
            cv2.line(grid, tuple(points[0]), tuple(points[1]), (245, 100, 100), 1)
            cv2.putText(
                grid,
                f"x={x:.1f}",
                tuple(points[0] + [-15, 18]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (245, 100, 100),
                1,
                cv2.LINE_AA,
            )
        image = cv2.addWeighted(image, 0.55, grid, 0.45, 0)
        axes(image, np.zeros(3), np.eye(3), 0.22, "W")
        centre = axes(image, position, rotation, 0.07, "R")
        cv2.circle(image, tuple(centre), 3, (255, 255, 255), -1)
        cv2.putText(
            image,
            "Fitted trunk",
            tuple(centre + [15, 20]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # The camera origin is behind its own image plane. Show its convention in an inset.
        cv2.rectangle(image, (15, 15), (355, 145), (20, 24, 30), -1)
        cv2.putText(
            image,
            "Camera frame (inset)",
            (28, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.arrowedLine(image, (70, 70), (170, 70), COLOURS[0], 2, tipLength=0.15)
        cv2.arrowedLine(image, (70, 70), (70, 115), COLOURS[1], 2, tipLength=0.2)
        cv2.putText(image, "+X right", (180, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOURS[0], 1)
        cv2.putText(image, "+Y down", (85, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOURS[1], 1)
        cv2.circle(image, (70, 70), 9, COLOURS[2], 1)
        cv2.line(image, (65, 65), (75, 75), COLOURS[2], 1)
        cv2.line(image, (75, 65), (65, 75), COLOURS[2], 1)
        cv2.putText(
            image, "+Z into image", (180, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOURS[2], 1
        )
        Image.fromarray(image).save(args.output / f"{camera.name}_xyz_axes.png")
        board.paste(Image.fromarray(image).resize((640, 360)), (index * 640, 150))
        draw.text((index * 640 + 20, 120), camera.name, font=font(20))
        x, y = centre
        x, y = int(np.clip(x - 80, 0, image.shape[1] - 160)), int(np.clip(y - 80, 150, 560))
        zoom = Image.fromarray(image[y : y + 160, x : x + 160]).resize(
            (480, 480), Image.Resampling.NEAREST
        )
        board.paste(zoom, (index * 640 + 80, 565))
        draw.text(
            (index * 640 + 80, 530), "Fitted trunk axes (3x image enlargement)", font=font(18)
        )
    draw.text((20, 1070), f"Fitted trunk XYZ: {np.round(position, 4).tolist()} m", font=font(20))
    board.save(args.output / "wall_camera_robot_axes.png")

    # A separate diagram shows camera origins, which cannot appear in their own images.
    scene = Image.new("RGB", (1400, 1000), (20, 24, 30))
    sd = ImageDraw.Draw(scene)
    eye = np.array([2.0, -4.0, 2.8])
    target = np.array([0.4, -1.0, 0.6])
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    def diagram(points):
        points = np.asarray(points) - target
        return np.column_stack([700 + points @ right * 300, 530 - points @ up * 300])

    def triad(origin, orientation, length, label, label_offset=(-30, 20)):
        pixels = diagram(np.vstack([origin, origin + length * orientation.T]))
        start = tuple(pixels[0])
        sd.ellipse((start[0] - 4, start[1] - 4, start[0] + 4, start[1] + 4), fill="white")
        for axis, colour in enumerate(COLOURS):
            end = tuple(pixels[axis + 1])
            sd.line([start, end], fill=colour, width=4)
            sd.ellipse((end[0] - 3, end[1] - 3, end[0] + 3, end[1] + 3), fill=colour)
            sd.text((end[0] + 5, end[1] - 14), "XYZ"[axis], fill=colour, font=font(18))
        sd.text((start[0] + label_offset[0], start[1] + label_offset[1]), label, font=font(18))

    wall = np.array([[0, 0, 0], [0.8, 0, 0], [0.8, 0, 1.4], [0, 0, 1.4]])
    sd.polygon([tuple(p) for p in diagram(wall)], fill=(140, 117, 77), outline=(240, 210, 150))
    for x in np.arange(-0.2, 1.21, 0.2):
        sd.line([tuple(p) for p in diagram([[x, -2.7, 0], [x, 0.2, 0]])], fill=(50, 55, 65))
    for y in np.arange(-2.6, 0.21, 0.2):
        sd.line([tuple(p) for p in diagram([[-0.2, y, 0], [1.2, y, 0]])], fill=(50, 55, 65))
    for camera, extrinsic in zip(cameras, extrinsics, strict=True):
        camera_position = -extrinsic[:, :3].T @ extrinsic[:, 3]
        for corner in wall:
            sd.line([tuple(p) for p in diagram([camera_position, corner])], fill=(65, 75, 95))
        triad(camera_position, extrinsic[:, :3].T, 0.30, camera.name)
    triad(np.zeros(3), np.eye(3), 0.35, "Wall origin", (-140, -25))
    triad(position, rotation, 0.20, "Fitted trunk", (25, 25))
    sd.text((25, 20), "Wall, calibrated cameras and fitted robot frame", font=font(28))
    sd.text(
        (25, 65),
        "X red | Y green | Z blue. Thin lines show camera-to-wall sight lines.",
        font=font(20),
    )
    sd.text(
        (25, 910),
        "Wall: X right, Y into wall, Z up. Camera: X right, Y down, Z forward.",
        font=font(20),
    )
    sd.text(
        (25, 945),
        "Camera and wall frames come from calibration; the trunk frame comes from ArUco.",
        font=font(19),
    )
    scene.save(args.output / "camera_wall_frames_3d.png")
    print(f"Saved labelled camera views and frame diagram in {args.output}")


if __name__ == "__main__":
    main()
