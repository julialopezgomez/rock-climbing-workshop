"""Check ArUco pose across temporary robot placements, without modifying the model.

MUJOCO_GL=egl third_party/simulator/.venv/bin/python scripts/benchmark_robot_pose.py
Custom pose: --pose X Y Z ROLL PITCH YAW (metres/degrees; repeatable).
Upright climb: --climb-only (fixed x/y, increasing height, small 3 degree tilts).
Angles are world-axis offsets from the normal hanging orientation, Rz @ Ry @ Rx.
These are static vision tests: poses may float or collide and are not climbing plans.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from robot_pose_aruco import (
    MARKER_CENTRE,
    R_TRUNK_MARKER,
    _quaternion,
    estimate_robot_pose,
    mujoco_pixel_intrinsics,
)
from visualize_aruco_tests import font, outline


def pose_cases():
    cases = []
    for height in (0.30, 0.70, 1.10):
        for x in (0.18, 0.40, 0.62):
            cases.append((f"grid_{len(cases):02d}", [x, -0.16, height, 0, 0, 0]))
    for name, angles in (
        ("roll_minus15", [-15, 0, 0]),
        ("roll_plus15", [15, 0, 0]),
        ("pitch_minus15", [0, -15, 0]),
        ("pitch_plus15", [0, 15, 0]),
        ("yaw_minus20", [0, 0, -20]),
        ("yaw_plus20", [0, 0, 20]),
        ("yaw_minus45", [0, 0, -45]),
        ("yaw_plus45", [0, 0, 45]),
        ("side_on", [0, 0, 85]),
        ("back_hidden", [0, 0, 180]),
    ):
        cases.append((name, [0.40, -0.20, 0.70, *angles]))
    return cases


def orientation(angles, baseline):
    x, y, z = np.radians(angles)
    rx = cv2.Rodrigues(np.array([x, 0.0, 0.0]))[0]
    ry = cv2.Rodrigues(np.array([0.0, y, 0.0]))[0]
    rz = cv2.Rodrigues(np.array([0.0, 0.0, z]))[0]
    return rz @ ry @ rx @ baseline


def climb_cases(position):
    """Vertical vision sweep at the original trunk distance, without a pose prior."""
    x, y, z = (float(value) for value in position)
    cases = [
        (f"height_{i:02d}", [x, y, float(height), 0, 0, 0])
        for i, height in enumerate(np.linspace(z, 1.10, 8))
    ]
    for axis, name in enumerate(("roll", "pitch", "yaw")):
        for angle in (-3, 3):
            angles = [0, 0, 0]
            angles[axis] = angle
            cases.append((f"small_{name}_{angle:+d}", [x, y, 0.65, *angles]))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wall", default="N2")
    parser.add_argument(
        "--output", type=Path, help="Defaults to pose_sweep or upright_climb under the debug folder"
    )
    parser.add_argument(
        "--pose",
        nargs=6,
        type=float,
        action="append",
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
    )
    parser.add_argument("--climb-only", action="store_true", help="Vertical motion and small tilts")
    args = parser.parse_args()
    if args.climb_only and args.pose:
        parser.error("Choose --climb-only or custom --pose values.")
    if args.output is None:
        args.output = (
            Path.home()
            / "robot-pose-debug-terminal"
            / ("upright_climb" if args.climb_only else "pose_sweep")
        )
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(2)
    from climbing_sim import JOINTS, ClimbingSim

    sim = ClimbingSim(args.wall)
    rows, thumbnails = [], []
    try:
        obs = sim.reset_hanging()
        original_state = sim.get_state()
        baseline = sim.data.body("trunk").xmat.reshape(3, 3).copy()
        initial_position = sim.data.body("trunk").xpos.copy()
        joints = dict(zip(JOINTS, obs["q"], strict=True))
        cameras = list(sim.cameras.values())
        matrices = [mujoco_pixel_intrinsics(c.K()) for c in cameras]
        conversion = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])
        extrinsics = [np.column_stack([c.R_wall_to_cam() @ conversion, c.tvec]) for c in cameras]
        cases = (
            [(f"custom_{i:02d}", value) for i, value in enumerate(args.pose)]
            if args.pose
            else climb_cases(initial_position)
            if args.climb_only
            else pose_cases()
        )
        for name, values in cases:
            folder = args.output / name
            folder.mkdir(exist_ok=True)
            rotation = orientation(values[3:], baseline)
            sim.reset(trunk_pos=values[:3], trunk_quat=_quaternion(rotation), joints=joints)
            images = [sim.render(c.name) for c in cameras]
            trunk = sim.data.body("trunk")
            position = trunk.xpos.copy()
            rotation = trunk.xmat.reshape(3, 3).copy()
            row = {
                "case": name,
                "requested_pose_xyz_rpy": values,
                "true_position_m": position.tolist(),
                "true_quaternion_wxyz": _quaternion(rotation),
            }
            # Truth is used only below for scoring and yellow reference outlines.
            try:
                result = estimate_robot_pose(images, matrices, camera_extrinsics=extrinsics)
                estimated_rotation = np.asarray(result["rotation_matrix"])
                estimated_position = np.asarray(result["position_m"])
                row.update(
                    {
                        "status": result["method"],
                        "estimate": result,
                        "position_error_mm": float(
                            np.linalg.norm(estimated_position - position) * 1000
                        ),
                        "orientation_error_deg": float(
                            np.degrees(
                                np.arccos(
                                    np.clip(
                                        (np.trace(estimated_rotation.T @ rotation) - 1) / 2, -1, 1
                                    )
                                )
                            )
                        ),
                    }
                )
            except ValueError as error:
                result = None
                row.update({"status": "unavailable", "reason": str(error)})
            board = Image.new("RGB", (1280, 435), (20, 24, 30))
            draw = ImageDraw.Draw(board)
            label = f"{name}: {row['status']}"
            if result:
                label += (
                    f" | {row['position_error_mm']:.2f} mm | {row['orientation_error_deg']:.2f} deg"
                )
            draw.text((15, 10), label, font=font(23))
            draw.text(
                (15, 44),
                "Yellow: true marker pose; green: estimated pose. Static vision test.",
                font=font(18),
            )
            local = np.array(
                [[-0.021, 0.021, 0], [0.021, 0.021, 0], [0.021, -0.021, 0], [-0.021, -0.021, 0]]
            )
            for index, (camera, image, matrix, extrinsic) in enumerate(
                zip(cameras, images, matrices, extrinsics, strict=True)
            ):
                Image.fromarray(image).save(folder / f"{camera.name}.png")
                overlay = image.copy()
                transforms = [(rotation, position, (255, 220, 20))]
                if result:
                    transforms.append((estimated_rotation, estimated_position, (60, 255, 100)))
                for r, t, colour in transforms:
                    world = (local @ R_TRUNK_MARKER.T + MARKER_CENTRE) @ r.T + t
                    pixels = cv2.projectPoints(
                        world,
                        cv2.Rodrigues(extrinsic[:, :3])[0],
                        extrinsic[:, 3],
                        matrix,
                        None,
                    )[0][:, 0]
                    outline(overlay, pixels, colour)
                Image.fromarray(overlay).save(folder / f"{camera.name}_overlay.png")
                board.paste(Image.fromarray(overlay).resize((640, 360)), (index * 640, 75))
            board.save(folder / "comparison.png")
            thumbnail = Image.new("RGB", (400, 290), (20, 24, 30))
            td = ImageDraw.Draw(thumbnail)
            td.text((8, 4), name, font=font(19))
            td.text(
                (8, 29),
                (
                    f"{row['status']}: {row['position_error_mm']:.2f} mm / "
                    f"{row['orientation_error_deg']:.2f} deg"
                )
                if result
                else "Unavailable: marker not reliably observed",
                font=font(14),
            )
            thumbnail.paste(Image.fromarray(images[0]).resize((400, 225)), (0, 65))
            thumbnails.append(thumbnail)
            (folder / "result.json").write_text(json.dumps(row, indent=2) + "\n")
            rows.append(row)
            print(label, flush=True)
        sim.set_state(original_state)
    finally:
        sim.close()

    summary = {
        "cases": len(rows),
        "static_vision_tests_only": True,
        "mode": "upright_climb" if args.climb_only else "general_sweep",
    }
    for status in ("stereo", "single_camera", "unavailable"):
        group = [r for r in rows if r["status"] == status]
        summary[status] = {"count": len(group)}
        if group and status != "unavailable":
            for metric in ("position_error_mm", "orientation_error_deg"):
                summary[status][metric] = {
                    "median": float(np.median([r[metric] for r in group])),
                    "max": float(max(r[metric] for r in group)),
                }
    (args.output / "results.json").write_text(
        json.dumps({"summary": summary, "cases": rows}, indent=2) + "\n"
    )
    sheet = Image.new("RGB", (1600, ((len(thumbnails) + 3) // 4) * 290), (20, 24, 30))
    for i, thumbnail in enumerate(thumbnails):
        sheet.paste(thumbnail, ((i % 4) * 400, (i // 4) * 290))
    sheet.save(args.output / "overview.png")
    (args.output / "README.md").write_text(
        "# Static ArUco position/orientation sweep\n\n"
        "Temporary robot poses are set through reset arguments. Robot geometry, marker, "
        "cameras, controls and the default initial pose are unchanged.\n\n"
        "These are static camera/estimator tests, not physically feasible climbing stances. "
        "No physics is advanced after placement. Some poses may float or intersect holds. "
        "Ground truth is used only for scoring and yellow reference outlines. The general "
        "sweep includes hidden-marker cases; --climb-only instead holds x/y fixed, sweeps "
        "height, and tests +/-3 degree tilts. The estimator still measures full rotation "
        "without forcing an upright quaternion or using a rotation prior.\n\n"
        "Open overview.png for all cases, results.json for every outcome, and each case's "
        "comparison.png for both camera views. Single-camera results retain ambiguity "
        "warnings and are summarised separately. Failed cases are retained.\n"
    )
    print(json.dumps(summary, indent=2))
    print(f"Images and results saved to {args.output}")


if __name__ == "__main__":
    main()
