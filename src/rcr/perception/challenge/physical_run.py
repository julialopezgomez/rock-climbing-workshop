"""Process independent real RGB recordings into reviewable geometry hypotheses."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

from .physical_calibration import calibrate_perforated_board, fit_grid_pose, hole_centres
from .physical_mapping import choose_scale, cup_instances, map_cups
from .physical_robot import foreground_box, label_hypothesis
from .real_recordings import NAMES, decode_recording, source_directory
from .src.calibration import project_wall


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def camera_stability(folder, metadata, camera, correspondences, geometry):
    reference, _ = cv2.findHomography(
        correspondences["indices"], correspondences["pixels"], cv2.RANSAC, 0.8
    )
    records = []
    for record in metadata["stability_frames"]:
        image = np.array(Image.open(folder / record["filename"]))
        scale = 720 / image.shape[0]
        small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        pixels = hole_centres(small) / scale
        coordinates = cv2.perspectiveTransform(pixels[:, None], np.linalg.inv(reference))[:, 0]
        indices = np.round(coordinates)
        good = (
            (np.linalg.norm(coordinates - indices, axis=1) < 0.2)
            & (indices[:, 0] >= 0)
            & (indices[:, 0] <= 51)
            & (indices[:, 1] >= 0)
            & (indices[:, 1] <= 93)
        )
        try:
            fitted, _, _ = fit_grid_pose(
                pixels[good], indices[good], np.array(metadata["K"]), geometry
            )
            a, b = [np.array(c["T_camera_wall"]) for c in [camera, fitted]]
            angle = np.degrees(Rotation.from_matrix(a[:3, :3] @ b[:3, :3].T).magnitude())
            position = np.linalg.norm(
                np.array(camera["camera_position_wall_m"]) - fitted["camera_position_wall_m"]
            )
            records.append(
                dict(
                    filename=record["filename"],
                    position_difference_mm=float(position * 1000),
                    rotation_difference_deg=float(angle),
                    reprojection_rms_px=fitted["reprojection_rms_px"],
                )
            )
        except ValueError as error:
            records.append(dict(filename=record["filename"], error=str(error)))
    return dict(
        records=records,
        scope=(
            "Within-recording repeatability with first-frame lattice indexing; "
            "not absolute accuracy"
        ),
    )


def frontend(args, stage):
    interpreter = args.sam3_python if stage == "sam" else args.robot_python
    if interpreter is None:
        return dict(completed=False, error=f"No private Python interpreter supplied for {stage}")
    (args.out / f"{stage}.json").unlink(missing_ok=True)
    if stage == "sam":
        (args.out / "robot_masks.npz").unlink(missing_ok=True)
    command = [
        str(interpreter),
        "-m",
        "rcr.perception.challenge.physical_frontends",
        stage,
        "--out",
        str(args.out),
    ]
    if stage == "sam":
        command += ["--checkpoint", str(args.sam3_checkpoint)]
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[3]))
    with (args.out / f"{stage}.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
    path = args.out / f"{stage}.json"
    return (
        json.loads(path.read_text())
        if path.exists()
        else dict(completed=False, returncode=result.returncode, error=f"See {stage}.log")
    )


def merge_static_map(empty, occupied, images, cameras):
    first = [dict(h) for h in empty if h["image_supported"]]
    second = [dict(h) for h in occupied if h["image_supported"]]
    matches = []
    if first and second:
        distance = np.linalg.norm(
            np.array([h["socket_hypothesis_wall_m"] for h in first])[:, None, :2]
            - np.array([h["socket_hypothesis_wall_m"] for h in second])[None, :, :2],
            axis=2,
        )
        same_color = (
            np.array([h["color"] for h in first])[:, None]
            == np.array([h["color"] for h in second])[None, :]
        )
        valid = (distance < 0.020) & same_color
        rows, columns = linear_sum_assignment(np.where(valid, distance, 1e3))
        matches = [
            (int(i), int(j), float(distance[i, j]))
            for i, j in zip(rows, columns, strict=True)
            if valid[i, j]
        ]
    by_first = {i: (j, residual) for i, j, residual in matches}
    used = {j for _, j, _ in matches}
    for i, hold in enumerate(first):
        hold["id"] = f"cup_{i:03d}"
        hold["source"] = "empty-wall RGB/CAD map"
        hold["reobserved_with_robot"] = i in by_first
        if i in by_first:
            hold["with_robot_association_residual_m"] = by_first[i][1]
    rejected = []
    for j, hold in enumerate(second):
        if j in used:
            continue
        available, supported = 0, 0
        for image, camera in zip(images, cameras, strict=True):
            point = project_wall(
                [hold["socket_hypothesis_wall_m"]], camera["K"], camera["T_camera_wall"]
            )[0]
            x, y = np.round(point).astype(int)
            if x < 12 or y < 12 or x >= image.shape[1] - 12 or y >= image.shape[0] - 12:
                continue
            available += 1
            hsv = cv2.cvtColor(image[y - 12 : y + 13, x - 12 : x + 13], cv2.COLOR_RGB2HSV)
            hue = (
                hsv[..., 0] < 30
                if hold["color"] == "orange"
                else (hsv[..., 0] >= 90) & (hsv[..., 0] <= 125)
            )
            supported += int(np.count_nonzero(hue & (hsv[..., 1] > 110) & (hsv[..., 2] > 50)) >= 15)
        if (
            len(hold["observed_by"]) < 2
            or hold["boundary_loss_px2"] > 4
            or (available and not supported)
        ):
            rejected.append(
                dict(
                    hold=hold,
                    reason="New occupied-wall candidate lacks two-view/empty-wall support",
                )
            )
            continue
        value = dict(
            hold,
            id=f"cup_{len(first):03d}",
            source="new two-view RGB/CAD observation",
            reobserved_with_robot=True,
        )
        first.append(value)
    return dict(
        schema="physical-cup-map-review-v1",
        frame="provisional_board",
        axes="x right, y up, z out from the wall",
        origin="bottom left, conditional on declared grid indexing and edge offsets",
        units="metres",
        holds=first,
        reobserved=len(matches),
        new_candidates_rejected=rejected,
        assumptions=[
            "Wall holds remain fixed between recordings",
            "No hidden or unobserved holds are inferred from a route",
        ],
        valid_for_control=False,
    )


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    for filename in ["summary.json", "overview.html", "real_images_report.pdf", "wall_map.json"]:
        (args.out / filename).unlink(missing_ok=True)
    source = source_directory(args.input, args.out)
    geometry = dict(
        wall_width_m=args.wall_width,
        wall_height_m=args.wall_height,
        hole_pitch_m=args.pitch,
        first_hole_left_m=args.first_hole_left,
        first_hole_top_m=args.first_hole_top,
    )
    metadata, images, cameras = {}, {}, {}
    for name in NAMES:
        folder = args.out / "decoded" / name
        metadata[name] = decode_recording(source / (name + ".bag"), folder)
        if np.any(metadata[name]["distortion"]):
            raise ValueError(
                "This recording adapter requires zero-distortion RGB; rectify other profiles first"
            )
        image = np.array(Image.open(folder / "rgb.png"))
        camera, correspondences = calibrate_perforated_board(image, metadata[name]["K"], geometry)
        camera["repeatability"] = camera_stability(
            folder, metadata[name], camera, correspondences, geometry
        )
        save(folder / "calibration.json", camera)
        np.savez_compressed(folder / "grid.npz", **correspondences)
        overlay = image.copy()
        for point in correspondences["pixels"][correspondences["inliers"]]:
            cv2.circle(overlay, tuple(np.round(point).astype(int)), 2, (0, 220, 80), 1)
        Image.fromarray(overlay).save(folder / "grid.jpg")
        images[name], cameras[name] = image, camera
        print(
            json.dumps(
                dict(
                    stage="camera",
                    recording=name,
                    reprojection_rms_px=camera["reprojection_rms_px"],
                )
            ),
            flush=True,
        )
    save(args.out / "recordings.json", metadata)
    boxes = {}
    foreground_errors = {}
    for side in ["l", "r"]:
        name, background = side + "_with_rob", side + "_without_rob"
        try:
            boxes[name] = foreground_box(
                images[name], images[background], cameras[name], cameras[background], geometry
            )
            x0, y0, x1, y1 = boxes[name]
            Image.fromarray(images[name][y0:y1, x0:x1]).save(args.out / (name + "_crop.png"))
        except ValueError as error:
            foreground_errors[name] = str(error)
    save(args.out / "robot_crops.json", boxes)
    robot_names = ["l_with_rob", "r_with_rob"]
    try:
        if foreground_errors:
            raise ValueError(f"Robot foreground unavailable: {foreground_errors}")
        label = label_hypothesis(
            [images[n] for n in robot_names],
            [boxes[n] for n in robot_names],
            [cameras[n] for n in robot_names],
        )
        save(args.out / "label_hypothesis.json", label)
    except ValueError as error:
        label = dict(error=str(error), valid_for_control=False)
        save(args.out / "label_hypothesis.json", label)
    sam = (
        frontend(args, "sam")
        if args.sam3_python and not foreground_errors
        else dict(completed=False, error="SAM 3 not requested or robot foreground unavailable")
    )
    learned = (
        frontend(args, "keypoints")
        if args.robot_python and not foreground_errors
        else dict(completed=False, error="Learned frontend not requested or foreground unavailable")
    )
    masks = None
    if sam["completed"]:
        with np.load(args.out / "robot_masks.npz") as archive:
            masks = {n: archive[n] for n in robot_names}
    views = {}
    for name in NAMES:
        instances, rectified = cup_instances(
            images[name], cameras[name], geometry, masks.get(name) if masks else None
        )
        views[name] = instances
        Image.fromarray(rectified).save(args.out / "decoded" / name / "rectified.jpg")
    scale, scale_trials = choose_scale(
        views["l_without_rob"], cameras["l_without_rob"], args.out / "cache"
    )
    save(args.out / "cad_scale.json", dict(selected=scale, trials=scale_trials))
    maps = {}
    for state in ["without_rob", "with_rob"]:
        names = [side + "_" + state for side in ["l", "r"]]
        holds = map_cups(
            [views[n] for n in names], [cameras[n] for n in names], scale, args.out / "cache"
        )
        maps[state] = dict(
            frame="provisional_board",
            axes="x right, y up, z out from the wall",
            units="metres",
            holds=holds,
            view_counts=[len(views[n]) for n in names],
            image_supported=sum(h["image_supported"] for h in holds),
            scope="Image/CAD support counts; no reference hold map or accuracy measurement",
        )
        save(args.out / (state + "_map.json"), maps[state])
        print(
            json.dumps(
                dict(stage="map", state=state, image_supported=maps[state]["image_supported"])
            ),
            flush=True,
        )
    fused = merge_static_map(
        maps["without_rob"]["holds"],
        maps["with_rob"]["holds"],
        [images[n] for n in ["l_without_rob", "r_without_rob"]],
        [cameras[n] for n in ["l_without_rob", "r_without_rob"]],
    )
    save(args.out / "wall_map.json", fused)
    trunk = (
        frontend(args, "trunk")
        if masks and "error" not in label
        else dict(
            completed=False,
            error="Rigid-trunk fitting requires robot masks and a visible label initializer",
        )
    )
    separation = {
        state: abs(
            metadata["l_" + state]["selected"]["timestamp_ms"]
            - metadata["r_" + state]["selected"]["timestamp_ms"]
        )
        / 1000
        for state in ["without_rob", "with_rob"]
    }
    summary = dict(
        schema="physical-perception-review-v1",
        frame="provisional_board",
        axes="x right, y up, z out from the wall",
        units="metres",
        geometry=geometry,
        metric_origin_verified=False,
        camera_reprojection_rms_px={n: cameras[n]["reprojection_rms_px"] for n in NAMES},
        capture_separation_s=separation,
        synchronized_pair=False,
        robot_static_pose_assumed=True,
        map_counts={state: maps[state]["image_supported"] for state in maps},
        fused_map_count=len(fused["holds"]),
        reobserved_cups=fused["reobserved"],
        sam3=sam,
        learned_frontend=learned,
        robot_trunk=trunk,
        marker=label,
        configured_robot=dict(available=False, reason="No measured joint-angle log supplied"),
        valid_for_control=False,
        team_pose=None,
        scope=(
            "Real RGB replay with declared geometry; no surveyed pose/hold references, "
            "contact state or closed-loop execution"
        ),
    )
    save(args.out / "summary.json", summary)
    from .physical_report import build_report

    build_report(args.out, images, cameras, maps, fused, summary)
    print(
        json.dumps(
            dict(
                completed=True,
                output=str(args.out),
                fused_map_count=len(fused["holds"]),
                root_hypothesis=trunk.get("hypothesis_available", False),
            )
        ),
        flush=True,
    )
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--wall-width", type=float, default=0.8)
    parser.add_argument("--wall-height", type=float, default=1.4)
    parser.add_argument("--pitch", type=float, default=0.015)
    parser.add_argument("--first-hole-left", type=float, default=0.010)
    parser.add_argument("--first-hole-top", type=float, default=0.010)
    parser.add_argument("--sam3-python", type=Path)
    parser.add_argument("--sam3-checkpoint", type=Path)
    parser.add_argument("--robot-python", type=Path)
    parser.add_argument("--gpu-confirmed", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.sam3_python and args.sam3_checkpoint is None:
        parser.error("--sam3-python requires an explicit --sam3-checkpoint")
    if not args.gpu_confirmed:
        command = [
            sys.executable,
            "-m",
            "rcr.perception.challenge.physical_run",
            *sys.argv[1:],
            "--gpu-confirmed",
        ]
        guarded = [
            sys.executable,
            "-m",
            "rcr.perception.challenge.gpu_guard",
            "--state",
            str(args.out / "gpu.json"),
            "--log",
            str(args.out / "run.log"),
            "--peak-mib",
            "12000" if args.sam3_python else "5000",
            "--",
            *command,
        ]
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[3]))
        return subprocess.call(guarded, env=env)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
