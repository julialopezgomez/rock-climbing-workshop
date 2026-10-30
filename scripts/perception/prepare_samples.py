#!/usr/bin/env python3
"""Prepare supplied stereo archives with an explicitly declared camera/wall rig."""

import argparse
import io
import json
import math
import re
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


def calibration(width, height, args):
    cameras = {}
    focal = height / (2 * math.tan(math.radians(args.fov) / 2))
    for eye, offset in [("left", -args.baseline / 2), ("right", args.baseline / 2)]:
        position = np.array([offset, args.camera_height, args.distance])
        forward = np.array([0, args.camera_height, 0]) - position
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0, 1, 0])
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        rotation = np.stack([right, down, forward])
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = -rotation @ position
        cameras[eye] = dict(
            width=width,
            height=height,
            K=[[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]],
            T_camera_scene=transform.tolist(),
            distortion=[0] * 5,
        )
    return dict(
        schema="rcr-calibration-v1",
        frame="scene",
        units="metres",
        image_width=width,
        image_height=height,
        cameras=cameras,
        wall_width_m=args.wall_width,
        wall_height_m=args.wall_height,
        baseline_m=args.baseline,
        camera_height_m=args.camera_height,
        camera_distance_m=args.distance,
        provenance="Declared public camera/wall preset; not fitted to scene labels",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--wall-width", type=float, default=3.0)
    parser.add_argument("--wall-height", type=float, default=1.4)
    parser.add_argument("--camera-height", type=float, default=0.7)
    parser.add_argument("--distance", type=float, default=1.0)
    parser.add_argument("--baseline", type=float, default=0.064)
    parser.add_argument("--fov", type=float, default=74.0)
    args = parser.parse_args()
    if (
        not 0 < args.fov < 180
        or min(args.wall_width, args.wall_height, args.distance, args.baseline) <= 0
    ):
        parser.error("Use positive metric dimensions and a field of view between 0 and 180 degrees")
    cases = []
    for archive in sorted(args.samples.glob("*.zip")):
        with zipfile.ZipFile(archive) as source:
            for name in sorted(source.namelist()):
                match = re.fullmatch(r"(.*/)?([A-Za-z0-9_]+)_cam1_(\d+)\.png", name)
                if not match:
                    continue
                directory, prefix, number = match.groups()
                label = (directory or "") + prefix + "_" + number + ".json"
                identifier = prefix + "_" + number
                folder = args.out / identifier
                if folder.exists():
                    raise FileExistsError(f"Refusing to overwrite {folder}")
                image = Image.open(io.BytesIO(source.read(name))).convert("RGB")
                if image.width % 2:
                    raise ValueError(f"Unequal stereo panels in {name}")
                folder.mkdir(parents=True)
                image.save(folder / "stereo.png")
                rig = calibration(image.width // 2, image.height, args)
                (folder / "calibration.json").write_text(json.dumps(rig, indent=2) + "\n")
                # Labels are separate scoring files; no label fields enter calibration.
                expected = json.loads(source.read(label))
                expected = expected.get("holds", expected)
                (folder / "expected_holds.json").write_text(json.dumps(expected, indent=2) + "\n")
                cases.append(
                    dict(id=identifier, split="development" if int(number) <= 2 else "validation")
                )
    if not cases:
        raise ValueError("No matching cam1 stereo PNG/JSON pairs found")
    (args.out / "manifest.json").write_text(
        json.dumps(
            dict(
                cases=cases,
                scope="Supplied scenes; first two per archive used for development",
            ),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(dict(cases=len(cases), output=str(args.out))))


if __name__ == "__main__":
    main()
