#!/usr/bin/env python3
"""Draw scoring overlays for the two retained hold fixtures, after inference."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from rcr.perception.challenge.src.rcr_contract import score_holds  # noqa: E402


def pixel(hold, camera):
    point = np.array([hold["position"][axis] for axis in "xyz"] + [1.0])
    point = np.asarray(camera["T_camera_scene"]) @ point
    point = np.asarray(camera["K"]) @ point[:3]
    return tuple(np.round(point[:2] / point[2]).astype(int))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=REPO / "docs/perception/figures")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for identifier, name in [("000000", "holds"), ("000001", "volumes")]:
        folder = REPO / "data/perception/holds" / identifier
        image = Image.open(folder / "stereo.png").convert("RGB")
        image = image.crop((0, 0, image.width // 2, image.height))
        camera = json.loads((folder / "calibration.json").read_text())["cameras"]["left"]
        expected = json.loads((folder / "expected_holds.json").read_text())
        predicted = json.loads((args.predictions / identifier / "holds.json").read_text())
        metrics = score_holds(predicted, expected)
        strict = {
            match["prediction_id"]
            for match in metrics["matches"]
            if match["translation_error_mm"] < 5 and match["rotation_error_deg"] < 15
        }
        draw = ImageDraw.Draw(image)
        for hold in expected.values():
            x, y = pixel(hold, camera)
            draw.line((x - 3, y, x + 3, y), fill="white", width=1)
            draw.line((x, y - 3, x, y + 3), fill="white", width=1)
        for key, hold in predicted.items():
            x, y = pixel(hold, camera)
            color = (80, 255, 120) if key in strict else (255, 190, 30)
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=color, width=1)
        image.save(args.out / f"{name}.jpg", quality=95)


if __name__ == "__main__":
    main()
