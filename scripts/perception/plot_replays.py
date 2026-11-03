#!/usr/bin/env python3
"""Plot saved root-pose regressions; evaluation labels are used only here."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from rcr.perception.challenge.robot_pose.frame_conventions import apply, project
    from rcr.perception.challenge.robot_pose.model_adapter import RobotModel

    model = RobotModel()
    records = json.loads((args.predictions / "verification.json").read_text())["records"]
    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    fig, axes = plt.subplots(1, len(records), figsize=(11, 3.4), squeeze=False)
    for ax, record in zip(axes.flat, records, strict=True):
        ident = record["case"]
        scene = args.dataset / "scenes" / ident
        packet = json.loads((scene / "input.json").read_text())
        reference = json.loads((scene / "evaluation.json").read_text())
        value = json.loads((args.predictions / ident / "perception.json").read_text())["prediction"]
        pose = value.get("robot_pose", value)
        points = np.array(list(model.landmarks(reference["joint_positions_rad"]).values()))
        camera = reference["cameras"][0]
        ax.imshow(Image.open(scene / packet["cameras"][0]["rgb"]))
        truth = project(apply(np.asarray(reference["T_wall_trunk"]), points), camera)
        estimated = project(apply(np.asarray(pose["T_wall_trunk"]), points), camera)
        for uv, color, size in [(truth, "tab:orange", 35), (estimated, "tab:blue", 12)]:
            ax.scatter(*uv.T, s=size, facecolors="none", edgecolors=color, linewidths=0.8)
        lower, upper = truth.min(0) - 35, truth.max(0) + 35
        ax.set_xlim(lower[0], upper[0])
        ax.set_ylim(upper[1], lower[1])
        ax.set_title(f"{ident}\n{record['position_mm']:.2f} mm / {record['rotation_deg']:.2f}°")
        ax.axis("off")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
