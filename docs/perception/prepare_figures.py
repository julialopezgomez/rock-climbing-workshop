#!/usr/bin/env python3
"""Build report figures from saved experiment artifacts. No inference is run."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from PIL import Image

for font in Path("/usr/share/texmf/fonts/opentype/public/lm").glob("lmroman10-*.otf"):
    font_manager.fontManager.addfont(str(font))


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "development/outputs/perception/final"
FIGURES = HERE / "figures"
DARK, MID, LIGHT = "0.15", "0.50", "0.80"
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": "black",
        "text.color": "black",
        "savefig.facecolor": "white",
        "axes.titleweight": "normal",
    }
)


def read(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def payload(name: str) -> dict:
    text = (RESULTS / name).read_text()
    return json.JSONDecoder().raw_decode(text.split("const D=", 1)[1])[0]


def photo(uri: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))).convert("RGB")


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def hold_images(data: dict) -> None:
    case = data["cases"][0]
    photo(case["image"]).save(FIGURES / "stereo_input.jpg", quality=95)
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.3))
    for ax, case in zip(axes.flat, data["cases"], strict=False):
        image = photo(case["image"])
        ax.imshow(image)
        ax.set_xlim(0, image.width // 2)
        for views, color, size in [
            (case["truth_overlay"], "#00c68d", 38),
            (case["methods"]["native"]["overlay"], "#ed318b", 13),
        ]:
            xy = np.array([v["points"][0] for v in views[0]])
            ax.scatter(
                xy[:, 0],
                xy[:, 1],
                s=size,
                facecolors="none",
                edgecolors=color,
                linewidths=0.8,
            )
        ax.set_title(case["id"].replace("_", " "), fontsize=10)
        ax.axis("off")
    axes.flat[-1].axis("off")
    axes.flat[-1].text(
        0.05,
        0.7,
        (
            "Five fixed scenes\nLatest public generator\n\n"
            "Green: reference anchors\nPink: recovered anchors\n\n"
            "Left view shown; both used."
        ),
        va="top",
        fontsize=11,
    )
    fig.tight_layout()
    save(fig, "hold_examples")


def robot_images(data: dict) -> None:
    methods = [
        ("keypoints_flex_known_learned_all", "Keypoints + PnP"),
        ("foundationpose_configured_uniform_serial_all", "FoundationPose"),
        ("adaptive_all_max", "CAD refinement"),
    ]
    fig, axes = plt.subplots(3, 5, figsize=(11, 8.1))
    for col, ident in enumerate(data["example_ids"]):
        c = data["cases"][ident]
        x, y, w, h = c["crops"][0]
        for row, (method, label) in enumerate(methods):
            ax = axes[row, col]
            ax.imshow(photo(c["photos"]["marked"][0]["src"]))
            for xy, color, size in [
                (c["truth"][0], "#00ce98", 20),
                (data["overlays"][method][ident]["views"][0], "#ef268a", 9),
            ]:
                xy = np.asarray(xy)
                if xy.size:
                    ax.scatter(
                        xy[:, 0],
                        xy[:, 1],
                        s=size,
                        facecolors="none",
                        edgecolors=color,
                        linewidths=0.8,
                    )
            ax.set_xlim(x, x + w)
            ax.set_ylim(y + h, y)
            ax.axis("off")
            m = data["overlays"][method][ident]["metrics"]
            ax.set_title(
                f"{ident}\n{m['translation_mm']:.2f} mm / {m['rotation_deg']:.2f}°",
                fontsize=8,
            )
            if col == 0:
                ax.text(
                    -0.07,
                    0.5,
                    label,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=10,
                )
    fig.tight_layout(w_pad=0.6, h_pad=1.2)
    save(fig, "robot_examples")
    c = data["cases"][data["example_ids"][0]]
    image = photo(c["photos"]["markerless"][0]["src"])
    x, y, w, h = c["crops"][0]
    image.crop((x, y, x + w, y + h)).resize((540, int(h / w * 540))).save(
        FIGURES / "robot_cover.png"
    )


def charts(benchmark: dict) -> None:
    names = [
        "FP + FoundationStereo",
        "FP + S2M2",
        "SAM-6D + FoundationStereo",
        "SAM-6D + S2M2",
        "Exact CAD + combined proposals",
        "Learned proposals + exact CAD",
    ]
    values = [447, 278, 164, 120, 975, 976]
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.barh(names[::-1], values[::-1], color="0.45")
    for i, v in enumerate(values[::-1]):
        ax.text(v + 12, i, str(v), va="center")
    ax.set_xlim(0, 1100)
    ax.set_xlabel("Strict hold poses out of 1,000 (same historical 100 scenes)")
    ax.grid(axis="x", alpha=0.12)
    ax.set_axisbelow(True)
    save(fig, "hold_methods")
    chosen = [
        ("keypoints_flex_known_learned_all", "Keypoints + PnP"),
        ("foundationpose_configured_uniform_serial_all", "FoundationPose + stereo"),
        ("megapose_configured_uniform_all", "MegaPose"),
        ("pipeline_fast_aruco_all", "Fast M3T pipeline"),
        ("resilient_all_accuracy", "Protected accuracy"),
        ("adaptive_all_max", "CAD refinement"),
    ]
    methods = {v["id"]: v for v in benchmark["original_methods"]}
    fig, ax = plt.subplots(figsize=(9, 3.6))
    yy = np.arange(len(chosen))
    ax.barh(
        yy - 0.17,
        [methods[k]["strict_5mm_5deg"] for k, _ in chosen],
        height=0.32,
        color=DARK,
        label="Root",
    )
    ax.barh(
        yy + 0.17,
        [methods[k]["endpoint_success_5mm"] for k, _ in chosen],
        height=0.32,
        color=MID,
        label="All endpoints",
    )
    for i, (k, _) in enumerate(chosen):
        for offset, field in [
            (-0.17, "strict_5mm_5deg"),
            (0.17, "endpoint_success_5mm"),
        ]:
            v = methods[k][field]
            ax.text(v + 1, i + offset, str(v), va="center", fontsize=8)
    ax.set_yticks(yy, [label for _, label in chosen])
    ax.invert_yaxis()
    ax.set_xlim(0, 112)
    ax.set_xlabel("Successful scenes out of the same original 100")
    ax.legend(loc="lower right")
    save(fig, "robot_methods")
    fig, ax = plt.subplots(figsize=(8.7, 3.2))
    labels = [
        "Full-silhouette fitting\ncutout, set 1",
        "Boundary fallback\ncutout, set 1",
        "Boundary fallback\ncutout, set 2",
    ]
    xx = np.arange(3)
    ax.bar(xx - 0.16, [0, 100, 100], 0.30, color=DARK, label="Root")
    ax.bar(xx + 0.16, [0, 99, 100], 0.30, color=MID, label="All endpoints")
    ax.set_xticks(xx, labels)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Successes / 100")
    ax.legend(loc="lower left")
    save(fig, "occlusion_results")


def live_figures(data: dict) -> None:
    m = data["initial"]["methods"]["boundary"]
    series = m["series"]
    time = np.array([s["time"] for s in series])
    time -= time[0]
    fig, axes = plt.subplots(2, 1, figsize=(9, 4.9), sharex=True)
    for ax, key, label in zip(
        axes,
        ["translation_mm", "endpoint_max_mm"],
        ["Root error (mm)", "Worst endpoint (mm)"],
        strict=False,
    ):
        for stage, color, label_stage in [
            ("before", "0.65", "Visual estimate"),
            ("after", DARK, "With socket prior"),
        ]:
            ax.plot(
                time,
                [s[stage].get(key, np.nan) for s in series],
                color=color,
                lw=0.9,
                label=label_stage,
            )
        ax.axhline(5, color="black", lw=0.8, ls="--", label="5 mm diagnostic threshold")
        ax.set_ylabel(label)
        ax.grid(alpha=0.12)
    axes[0].legend(ncol=3, fontsize=8)
    axes[-1].set_xlabel("Time in recorded trajectory (s)")
    fig.tight_layout()
    save(fig, "live_errors")
    examples = data["final"]["methods"]["boundary"]["examples"]
    fig, axes = plt.subplots(1, 5, figsize=(11, 3.8))
    for ax, c in zip(axes, examples, strict=False):
        ax.imshow(photo(c["images"][0]["src"]))
        x, y, w, h = c["crops"][0]
        for name, color in [("truth", "#00ce98"), ("refined", "#ef268a")]:
            xy = np.asarray(c["overlays"][0][name])
            ax.scatter(
                xy[:, 0],
                xy[:, 1],
                s=6,
                facecolors="none",
                edgecolors=color,
                linewidths=0.6,
            )
        ax.set_xlim(x, x + w)
        ax.set_ylim(y + h, y)
        ax.axis("off")
        ax.set_title(f"Frame {c['id']}\n{c['after']['translation_mm']:.2f} mm", fontsize=9)
    fig.tight_layout()
    save(fig, "live_examples")


def main() -> None:
    global ROOT, RESULTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace-root",
        type=Path,
        required=True,
        help="Original experiment workspace containing development/outputs/perception",
    )
    args = parser.parse_args()
    ROOT = args.workspace_root.resolve()
    RESULTS = ROOT / "development/outputs/perception/final"
    FIGURES.mkdir(exist_ok=True)
    benchmark = read("benchmark.json")
    hold_images(payload("holds_overview.html"))
    robot_images(payload("overview.html"))
    charts(benchmark)
    live_figures(payload("live_overview.html"))
    files = [
        "benchmark.json",
        "hold_benchmark.json",
        "readiness.json",
        "overview.html",
        "holds_overview.html",
        "live_overview.html",
    ]
    snapshot = {
        "scope": "Saved results only; this report runs no new experiments.",
        "source_sha256": {
            name: hashlib.sha256((RESULTS / name).read_bytes()).hexdigest() for name in files
        },
        "robot_methods": benchmark["original_methods"],
        "hold_sets": read("hold_benchmark.json"),
        "readiness": read("readiness.json"),
        "historical_source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                ROOT / "perception" / name
                for name in [
                    "RCR_BENCHMARK.md",
                    "RCR_SEARCH_100.md",
                    "RCR_MODEL_TUNING.md",
                    "RCR_S2M2.md",
                    "RCR_MAXXING.md",
                    "FINAL.md",
                ]
            ]
        },
    }
    (HERE / "metrics_snapshot.json").write_text(json.dumps(snapshot, indent=2))
    print(f"Figures written to {FIGURES}")


if __name__ == "__main__":
    main()
