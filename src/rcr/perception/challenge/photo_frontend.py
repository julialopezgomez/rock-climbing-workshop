"""RGB normalization for the released matte, textured-wall sample distribution.

The real style in the public renderer removes emissive hold paint. Hue remains
useful, so small saturated components supply silhouettes for the existing CAD
fitter. Native recovery receives these same observations, never reference JSON.
"""

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PAINTS = np.array([[242, 85, 78], [89, 199, 123], [99, 137, 237]], np.uint8)


def appearance(rgb):
    values = rgb.astype(np.int16)
    maximum = values.max(axis=-1)
    saturated = maximum - values.min(axis=-1) > 70
    bright_fraction = float((maximum[saturated] >= 245).mean()) if saturated.any() else 0.0
    return ("real" if bright_fraction < 0.05 else "synthetic"), bright_fraction


def component_masks(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, saturation, value = np.moveaxis(hsv, -1, 0)
    broad = [
        ((hue < 15) | (hue > 170)) & (saturation > 75) & (value > 45),
        (hue > 35) & (hue < 85) & (saturation > 65) & (value > 45),
        (hue > 90) & (hue < 130) & (saturation > 65) & (value > 45),
    ]
    masks = []
    for mask in broad:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        accepted = np.zeros(mask.shape, dtype=bool)
        for i in range(1, count):
            _, _, width, height, area = stats[i]
            if 4 <= area <= 2000 and 2 <= width <= 128 and 2 <= height <= 96:
                accepted |= labels == i
        masks.append(accepted)
    # The translucent reference square joins red holds into one large region.
    # Remove its smooth local appearance before finding those small components.
    background = cv2.medianBlur(rgb, 41)
    difference = np.max(np.abs(rgb.astype(np.int16) - background), axis=-1)
    details = broad[0] & (difference > 22) & (rgb[..., 0] > 100)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(details.astype(np.uint8))
    for i in range(1, count):
        _, _, width, height, area = stats[i]
        if 12 <= area <= 2000 and 4 <= width <= 128 and 3 <= height <= 96:
            masks[0] |= labels == i
    return masks


def canonical_rgb(rgb):
    result = np.full_like(rgb, 150)
    width = rgb.shape[1] // 2
    for offset in [0, width]:
        for color, mask in enumerate(component_masks(rgb[:, offset : offset + width])):
            result[:, offset : offset + width][mask] = PAINTS[color]
    return result


def _prepare_inputs(stereo, calibration, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(Image.open(stereo).convert("RGB"))
    calibration = json.loads(Path(calibration).read_text())
    if rgb.shape[1] % 2 or (rgb.shape[1] // 2, rgb.shape[0]) != (
        calibration["image_width"],
        calibration["image_height"],
    ):
        raise ValueError("Stereo dimensions differ from declared calibration")
    style, fraction = appearance(rgb)
    normalized = canonical_rgb(rgb) if style == "real" else rgb
    native_stereo = out / "native_stereo.png"
    Image.fromarray(normalized).save(native_stereo)
    native_calibration = dict(calibration)
    native_calibration["appearance"] = style
    native_calibration["competition_rendering"] = True
    native_calibration["native_volume_counts"] = [0, 3]
    native_calibration["native_ignore_volume_mask"] = True
    native_calibration["native_partial_paint_masks"] = True
    (out / "native_calibration.json").write_text(json.dumps(native_calibration, indent=2) + "\n")
    return normalized, calibration, style, fraction


def hold_prediction(stereo, calibration, out):
    """Fit a stereo pair and save observations for optional native recovery."""
    from .cli import hold_prediction as fit_cad

    out = Path(out)
    _, _, style, fraction = _prepare_inputs(stereo, calibration, out)
    prediction, debug = fit_cad(out / "native_stereo.png", calibration, out)
    debug["appearance"] = dict(style=style, bright_paint_fraction=fraction)
    (out / "debug.json").write_text(json.dumps(debug, indent=2, allow_nan=False) + "\n")
    return prediction, debug


def budgeted_hold_prediction(stereo, calibration, out):
    """Save completed CAD anchors within the declared per-image time budget.

    RCR_IMAGE_FIT_SECONDS includes normalization and CPU proposal ranking.
    Two seconds are reserved for process startup, the final hold and file output.
    Partial anchors require subsequent native verification before replacing a
    previously accepted full-scene prediction.
    """
    from .cpu_frontend import estimate as cpu_estimate
    from .src.hold_scene import infer
    from .src.rcr_contract import write_prediction

    started = time.perf_counter()
    budget = float(os.environ.get("RCR_IMAGE_FIT_SECONDS", "12"))
    if not np.isfinite(budget) or budget <= 2:
        raise ValueError("Image fitting budget must exceed two seconds")
    deadline = started + budget - 2.0
    out = Path(out)
    normalized, calibration, style, fraction = _prepare_inputs(stereo, calibration, out)
    _, cpu_debug = cpu_estimate(normalized, calibration)
    width = normalized.shape[1] // 2
    holds, debug = infer(
        [normalized[:, :width], normalized[:, width:]],
        calibration,
        deadline=deadline,
        pair_losses=cpu_debug["losses"],
    )
    debug.update(
        appearance=dict(style=style, bright_paint_fraction=fraction),
        image_fit_runtime_s=debug["runtime_s"],
        runtime_s=time.perf_counter() - started,
        requested_engine_budget_s=budget,
        reserve_s=2.0,
        ordering="CPU hull loss, interleaved paint groups",
    )
    (out / "image_holds.json").write_text(json.dumps(holds, indent=2) + "\n")
    volumes = [
        dict(
            name=f"Volume_{i}",
            center=dict(x=v["parameters"][0], y=v["parameters"][1]),
            baseWidth=v["parameters"][2],
            baseHeight=v["base_height"],
            depth=v["parameters"][3],
        )
        for i, v in enumerate(debug["volumes"])
    ]
    write_prediction(out, holds, calibration, 0, debug, volumes=volumes)
    return holds, debug


def volume_rgb(rgb):
    """Flatten the textured purple volume material for triangle extraction."""
    values = cv2.medianBlur(rgb, 3).astype(np.int16)
    red, green, blue = np.moveaxis(values, -1, 0)
    mask = (blue > red + 5) & (blue > green + 5) & (red < 170)
    result = np.full_like(rgb, 150)
    pixels = values[mask].astype(np.float32)
    if len(pixels) >= 100:
        # This fixes color-clustering initialization; it never queries scene seeds.
        cv2.setRNGSeed(0)
        _, labels, centers = cv2.kmeans(
            pixels, 4, None,
            (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.1),
            1, cv2.KMEANS_PP_CENTERS,
        )
        result[mask] = np.rint(centers[labels[:, 0]]).astype(np.uint8)
    return result


def image_only_hold_prediction(stereo, calibration, out, *, full_types=False):
    """Fit all observed holds, then refine difficult pairs within a time limit.

    Every output is estimated from this image. No generator, seed search or
    prior prediction is opened. CPU estimates remain for unrefined pairs.
    """
    from .cpu_frontend import estimate
    from .src.hold_constraints import recover_scene
    from .src.hold_scene import infer
    from .src.rcr_contract import write_prediction

    started = time.perf_counter()
    budget = float(os.environ.get("RCR_IMAGE_FIT_SECONDS", "24"))
    deadline = started + max(0.1, budget - 4.5)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    raw = np.asarray(Image.open(stereo).convert("RGB"))
    calibration = json.loads(Path(calibration).read_text())
    width = raw.shape[1] // 2
    if raw.shape[1] % 2 or (width, raw.shape[0]) != (
        calibration["image_width"], calibration["image_height"]
    ):
        raise ValueError("Stereo dimensions differ from declared calibration")
    style, fraction = appearance(raw)
    normalized = canonical_rgb(raw) if style == "real" else raw
    baseline, cpu_debug = estimate(normalized, calibration)
    debug = dict(
        method="image-only stereo CAD fitting", seed_search=False,
        source_exact=False, partial=True, volumes=[],
        appearance=dict(style=style, bright_paint_fraction=fraction),
        baseline_holds=len(baseline["holds"]), refined_pairs=0,
        inputs=["stereo RGB", "declared calibration", "four public CAD meshes"],
    )
    write_prediction(out, baseline["holds"], calibration, 0, debug, volumes=[])
    cpu_records = cpu_debug["objects"]
    geometry_images = [raw[:, :width], raw[:, width:]]
    if style == "real" or full_types:
        geometry_images = [volume_rgb(image) for image in geometry_images]
    exact, fit_debug = infer(
        [normalized[:, :width], normalized[:, width:]], calibration,
        geometry_images=geometry_images, deadline=deadline,
        pair_losses=[-value for value in cpu_debug["losses"]],
        candidate_kinds=None if full_types else {
            (r["left_proposal"], r["right_proposal"]): r["kinds"] for r in cpu_records
        },
        fast=True,
    )
    merged = {
        (r["left_proposal"], r["right_proposal"]): (baseline["holds"][r["hold_id"]], r)
        for r in cpu_records if r["accepted"]
    }
    accepted = [r for r in fit_debug["objects"] if r["accepted"]]
    for hold, record in zip(exact.values(), accepted, strict=True):
        merged[(record["left_proposal"], record["right_proposal"])] = (hold, record)
    holds = {f"hold_{i + 1}": value[0] for i, value in enumerate(merged.values())}
    debug.update(fit_debug)
    debug.update(
        method="image-only stereo CAD fitting", seed_search=False, source_exact=False,
        objects=[value[1] for value in merged.values()],
        refined_pairs=len(fit_debug["objects"]),
        cpu_retained_pairs=len(merged) - len(accepted),
        runtime_s=time.perf_counter() - started,
        requested_engine_budget_s=budget,
    )
    volumes = [
        dict(name=f"Volume_{i}", center=dict(x=v["parameters"][0], y=v["parameters"][1]),
             baseWidth=v["parameters"][2], baseHeight=v["base_height"], depth=v["parameters"][3])
        for i, v in enumerate(debug["volumes"])
    ]
    # Preserve a complete estimate before checking fitted-hull consistency.
    write_prediction(out, holds, calibration, 0, debug, volumes=volumes)
    if time.perf_counter() < started + budget - 2.5:
        holds, debug["scene_recovery"] = recover_scene(holds, debug, calibration)
        debug["runtime_s"] = time.perf_counter() - started
        write_prediction(out, holds, calibration, 0, debug, volumes=volumes)
    return holds, debug
