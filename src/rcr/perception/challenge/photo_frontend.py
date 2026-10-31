"""RGB normalization for the released matte, textured-wall sample distribution.

The real style in the public renderer removes emissive hold paint. Hue remains
useful, so small saturated components supply silhouettes for the existing CAD
fitter. Native recovery receives these same observations, never reference JSON.
"""

import json
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


def hold_prediction(stereo, calibration, out):
    """Fit a stereo pair and save observations for optional native recovery."""
    from .cli import hold_prediction as fit_cad

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(Image.open(stereo).convert("RGB"))
    style, fraction = appearance(rgb)
    normalized = canonical_rgb(rgb) if style == "real" else rgb
    native_stereo = out / "native_stereo.png"
    Image.fromarray(normalized).save(native_stereo)
    native_calibration = json.loads(Path(calibration).read_text())
    native_calibration["appearance"] = style
    native_calibration["competition_rendering"] = True
    native_calibration["native_volume_counts"] = [0, 3]
    native_calibration["native_ignore_volume_mask"] = True
    native_calibration["native_partial_paint_masks"] = True
    (out / "native_calibration.json").write_text(json.dumps(native_calibration, indent=2) + "\n")
    prediction, debug = fit_cad(native_stereo, calibration, out)
    debug["appearance"] = dict(style=style, bright_paint_fraction=fraction)
    (out / "debug.json").write_text(json.dumps(debug, indent=2, allow_nan=False) + "\n")
    return prediction, debug
