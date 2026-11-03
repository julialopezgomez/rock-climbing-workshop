"""Local colour segmentation trained on development CAD mask annotations.

The predictor uses pixel colours and local colour differences. It receives no
scene identity, position, generator state or reference geometry at inference.
"""

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

PAINTS = np.array([[242, 85, 78], [89, 199, 123], [99, 137, 237]], np.uint8)
MODEL = Path(__file__).parent / "assets/weights/paint_forest.npz"


def pixel_features(rgb):
    """RGB, HSV, two local contrasts, and normalized RGB; no coordinates."""
    rgb = np.asarray(rgb, dtype=np.uint8)
    raw = rgb.astype(np.float32)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    small = cv2.medianBlur(rgb, 5).astype(np.float32)
    broad = cv2.medianBlur(rgb, 31).astype(np.float32)
    normalized = raw / (raw.sum(axis=-1, keepdims=True) + 1) * 255
    return np.concatenate([raw, hsv, raw - small, raw - broad, normalized], axis=-1)


@lru_cache(maxsize=1)
def forest():
    if not MODEL.is_file():
        raise FileNotFoundError(f"Paint segmentation model missing: {MODEL}")
    with np.load(MODEL) as arrays:
        return [
            tuple(
                arrays[f"{name}_{i}"].copy()
                for name in ("feature", "threshold", "left", "right", "value")
            )
            for i in range(int(arrays["trees"]))
        ]


def predict_labels(rgb):
    values = pixel_features(rgb)
    hsv = values[..., 3:6]
    candidates = (hsv[..., 1] > 20) & (hsv[..., 2] > 20)
    x = values[candidates]
    result = np.zeros(rgb.shape[:2], np.uint8)
    if not len(x):
        return result
    votes = np.zeros((len(x), 4), np.float32)
    rows = np.arange(len(x))
    for feature, threshold, left, right, value in forest():
        nodes = np.zeros(len(x), np.int32)
        for _ in range(17):
            active = feature[nodes] >= 0
            if not active.any():
                break
            index = rows[active]
            current = nodes[active]
            go_left = x[index, feature[current]] <= threshold[current]
            nodes[active] = np.where(go_left, left[current], right[current])
        votes += value[nodes]
    result[candidates] = votes.argmax(axis=1).astype(np.uint8)
    return result


def component_masks(rgb):
    """Return complete small paint components, including enclosed dark cavities."""
    labels = predict_labels(rgb)
    masks = []
    for color in range(1, 4):
        raw = binary_fill_holes(labels == color).astype(np.uint8)
        count, components, stats, _ = cv2.connectedComponentsWithStats(raw)
        accepted = np.zeros(raw.shape, bool)
        for index in range(1, count):
            _, _, width, height, area = stats[index]
            if 4 <= area <= 2000 and 2 <= width <= 128 and 2 <= height <= 96:
                accepted |= components == index
        masks.append(accepted)
    return masks


def canonical_rgb(rgb):
    """Convert a stereo pair to the standard paint palette for CAD fitting."""
    result = np.full_like(rgb, 150)
    width = rgb.shape[1] // 2
    for offset in (0, width):
        for color, mask in enumerate(component_masks(rgb[:, offset : offset + width])):
            result[:, offset : offset + width][mask] = PAINTS[color]
    return result
