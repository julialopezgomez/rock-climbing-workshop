"""RGB-derived crop shared by the trained landmark frontends."""

import numpy as np

from ..configured_fit import segment_robot


def image_crop(image, camera, margin=0.3):
    _, box = segment_robot(image, camera)
    x0, y0, x1, y1 = np.asarray(box, float)
    side = max(x1 - x0 + 1, y1 - y0 + 1) * (1 + 2 * margin)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0, x1 = max(0, int(cx - side / 2)), min(image.shape[1], int(cx + side / 2) + 1)
    y0, y1 = max(0, int(cy - side / 2)), min(image.shape[0], int(cy + side / 2) + 1)
    return image[y0:y1, x0:x1].copy(), np.array([x0, y0]), np.array(box)
