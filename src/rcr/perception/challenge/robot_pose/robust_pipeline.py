"""Optional exposure-normalized wrapper around the frozen robot estimators.

The correction uses only RGB highlights. It is deliberately inactive on the
native training renders, and has no access to reference poses or light settings.
"""

import time

import numpy as np

from .pipeline import RobotPoseEstimator, TrunkPoseEstimator


def normalize_exposure(image):
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Exposure correction expects uint8 RGB")
    histogram = np.bincount(image.max(axis=2).ravel(), minlength=256)
    white = int(np.searchsorted(np.cumsum(histogram), 0.99 * image.shape[0] * image.shape[1]))
    gain = min(2.0, 255.0 / white) if 40 <= white < 220 else 1.0
    adjusted = (
        image
        if gain == 1.0
        else np.clip(np.rint(image.astype(np.float32) * gain), 0, 255).astype(np.uint8)
    )
    return adjusted, dict(
        highlight_percentile=99,
        highlight_value=white,
        gain=gain,
        policy="RGB highlight correction; no light or pose ground truth",
    )


class ExposureNormalizedEstimator:
    def __init__(self, *, profile="max", **options):
        cls = TrunkPoseEstimator if profile == "trunk" else RobotPoseEstimator
        self.actor = cls(profile=profile, **options)

    def infer(self, packet, images, *, reset=False):
        started = time.perf_counter()
        corrected, info = zip(*(normalize_exposure(image) for image in images), strict=False)
        value = self.actor.infer(packet, list(corrected), reset=reset)
        value["method"] = "exposure_normalized_" + value["method"]
        value["diagnostics"].update(
            exposure_correction=list(info), runtime_s=time.perf_counter() - started
        )
        return value

    def close(self):
        self.actor.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
