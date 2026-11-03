"""Local image classification for choosing among fitted CAD hypotheses.

The classifier consumes centered image crops. Training labels and scene seed
metadata are absent from the runtime interface.
"""

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

CLASSES = ("background", "microban_hold", "sidepull_hold", "fingerpocket_hold")
MODEL_PATH = Path(__file__).parent / "assets/weights/hold_crop_forest.npz"
YAW_MODEL_PATH = Path(__file__).parent / "assets/weights/finger_yaw_forest.npz"


def features(rgb, center):
    """Use local appearance only, without image coordinates or scene identity."""
    vector = []
    for side in (24, 40):
        crop = cv2.getRectSubPix(rgb, (side, side), tuple(map(float, center)))
        crop = cv2.resize(crop, (24, 24), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV).astype(np.float32)
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        dx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0)
        dy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1)
        angles = np.arctan2(dy, dx) % np.pi
        bins = np.minimum((angles * (8 / np.pi)).astype(int), 7)
        magnitude = np.hypot(dx, dy)
        yy, xx = np.mgrid[:24, :24]
        histogram = np.zeros((6, 6, 8), np.float32)
        np.add.at(histogram, (yy // 4, xx // 4, bins), magnitude)
        histogram /= np.sqrt(np.square(histogram).sum(axis=-1, keepdims=True) + 1e-5)
        vector.extend(histogram.ravel())
        # Saturation and intensity make the model independent of paint hue.
        small = cv2.resize(hsv[:, :, 1:], (12, 12), interpolation=cv2.INTER_AREA)
        vector.extend((small / 255).ravel())
    return np.asarray(vector, dtype=np.float32)


@lru_cache(maxsize=1)
def classifier():
    with np.load(MODEL_PATH) as arrays:
        return [
            tuple(
                arrays[f"{name}_{i}"].copy()
                for name in ("feature", "threshold", "left", "right", "value")
            )
            for i in range(int(arrays["trees"]))
        ]


def probabilities(rgb, center):
    x = features(rgb, center)
    result = np.zeros(len(CLASSES), dtype=float)
    for feature, threshold, left, right, value in classifier():
        node = 0
        while feature[node] >= 0:
            node = left[node] if x[feature[node]] <= threshold[node] else right[node]
        result += value[node]
    return result / max(result.sum(), 1)


@lru_cache(maxsize=1)
def yaw_model():
    with np.load(YAW_MODEL_PATH) as arrays:
        return [
            tuple(
                arrays[f"{name}_{i}"].copy()
                for name in ("feature", "threshold", "left", "right", "value")
            )
            for i in range(int(arrays["trees"]))
        ]


def finger_yaw(rgb, center):
    """Estimate wall Fingerpocket yaw from image appearance; no pose labels."""
    x = features(rgb, center)
    estimates = []
    for feature, threshold, left, right, value in yaw_model():
        node = 0
        while feature[node] >= 0:
            node = left[node] if x[feature[node]] <= threshold[node] else right[node]
        estimates.append(value[node])
    return float(np.mean(estimates))


def postprocess(
    holds,
    debug,
    raw,
    calibration,
    *,
    background_threshold=0.85,
    type_threshold=0.65,
    finger_yaw_policy="unchanged",
):
    """Choose supported CAD types; retain uncertain image estimates."""
    from scipy.spatial.transform import Rotation

    from .src.rcr_methods import _calibration, _project, normalized_mesh

    k, t = _calibration(calibration)
    image = raw[:, : raw.shape[1] // 2]
    observed = None
    if any("centers" not in r for r in debug.get("objects", [])):
        from .photo_frontend import appearance, canonical_rgb
        from .src.hold_scene import proposals

        real = appearance(raw)[0] == "real"
        if real and debug.get("options", {}).get("masks"):
            from .improved_masks import canonical_rgb

        normalized = canonical_rgb(raw) if real else raw
        observed = proposals(normalized[:, : normalized.shape[1] // 2])
    centers = {
        kind: normalized_mesh(kind).vertices.mean(0) for kind in {h["type"] for h in holds.values()}
    }
    result = {}
    records = []
    diagnostics = []
    for name, hold in holds.items():
        index = int(name.rsplit("_", 1)[-1]) - 1
        record = debug["objects"][index] if index < len(debug.get("objects", [])) else {}
        r = Rotation.from_euler(
            "XYZ", [hold["orientation"][a] for a in ("roll", "pitch", "yaw")]
        ).as_matrix()
        point = np.array([hold["position"][a] for a in "xyz"])
        center, _ = _project(point + r @ centers[hold["type"]], k[0], t[0])
        if "centers" in record:
            center = record["centers"][0]
        elif observed is not None and record.get("left_proposal", -1) in range(len(observed)):
            center = observed[record["left_proposal"]]["center"]
        else:
            center = center - 0.5
        probability = probabilities(image, center)
        choice = int(np.argmax(probability))
        selected = CLASSES[choice]
        action = "retain"
        if selected == "background" and probability[0] >= background_threshold:
            action = "reject"
        elif (
            selected != "background"
            and selected != hold["type"]
            and probability[choice] >= type_threshold
        ):
            candidates = [h for h in record.get("hypotheses", []) if h.get("kind") == selected]
            if candidates:
                best = min(candidates, key=lambda h: h["loss"])
                hold = dict(
                    type=selected,
                    color=hold["color"],
                    position=best["position"],
                    orientation=best["orientation"],
                )
                record = {**record, **best, "accepted": True}
                action = "change_type"
        if action != "reject":
            if hold["type"] == "fingerpocket_hold" and abs(hold["position"]["z"]) < 0.002:
                if finger_yaw_policy == "learned":
                    hold = {
                        **hold,
                        "orientation": {**hold["orientation"], "yaw": finger_yaw(image, center)},
                    }
                elif finger_yaw_policy == "half":
                    hold = {
                        **hold,
                        "orientation": {
                            **hold["orientation"],
                            "yaw": hold["orientation"]["yaw"] * 0.5,
                        },
                    }
            result[f"hold_{len(result) + 1}"] = hold
            records.append(record)
        diagnostics.append(dict(hold=name, probabilities=probability.tolist(), action=action))
    return result, {**debug, "objects": records, "crop_classifier": diagnostics}
