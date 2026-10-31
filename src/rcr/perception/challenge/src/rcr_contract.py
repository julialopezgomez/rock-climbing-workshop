"""Strict RCR hold JSON adapters and explicitly local diagnostic scoring.

The upstream viewer accepts a hold-only mapping or formatVersion1 scene JSON.
Positions are METRES; orientation is intrinsic XYZ (THREE.Euler XYZ), radians.
Local diagnostics remain available for earlier experiments. The subsequently
released comparison score is implemented separately in ``official_score.py``.
"""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

TYPES = {"microban_hold", "cube_hold", "sidepull_hold", "fingerpocket_hold"}
HOLD_FIELDS = {"type", "color", "position", "orientation"}


def validate_holds(data):
    if not isinstance(data, dict):
        raise ValueError("Hold output must be an object keyed by hold ID")
    if len(data) > 50:
        raise ValueError("The upstream viewer accepts at most 50 holds")
    for key, hold in data.items():
        if (
            not isinstance(key, str)
            or not key
            or not isinstance(hold, dict)
            or set(hold) != HOLD_FIELDS
        ):
            raise ValueError("Each named hold must contain exactly type,color,position,orientation")
        if hold["type"] not in TYPES:
            raise ValueError("Unsupported hold type")
        if not isinstance(hold["color"], str) or not re.fullmatch(
            r"#[0-9a-fA-F]{6}", hold["color"]
        ):
            raise ValueError("Hold color must be a six-digit hex RGB string")
        for field, axes in [
            ("position", {"x", "y", "z"}),
            ("orientation", {"roll", "pitch", "yaw"}),
        ]:
            value = hold[field]
            if not isinstance(value, dict) or set(value) != axes:
                raise ValueError(f"{field} has incorrect keys")
            if any(
                isinstance(v, bool)
                or not isinstance(v, (int, float, np.integer, np.floating))
                or not np.isfinite(v)
                for v in value.values()
            ):
                raise ValueError(f"{field} must contain finite numeric values")
    return data


def versioned_prediction(holds, calibration, seed):
    """Schema-compatible envelope with declared public setup, never copied GT.

    The caller may attach image-estimated volumes through write_prediction.
    The hold-only JSON preserves an existing viewer scene when loaded.
    """
    validate_holds(holds)
    width = float(calibration.get("wall_width_m", 0.8))
    height = float(calibration.get("wall_height_m", 1.4))
    marker = 0.1 if width < 2 else 1.0
    wall = copy.deepcopy(
        calibration.get(
            "wall",
            dict(
                width=width,
                height=height,
                depth=0.025,
                originMarker=dict(
                    width=min(marker, width),
                    height=min(marker, height),
                    x=min(marker, width) / 2,
                    y=min(marker, height) / 2,
                ),
            ),
        )
    )
    return {
        "formatVersion": 1,
        "seed": int(seed),
        "materialSeed": 0,
        "scene": {"wall": wall, "volumes": []},
        "holds": holds,
    }


def write_prediction(folder, holds, calibration, seed, debug=None, *, volumes=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    validate_holds(holds)

    def atomic(name, data):
        text = json.dumps(data, indent=2, allow_nan=False)
        path = folder / name
        temporary = folder / f".{name}.{os.getpid()}.tmp"
        temporary.write_text(text)
        temporary.replace(path)

    atomic("holds.json", holds)
    prediction = versioned_prediction(holds, calibration, seed)
    prediction["scene"]["volumes"] = volumes or []
    atomic("prediction.json", prediction)
    if debug is not None:
        atomic("debug.json", debug)


def vectors(holds):
    ids = list(holds)
    points = np.array(
        [[holds[k]["position"][a] for a in ["x", "y", "z"]] for k in ids], dtype=float
    ).reshape(-1, 3)
    eulers = np.array(
        [[holds[k]["orientation"][a] for a in ["roll", "pitch", "yaw"]] for k in ids], dtype=float
    ).reshape(-1, 3)
    return ids, points, eulers


def score_holds(prediction, truth, tolerance_m=0.05):
    """Cardinality-first typed matching within Euclidean50mm, not official score."""
    validate_holds(prediction)
    validate_holds(truth)
    pids, p, pe = vectors(prediction)
    tids, t, te = vectors(truth)
    matches = []
    if len(p) and len(t):
        distance = np.linalg.norm(p[:, None] - t[None, :], axis=-1)
        same = np.array([[prediction[a]["type"] == truth[b]["type"] for b in tids] for a in pids])
        valid = same & (distance <= tolerance_m)
        reward = np.where(valid, 1 + (1 - distance / tolerance_m) / (min(distance.shape) + 1), 0)
        ii, jj = linear_sum_assignment(-reward)
        for i, j in zip(ii, jj, strict=False):
            if not valid[i, j]:
                continue
            rp = Rotation.from_euler("XYZ", pe[i])
            rt = Rotation.from_euler("XYZ", te[j])
            angle = float((rp * rt.inv()).magnitude())
            delta = p[i] - t[j]
            # Wrapped component differences are reported separately from true
            # SO(3) angle because Euler parameterization is not unique.
            de = (pe[i] - te[j] + np.pi) % (2 * np.pi) - np.pi
            matches.append(
                {
                    "prediction_id": pids[i],
                    "truth_id": tids[j],
                    "translation_error_mm": float(distance[i, j] * 1000),
                    "translation_delta_mm": (delta * 1000).tolist(),
                    "position_squared_error_m2": float(delta @ delta),
                    "rotation_error_deg": float(np.degrees(angle)),
                    "rotation_squared_error_rad2": angle * angle,
                    "wrapped_euler_squared_error_rad2": float(de @ de),
                }
            )
    tp = len(matches)
    errors = [m["translation_error_mm"] for m in matches]
    rotations = [m["rotation_error_deg"] for m in matches]
    return {
        "scope": "local diagnostic; official weighted scorer unavailable",
        "matching": "correct type, Euclidean distance<=50mm, cardinality first; IDs need not agree",
        "tolerance_mm": tolerance_m * 1000,
        "gt_holds": len(t),
        "predicted_holds": len(p),
        "tp": tp,
        "fp": len(p) - tp,
        "fn": len(t) - tp,
        "precision": tp / len(p) if len(p) else None,
        "recall": tp / len(t) if len(t) else None,
        "position_median_mm": float(np.median(errors)) if errors else None,
        "position_p90_mm": float(np.percentile(errors, 90)) if errors else None,
        "rotation_median_deg": float(np.median(rotations)) if rotations else None,
        "position_squared_error_m2": sum(m["position_squared_error_m2"] for m in matches),
        "rotation_squared_error_rad2": sum(m["rotation_squared_error_rad2"] for m in matches),
        "matches": matches,
    }


def load_strict_json(path):
    def no_duplicates(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError(f"Duplicate JSON key: {key}")
            obj[key] = value
        return obj

    def no_constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    return json.loads(
        Path(path).read_text(), object_pairs_hook=no_duplicates, parse_constant=no_constant
    )
