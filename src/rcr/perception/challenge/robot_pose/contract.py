"""Internal RobotState adapter; the official offline robot JSON is UNRELEASED."""

import json
from pathlib import Path

import numpy as np

from .frame_conventions import to_wxyz, validate_transform


def prediction(t, *, method, timestamp_s, inputs, diagnostics=None, joints=None, reason=None):
    if not np.isfinite(timestamp_s):
        raise ValueError("Timestamp must be finite and in the camera clock domain")
    out = dict(
        schema="robot-pose-v1",
        parent_frame="wall",
        child_frame="trunk",
        units="metres",
        timestamp_s=float(timestamp_s),
        method=method,
        valid=t is not None,
        confidence=None,
        confidence_note="Residuals are diagnostic; probability is not calibrated.",
        inputs=list(inputs),
        diagnostics=diagnostics or {},
        joints_rad=joints,
        reason=reason,
    )
    if t is None:
        out.update(T_wall_trunk=None, translation_m=None, rotation_wxyz=None)
    else:
        t = validate_transform(t)
        out.update(
            T_wall_trunk=t.tolist(),
            translation_m=t[:3, 3].tolist(),
            rotation_wxyz=to_wxyz(t[:3, :3]).tolist(),
        )
    json.dumps(out, allow_nan=False)
    return out


def write_prediction(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False))
