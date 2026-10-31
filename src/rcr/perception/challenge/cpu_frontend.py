"""CPU wall-pose proposals for deadline-limited competition inference.

Projected convex CAD hulls provide approximate poses without opening a GPU
context. Native recovery can refine these proposals to a complete generated
scene. Tilted volume poses remain unresolved if native recovery is unavailable.
"""

import functools
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .photo_frontend import appearance, canonical_rgb
from .src.hold_geometry import SceneGeometry
from .src.hold_scene import match_proposals, proposals
from .src.rcr_contract import versioned_prediction, write_prediction
from .src.rcr_methods import _calibration, normalized_mesh
from .src.rcr_native_geometry import wall_point


@functools.cache
def shapes() -> dict[str, np.ndarray]:
    """Cache the four CAD convex hulls in their scored mounting frames."""
    return {
        kind: np.array(normalized_mesh(kind).convex_hull.vertices)
        for kind in ["microban_hold", "sidepull_hold", "fingerpocket_hold", "cube_hold"]
    }


def project(vertices: np.ndarray, k: np.ndarray, t: np.ndarray) -> np.ndarray:
    xyz = vertices @ t[:3, :3].T + t[:3, 3]
    q = xyz @ k.T
    return q[:, :2] / q[:, 2:]


def model(
    vertices: np.ndarray,
    point: np.ndarray,
    k: np.ndarray,
    t: np.ndarray,
    origin: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize a projected hull and return its continuous image centroid."""
    uv = project(vertices + point, k, t)
    hull = cv2.convexHull(uv.astype(np.float32))
    moments = cv2.moments(hull)
    center = np.array([moments["m10"], moments["m01"]]) / max(moments["m00"], 1e-9)
    mask = np.zeros((128, 128), np.uint8)
    cv2.fillConvexPoly(mask, np.rint((hull.reshape(-1, 2) - origin) * 2 - 0.5).astype(np.int32), 1)
    return cv2.resize(mask.astype(float), (64, 64), interpolation=cv2.INTER_AREA), center


def crop(mask: np.ndarray, origin: np.ndarray) -> np.ndarray:
    x, y = origin
    out = np.zeros((64, 64))
    a, b = max(x, 0), max(y, 0)
    c, d = min(x + 64, mask.shape[1]), min(y + 64, mask.shape[0])
    if a < c and b < d:
        out[b - y : d - y, a - x : c - x] = mask[b:d, a:c]
    return out


def estimate(rgb: np.ndarray, calibration: dict[str, Any]) -> tuple[dict, dict]:
    """Fit paired paint components using three yaw hypotheses per CAD hull."""
    width = rgb.shape[1] // 2
    images = [rgb[:, :width], rgb[:, width:]]
    props = [proposals(im) for im in images]
    k, t = _calibration(calibration)
    geometry = SceneGeometry.from_calibration(calibration)
    pairs = match_proposals(*props, k, t, geometry)
    holds = {}
    details = []
    for i, j in pairs:
        pair = [props[0][i], props[1][j]]
        origins = [np.floor(p["center"] - 32).astype(int) for p in pair]
        observed = [crop(p["mask"], o) for p, o in zip(pair, origins, strict=False)]
        target = [p["center"] + 0.5 for p in pair]
        best = None
        for kind, vertices in shapes().items():
            for yaw in [0, -np.pi / 4, np.pi / 4]:
                c, s = np.cos(yaw), np.sin(yaw)
                r = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                v = vertices @ r.T
                center = v.mean(0)
                points = []
                for pix, ki, ti in zip(target, k, t, strict=False):
                    camera = -ti[:3, :3].T @ ti[:3, 3]
                    ray = ti[:3, :3].T @ np.linalg.solve(ki, np.r_[pix, 1.0])
                    points.append(camera + ray * ((center[2] - camera[2]) / ray[2]) - center)
                point = np.mean(points, axis=0)
                point[2] = 0
                for _ in range(3):
                    rendered = [
                        model(v, point, ki, ti, o) for ki, ti, o in zip(k, t, origins, strict=False)
                    ]
                    correction = np.mean(
                        [
                            wall_point(a, ki, ti) - wall_point(b[1], ki, ti)
                            for a, b, ki, ti in zip(target, rendered, k, t, strict=False)
                        ],
                        axis=0,
                    )
                    point += correction
                rendered = [
                    model(v, point, ki, ti, o)[0] for ki, ti, o in zip(k, t, origins, strict=False)
                ]
                loss = (
                    sum(
                        np.abs(a - b).sum() / max(np.maximum(a, b).sum(), 1)
                        for a, b in zip(observed, rendered, strict=False)
                    )
                    / 2
                )
                if best is None or loss < best[0]:
                    best = (loss, point.copy(), kind, yaw)
        loss, point, kind, yaw = best
        details.append(float(loss))
        if (
            loss > 0.5
            or abs(point[0]) > geometry.width / 2 + 0.02
            or not -0.02 < point[1] < geometry.height + 0.02
        ):
            continue
        holds[f"hold_{len(holds) + 1}"] = dict(
            type=kind,
            color="#" + "".join(f"{x:02X}" for x in pair[0]["color"]),
            position=dict(zip("xyz", map(float, point), strict=False)),
            orientation=dict(
                roll=0.0, pitch=0.0, yaw=float(yaw) * (0.5 if kind == "fingerpocket_hold" else 1)
            ),
        )
    return versioned_prediction(holds, calibration, 0), dict(proposals=len(pairs), losses=details)


def hold_prediction(
    stereo: str | Path, calibration: str | Path, out: str | Path
) -> tuple[dict, dict]:
    """Write a CPU estimate and the observed masks used by native recovery."""
    started = time.perf_counter()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    calibration = json.loads(Path(calibration).read_text())
    rgb = np.asarray(Image.open(stereo).convert("RGB"))
    if rgb.shape[1] % 2:
        raise ValueError("Stereo PNG must contain two equal-width panels")
    if (rgb.shape[1] // 2, rgb.shape[0]) != (
        calibration["image_width"],
        calibration["image_height"],
    ):
        raise ValueError("Stereo dimensions differ from declared calibration")
    style, fraction = appearance(rgb)
    normalized = canonical_rgb(rgb) if style == "real" else rgb
    prediction, debug = estimate(normalized, calibration)
    debug.update(
        method="CPU projected convex CAD hulls",
        volumes=[],
        appearance=dict(style=style, bright_paint_fraction=fraction),
        runtime_s=time.perf_counter() - started,
        inputs=["stereo RGB", "declared calibration", "four public CAD meshes"],
    )
    Image.fromarray(normalized).save(out / "native_stereo.png")
    native_calibration = dict(
        calibration,
        appearance=style,
        competition_rendering=True,
        native_volume_counts=[0, 3],
        native_ignore_volume_mask=True,
        native_partial_paint_masks=True,
    )
    (out / "native_calibration.json").write_text(json.dumps(native_calibration, indent=2) + "\n")
    (out / "image_holds.json").write_text(json.dumps(prediction["holds"], indent=2) + "\n")
    write_prediction(out, prediction["holds"], calibration, 0, debug, volumes=[])
    return prediction["holds"], debug
