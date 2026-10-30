"""Fit an upright cup's projected CAD silhouette using RGB masks only.

This is a constrained two-coordinate fit on the known socket plane. The renderer
uses the CAD mesh and IMAGE-ESTIMATED wall calibration, never a GT object pose.
Random restarts or more passes cannot add image information; each candidate is
ranked by a robust symmetric silhouette-boundary distance.
"""

import json
import os

from .._paths import CACHE, simulator_asset

os.environ.setdefault("MUJOCO_GL", "egl")
import cv2
import mujoco
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize

try:
    from .calibration import intersect_wall_plane, project_wall
except ImportError:
    from calibration import intersect_wall_plane, project_wall

ROOT = CACHE
R_WALL_OBJECT = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])


class SilhouetteRenderer:
    def __init__(self, mesh, side=64, scale=4):
        self.side, self.scale = side, scale
        spec = mujoco.MjSpec()
        spec.add_mesh(name="cup", file=str(mesh))
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_MESH, meshname="cup", contype=0, conaffinity=0
        )
        spec.worldbody.add_camera(
            name="view",
            resolution=[side * scale] * 2,
            sensor_size=[float(side * scale)] * 2,
            focal_length=[900, 900],
        )
        self.model = spec.compile()
        self.model.vis.quality.offsamples = 0
        self.model.vis.global_.offwidth = self.model.vis.global_.offheight = side * scale
        self.model.vis.map.znear = 0.00001
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=side * scale, width=side * scale)

    def render(self, point, K, T_camera_wall, origin):
        R = T_camera_wall[:3, :3] @ R_WALL_OBJECT
        t = T_camera_wall[:3, :3] @ point + T_camera_wall[:3, 3]
        self.model.cam_pos[0] = -R.T @ t
        axes = np.stack([R[0], -R[1], -R[2]], axis=1)
        mujoco.mju_mat2Quat(self.model.cam_quat[0], axes.ravel())
        # MuJoCo stores focal/principal lengths in the sensor's units.
        self.model.cam_intrinsic[0, :2] = [K[0, 0] * self.scale, K[1, 1] * self.scale]
        cx = (K[0, 2] - origin[0]) * self.scale
        cy = (K[1, 2] - origin[1]) * self.scale
        self.model.cam_intrinsic[0, 2:] = [
            self.side * self.scale / 2 - cx,
            self.side * self.scale / 2 - cy,
        ]
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera="view")
        self.renderer.enable_segmentation_rendering()
        seg = self.renderer.render()
        self.renderer.disable_segmentation_rendering()
        mask = (seg[..., 0] == 0).astype(np.float32)
        return cv2.resize(mask, (self.side, self.side), interpolation=cv2.INTER_AREA) > 0.5

    def close(self):
        self.renderer.close()


def boundary(mask):
    mask = np.asarray(mask, np.uint8)
    return (mask - cv2.erode(mask, np.ones((3, 3), np.uint8))).astype(bool)


def fit_shift(observed, template, starts=1, clip_px=3.0):
    a, b = boundary(observed), boundary(template)
    ay, ax = np.where(a)
    by, bx = np.where(b)
    if len(ax) < 4 or len(bx) < 4:
        raise ValueError("Insufficient silhouette boundary")
    da = cv2.distanceTransform((~a).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    db = cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    def loss(shift):
        dx, dy = shift
        d1 = map_coordinates(db, [ay - dy, ax - dx], order=1, mode="constant", cval=20)
        d2 = map_coordinates(da, [by + dy, bx + dx], order=1, mode="constant", cval=20)
        # Truncate to tolerate missing/color-rejected portions of the outline.
        return float(
            np.minimum(d1, clip_px).dot(np.minimum(d1, clip_px)) / len(d1)
            + np.minimum(d2, clip_px).dot(np.minimum(d2, clip_px)) / len(d2)
        )

    seeds = (
        [np.zeros(2)]
        if starts == 1
        else [np.array([x, y], float) for x in [-3, 0, 3] for y in [-3, 0, 3]]
    )
    candidates = []
    for seed in seeds:
        fit = minimize(
            loss,
            seed,
            method="Powell",
            bounds=[(-7, 7), (-7, 7)],
            options={"maxiter": 35, "xtol": 0.015, "ftol": 1e-5},
        )
        candidates.append({"shift_px": fit.x, "loss": loss(fit.x), "evaluations": fit.nfev})
    return min(candidates, key=lambda c: c["loss"]), candidates


def refine(point, mask, K, T, renderer, starts=1, passes=1):
    initial_uv = project_wall([point], K, T)[0]
    origin = np.floor(initial_uv - renderer.side / 2).astype(int)
    observed = np.zeros((renderer.side, renderer.side), bool)
    x0, y0 = origin
    x1, y1 = max(x0, 0), max(y0, 0)
    x2, y2 = min(x0 + renderer.side, mask.shape[1]), min(y0 + renderer.side, mask.shape[0])
    observed[y1 - y0 : y2 - y0, x1 - x0 : x2 - x0] = mask[y1:y2, x1:x2]
    point = np.asarray(point).copy()
    history = []
    for _ in range(passes):
        template = renderer.render(point, K, T, origin)
        fit, candidates = fit_shift(observed, template, starts)
        uv = project_wall([point], K, T)[0] + fit["shift_px"]
        point = intersect_wall_plane([uv], K, T, 0.026)[0]
        history.append(
            {
                "loss": fit["loss"],
                "shift_px": fit["shift_px"].tolist(),
                "observed_to_model_area_ratio": float(observed.sum() / max(template.sum(), 1)),
                "candidates": len(candidates),
                "evaluations": sum(c["evaluations"] for c in candidates),
            }
        )
    return point, history


def summarize(records):
    summary = {}
    for starts, passes in [(1, 1), (9, 1), (9, 2)]:
        r = [
            r
            for r in records
            if r["starts"] == starts and r["passes"] == passes and "fitted_error_mm" in r
        ]
        e = np.array([r["fitted_error_mm"] for r in r])
        base = np.array([r["baseline_error_mm"] for r in r])
        accepted = [
            v
            for v in r
            if v["history"][-1]["loss"] <= 4
            and v["history"][0]["observed_to_model_area_ratio"] >= 0.65
        ]
        accepted_errors = np.array([v["fitted_error_mm"] for v in accepted])
        summary[f"{starts}starts_{passes}passes"] = {
            "matched_cases": len(r),
            "median_mm": float(np.median(e)),
            "p90_mm": float(np.percentile(e, 90)),
            "max_mm": float(e.max()),
            "within_5mm": int((e < 5).sum()),
            "baseline_median_mm": float(np.median(base)),
            "baseline_p90_mm": float(np.percentile(base, 90)),
            "runtime_s_median": float(np.median([r["runtime_s"] for r in r])),
            "quality_gate": {
                "definition": (
                    "truncated symmetric boundary loss<=4pixel² and observed"
                    " area>=65%projected CAD area"
                ),
                "accepted": len(accepted),
                "rejected": len(r) - len(accepted),
                "median_mm": float(np.median(accepted_errors)) if len(accepted) else None,
                "p90_mm": float(np.percentile(accepted_errors, 90)) if len(accepted) else None,
                "max_mm": float(accepted_errors.max()) if len(accepted) else None,
                "within_5mm": int((accepted_errors < 5).sum()),
            },
        }
    return summary


def quality_gate(history):
    return {
        "accepted": bool(
            history[-1]["loss"] <= 4 and history[0]["observed_to_model_area_ratio"] >= 0.65
        ),
        "boundary_loss_px2": history[-1]["loss"],
        "observed_to_model_area_ratio": history[0]["observed_to_model_area_ratio"],
        "definition": (
            "boundary loss<=4pixel² and observed area>=65%projected CAD area; development threshold"
        ),
    }


def rescore_quality(args):
    """Add observable quality checks to cached fits without rerunning optimizers."""
    data = json.loads((args.out / "results.json").read_text())
    renderer = SilhouetteRenderer(simulator_asset("assets/hold/hold_visual.stl"))
    cache = {}
    geo = {
        (r["scene"], r["camera"]): r
        for r in json.loads((args.geometry / "results.json").read_text())["records"]
    }
    try:
        for record in data["records"]:
            key = (record["scene"], record["camera"], record["prediction"])
            if key not in cache:
                view = geo[key[:2]]
                meta = json.loads((args.dataset / "scenes" / key[0] / "metadata.json").read_text())
                K = np.array(meta["cameras"][key[1]]["K"])
                T = np.array(view["calibration"]["T_camera_wall"])
                labels = np.load(view["predictions_file"])["instances"]
                point = np.array(record["initial_socket_wall_m"])
                origin = np.floor(project_wall([point], K, T)[0] - renderer.side / 2).astype(int)
                template = renderer.render(point, K, T, origin)
                cache[key] = float((labels == key[2] + 1).sum() / max(template.sum(), 1))
            record["history"][0]["observed_to_model_area_ratio"] = cache[key]
            record["quality"] = quality_gate(record["history"])
        (args.out / "results.json").write_text(json.dumps(data, indent=2))
        summary = summarize(data["records"])
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
    finally:
        renderer.close()
