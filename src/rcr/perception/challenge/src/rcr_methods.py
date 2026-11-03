"""Stereo RGB baselines for jackybaltes/rcr-perception's actual scene format.

No ground-truth scene, hold colors/positions or instance masks enter inference.
Metric calibration is the publicly declared generator rig. Native random scenes
currently generate Microban holds only; type classification is not exercised.
"""

from __future__ import annotations

import os
import time

from .._paths import CACHE

os.environ.setdefault("MUJOCO_GL", "egl")
import cv2
import numpy as np
import trimesh
from scipy.ndimage import map_coordinates
from scipy.optimize import linear_sum_assignment, minimize
from scipy.spatial.transform import Rotation

ROOT = CACHE
METHODS = ("stereo_centroid", "stereo_surface", "cad_stereo_single", "cad_stereo_multistart")
_LAST = None
_RENDERERS = {}


def normalized_mesh(kind="microban_hold"):
    name = "microban_hold.stl" if kind == "microban_hold" else "cubehold_blend.stl"
    mesh = trimesh.load(CACHE / "assets/rcr" / name, force="mesh")
    v = np.asarray(mesh.vertices) @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0.0]]).T
    lo, hi = v.min(0), v.max(0)
    v[:, :2] -= (lo[:2] + hi[:2]) / 2
    v[:, 2] -= lo[2]
    return trimesh.Trimesh(v, mesh.faces, process=False)


def _calibration(calibration):
    cameras = calibration["cameras"]
    K = [np.array(cameras[c]["K"], float) for c in ["left", "right"]]
    T = [np.array(cameras[c]["T_camera_scene"], float) for c in ["left", "right"]]
    return K, T


def _project(point, K, T):
    p = T[:3, :3] @ point + T[:3, 3]
    return (K @ p)[:2] / p[2], float(p[2])


def _rotation(normal, yaw):
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    z = np.array([0.0, 0, 1])
    cross = np.cross(z, n)
    c = np.clip(z @ n, -1, 1)
    if np.linalg.norm(cross) < 1e-10:
        align = np.eye(3) if c > 0 else Rotation.from_rotvec([np.pi, 0, 0]).as_matrix()
    else:
        align = Rotation.from_rotvec(cross / np.linalg.norm(cross) * np.arccos(c)).as_matrix()
    return align @ Rotation.from_rotvec([0, 0, yaw]).as_matrix()


def _boundary(mask):
    a = np.asarray(mask, np.uint8)
    return (a - cv2.erode(a, np.ones((3, 3), np.uint8))).astype(bool)


def _proposal(rgb, mask, index):
    y, x = np.where(mask)
    if not len(x):
        return None
    w, h = int(x.max() - x.min() + 1), int(y.max() - y.min() + 1)
    if len(x) < 25 or w < 3 or h < 3 or w > 36 or h > 36 or len(x) > 1200:
        return None
    pixels = rgb[mask]
    color = np.median(pixels, axis=0)
    hsv = cv2.cvtColor(color.astype(np.uint8).reshape(1, 1, 3), cv2.COLOR_RGB2HSV)[0, 0]
    hue = float(hsv[0])
    group = 0 if hue < 22 or hue > 160 else 1 if hue < 98 else 2
    return {
        "mask": mask,
        "center": np.array([x.mean(), y.mean()]),
        "bbox": [int(x.min()), int(y.min()), w, h],
        "area": len(x),
        "color": color,
        "hue_group": group,
        "proposal_index": index,
    }


def detect_proposals(rgb, masks=None):
    """RGB-only color components, or externally predicted instance masks."""
    rgb = np.asarray(rgb, np.uint8)[..., :3]
    candidates = []
    if masks is None:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        saturated = (hsv[..., 1] >= 72) & (hsv[..., 2] >= 190)
        groups = [
            (hsv[..., 0] < 22) | (hsv[..., 0] > 160),
            (hsv[..., 0] >= 22) & (hsv[..., 0] < 98),
            (hsv[..., 0] >= 98) & (hsv[..., 0] <= 160),
        ]
        masks = []
        for group in groups:
            binary = (saturated & group).astype(np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            n, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
            for i in range(1, n):
                if 8 <= stats[i, cv2.CC_STAT_AREA] <= 2000:
                    masks.append(labels == i)
    else:
        # SAM may return remote disconnected islands. Keep its largest component
        # for localization, preserving raw model masks in the separate cache.
        clean = []
        for mask in masks:
            mask = np.asarray(mask, bool)
            n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
            if n > 1:
                clean.append(labels == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])))
        masks = clean
    for i, mask in enumerate(masks):
        p = _proposal(rgb, np.asarray(mask, bool), i)
        if p is not None:
            candidates.append(p)
    return candidates


def match_proposals(left, right, K, T):
    if not left or not right:
        return []
    relative = T[1] @ np.linalg.inv(T[0])
    t = relative[:3, 3]
    skew = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    F = np.linalg.inv(K[1]).T @ skew @ relative[:3, :3] @ np.linalg.inv(K[0])
    cost = np.full((len(left), len(right)), 1e6)
    info = {}
    for i, a in enumerate(left):
        u = np.r_[a["center"], 1.0]
        line = F @ u
        for j, b in enumerate(right):
            if a["hue_group"] != b["hue_group"]:
                continue
            v = np.r_[b["center"], 1.0]
            epipolar = abs(line @ v) / max(np.linalg.norm(line[:2]), 1e-9)
            if epipolar > 3.0:
                continue
            p = cv2.triangulatePoints(
                K[0] @ T[0][:3],
                K[1] @ T[1][:3],
                a["center"].reshape(2, 1),
                b["center"].reshape(2, 1),
            )[:, 0]
            p = p[:3] / p[3]
            z = (T[0][:3, :3] @ p + T[0][:3, 3])[2]
            if not np.isfinite(p).all() or not (0.3 < z < 2.5):
                continue
            if not (-0.25 < p[2] < 0.55):
                continue
            # Known CAD envelope supplies a metric gate, not a true hold map.
            physical_width = (a["bbox"][2] + b["bbox"][2]) / 2 * z / K[0][0, 0]
            if not (0.014 < physical_width < 0.09):
                continue
            size_penalty = abs(np.log(a["area"] / b["area"]))
            cost[i, j] = epipolar + 1.5 * size_penalty + 0.03 * abs(z - 1.1)
            info[(i, j)] = (p, epipolar)
    ii, jj = linear_sum_assignment(cost)
    return [
        (int(i), int(j), *info[(i, j)]) for i, j in zip(ii, jj, strict=False) if cost[i, j] < 1e5
    ]


def _stereo_cloud(left_rgb, right_rgb, K, T):
    size = (left_rgb.shape[1], left_rgb.shape[0])
    rel = T[1] @ np.linalg.inv(T[0])
    R1, R2, P1, P2, Q, *_ = cv2.stereoRectify(
        K[0],
        None,
        K[1],
        None,
        size,
        rel[:3, :3].copy(),
        rel[:3, 3].copy().reshape(3, 1),
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )
    maps = []
    gray = []
    for rgb, k, r, p in zip([left_rgb, right_rgb], K, [R1, R2], [P1, P2], strict=False):
        mx, my = cv2.initUndistortRectifyMap(k, None, r, p[:3, :3], size, cv2.CV_32FC1)
        maps.append((mx, my))
        gray.append(cv2.remap(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), mx, my, cv2.INTER_LINEAR))
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=96,
        blockSize=3,
        P1=72,
        P2=288,
        disp12MaxDiff=1,
        uniquenessRatio=8,
        speckleWindowSize=30,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disp = matcher.compute(*gray).astype(np.float32) / 16
    rect = cv2.reprojectImageTo3D(disp, Q)
    camera = rect @ R1
    world = (camera - T[0][:3, 3]) @ T[0][:3, :3]
    valid = (
        (disp > 0) & np.isfinite(world).all(-1) & (camera[..., 2] > 0.3) & (camera[..., 2] < 2.0)
    )
    return {"world": world, "valid": valid, "map": maps[0], "R1": R1, "P1": P1}


def _surface_normal(cloud, proposal):
    if cloud is None:
        return np.array([0.0, 0, 1]), None
    mx, my = cloud["map"]
    mask = cv2.remap(proposal["mask"].astype(np.uint8), mx, my, cv2.INTER_NEAREST) > 0
    ring = cv2.dilate(mask.astype(np.uint8), np.ones((35, 35), np.uint8)).astype(
        bool
    ) & ~cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    points = cloud["world"][ring & cloud["valid"]]
    if len(points) < 30:
        return np.array([0.0, 0, 1]), None
    points = points[:: max(1, len(points) // 500)]
    rng = np.random.default_rng(0)
    best = None
    for _ in range(25):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(b - a, c - a)
        length = np.linalg.norm(n)
        if length < 1e-8:
            continue
        n /= length
        if n[2] < 0:
            n = -n
        if n[2] < 0.25:
            continue
        distance = np.abs((points - a) @ n)
        inliers = distance < 0.004
        if best is None or inliers.sum() > best[0]:
            best = (inliers.sum(), inliers)
    if best is None or best[0] < 25:
        return np.array([0.0, 0, 1]), None
    p = points[best[1]]
    center = np.median(p, axis=0)
    _, _, v = np.linalg.svd(p - center, full_matrices=False)
    normal = v[-1]
    if normal[2] < 0:
        normal = -normal
    if normal[2] < 0.25:
        return np.array([0.0, 0, 1]), None
    return normal, center


class CadRenderer:
    """One exact metric mesh, one small offscreen context, reused across cases."""

    def __init__(self, kind):
        import mujoco

        self.mujoco = mujoco
        mesh = normalized_mesh(kind)
        directory = ROOT / "baseline_templates"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (kind + "_normalized.stl")
        if not path.exists():
            mesh.export(path)
        spec = mujoco.MjSpec()
        spec.add_mesh(name="hold", file=str(path))
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_MESH, meshname="hold", contype=0, conaffinity=0
        )
        spec.worldbody.add_camera(
            name="view", resolution=[128, 128], sensor_size=[128.0, 128.0], focal_length=[900, 900]
        )
        self.model = spec.compile()
        self.model.vis.quality.offsamples = 0
        self.model.vis.global_.offwidth = self.model.vis.global_.offheight = 128
        self.model.vis.map.znear = 0.00001
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=128, width=128)
        self.center = np.asarray(mesh.vertices).mean(0)

    def render(self, point, R, K, T, origin):
        r = T[:3, :3] @ R
        t = T[:3, :3] @ point + T[:3, 3]
        self.model.cam_pos[0] = -r.T @ t
        axes = np.stack([r[0], -r[1], -r[2]], axis=1)
        self.mujoco.mju_mat2Quat(self.model.cam_quat[0], axes.ravel())
        self.model.cam_intrinsic[0, :2] = [K[0, 0] * 2, K[1, 1] * 2]
        self.model.cam_intrinsic[0, 2:] = [
            64 - (K[0, 2] - origin[0]) * 2,
            64 - (K[1, 2] - origin[1]) * 2,
        ]
        self.mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera="view")
        self.renderer.enable_segmentation_rendering()
        seg = self.renderer.render()
        self.renderer.disable_segmentation_rendering()
        return (
            cv2.resize(
                (seg[..., 0] == 0).astype(np.float32), (64, 64), interpolation=cv2.INTER_AREA
            )
            > 0.5
        )


def _renderer(kind):
    if kind not in _RENDERERS:
        _RENDERERS[kind] = CadRenderer(kind)
    return _RENDERERS[kind]


def _template_loss(point, R, proposals, K, T, renderer):
    views = []
    for proposal, k, t in zip(proposals, K, T, strict=False):
        uv, z = _project(point, k, t)
        origin = np.floor(proposal["center"] - 32).astype(int)
        x, y = origin
        mask = np.zeros((64, 64), np.uint8)
        x1, y1 = max(x, 0), max(y, 0)
        x2, y2 = min(x + 64, proposal["mask"].shape[1]), min(y + 64, proposal["mask"].shape[0])
        mask[y1 - y : y2 - y, x1 - x : x2 - x] = proposal["mask"][y1:y2, x1:x2]
        template = renderer.render(point, R, k, t, origin)
        a, b = _boundary(mask), _boundary(template)
        ay, ax = np.where(a)
        by, bx = np.where(b)
        if len(ax) < 4 or len(bx) < 4:
            return None
        da = cv2.distanceTransform((~a).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        db = cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        views.append((uv, z, origin, ax, ay, bx, by, da, db, k, t))

    def loss(offset):
        candidate = point + np.asarray(offset) * 0.01
        score = 0.0
        for uv, z, origin, ax, ay, bx, by, da, db, k, t in views:
            new_uv, new_z = _project(candidate, k, t)
            scale = z / new_z
            # Perspective translation/scale of the exact CAD projection.
            old_anchor = uv - origin
            new_anchor = new_uv - origin
            px = new_anchor[0] + (bx - old_anchor[0]) * scale
            py = new_anchor[1] + (by - old_anchor[1]) * scale
            sx = old_anchor[0] + (ax - new_anchor[0]) / scale
            sy = old_anchor[1] + (ay - new_anchor[1]) / scale
            d1 = map_coordinates(db, [sy, sx], order=1, mode="constant", cval=20)
            d2 = map_coordinates(da, [py, px], order=1, mode="constant", cval=20)
            score += float(
                np.minimum(d1, 4).dot(np.minimum(d1, 4)) / len(d1)
                + np.minimum(d2, 4).dot(np.minimum(d2, 4)) / len(d2)
            )
        return score

    return loss


def _infer_pose(a, b, centroid, K, T, method, cloud):
    normal, plane = (
        _surface_normal(cloud, a) if method != "stereo_centroid" else (np.array([0.0, 0, 1]), None)
    )
    # Mesh average approximates the visible surface offset; exact silhouette
    # refinement below estimates the exported backplate anchor directly.
    initial = centroid - np.array([0.0, 0.0, 0.018])
    if method == "stereo_surface" and plane is not None:
        initial = initial - normal * ((initial - plane) @ normal)
    if method in {"stereo_centroid", "stereo_surface"}:
        R = _rotation(normal, 0.0)
        return initial, R, {"method": method, "surface_normal": normal.tolist(), "cad_fit": None}
    renderer = _renderer("microban_hold")
    yaw_candidates = [0.0] if method == "cad_stereo_single" else [0.0, -np.pi / 4, np.pi / 4]
    normals = [normal]
    if method == "cad_stereo_multistart" and np.linalg.norm(normal - [0, 0, 1]) > 0.08:
        normals.append(np.array([0.0, 0, 1]))
    best = None
    candidates = []
    for n in normals:
        for yaw in yaw_candidates:
            R = _rotation(n, yaw)
            loss = _template_loss(initial, R, [a, b], K, T, renderer)
            if loss is None:
                continue
            fit = minimize(
                loss,
                np.zeros(3),
                method="Powell",
                bounds=[(-4, 4), (-4, 4), (-12, 12)],
                options={"maxiter": 12, "xtol": 0.015, "ftol": 1e-4},
            )
            candidate = {
                "position": initial + fit.x * 0.01,
                "R": R,
                "loss": float(loss(fit.x)),
                "evaluations": int(fit.nfev),
                "yaw_candidate": float(yaw),
                "normal": n.tolist(),
            }
            candidates.append(candidate)
            if best is None or candidate["loss"] < best["loss"]:
                best = candidate
    if best is None:
        return (
            initial,
            _rotation(normal, 0),
            {"method": method, "cad_fit": {"status": "insufficient boundary"}},
        )
    return (
        best["position"],
        best["R"],
        {
            "method": method,
            "surface_normal": normal.tolist(),
            "cad_fit": {k: v for k, v in best.items() if k not in {"position", "R"}},
            "candidate_count": len(candidates),
        },
    )


def _prepare(left_rgb, right_rgb, calibration, left_masks, right_masks):
    global _LAST
    key = (id(left_rgb), id(right_rgb), id(left_masks), id(right_masks))
    if _LAST is not None and _LAST[0] == key:
        return _LAST[1]
    K, T = _calibration(calibration)
    left = detect_proposals(left_rgb, left_masks)
    right = detect_proposals(right_rgb, right_masks)
    pairs = match_proposals(left, right, K, T)
    prepared = {
        "K": K,
        "T": T,
        "left": left,
        "right": right,
        "pairs": pairs,
        "cloud": None,
        "left_rgb": left_rgb,
        "right_rgb": right_rgb,
    }
    _LAST = (key, prepared)
    return prepared


def infer_scene(
    left_rgb, right_rgb, calibration, method="stereo_centroid", left_masks=None, right_masks=None
):
    """Return ``{'holds': legacy_mapping, 'debug': ...}`` from images/calibration.

    Supplied masks must be predicted bool[N,H,W], e.g. cached SAM3 masks. The
    function never opens a scene label file. Camera frame is standard OpenCV.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method {method}")
    start = time.perf_counter()
    p = _prepare(left_rgb, right_rgb, calibration, left_masks, right_masks)
    if method != "stereo_centroid" and p["cloud"] is None:
        p["cloud"] = _stereo_cloud(left_rgb, right_rgb, p["K"], p["T"])
    holds = {}
    details = []
    for i, j, centroid, epipolar in p["pairs"]:
        a, b = p["left"][i], p["right"][j]
        position, R, detail = _infer_pose(a, b, centroid, p["K"], p["T"], method, p["cloud"])
        angles = Rotation.from_matrix(R).as_euler("XYZ")
        color = np.rint((a["color"] + b["color"]) / 2).clip(0, 255).astype(np.uint8)
        name = f"hold_{len(holds) + 1}"
        holds[name] = {
            "type": "microban_hold",
            "color": "#" + "".join(f"{int(c):02X}" for c in color),
            "position": dict(zip(["x", "y", "z"], map(float, position), strict=False)),
            "orientation": dict(zip(["roll", "pitch", "yaw"], map(float, angles), strict=False)),
        }
        details.append(
            {
                "id": name,
                "left_proposal": i,
                "right_proposal": j,
                "left_bbox": a["bbox"],
                "right_bbox": b["bbox"],
                "epipolar_error_px": float(epipolar),
                **detail,
            }
        )
    return {
        "holds": holds,
        "debug": {
            "method": method,
            "runtime_s": time.perf_counter() - start,
            "left_proposals": len(p["left"]),
            "right_proposals": len(p["right"]),
            "matched_pairs": len(p["pairs"]),
            "objects": details,
            "input_provenance": (
                "RGB stereo + declared calibrated rig + public normalized CAD; no scene labels"
            ),
            "type_scope": (
                "Native upstream random generator emits Microban only; c"
                "lassification is unexercised"
            ),
            "pose_scope": (
                "centroid baseline yaw0; CAD methods search discrete sou"
                "rce-supported yaw and image-stereo-estimated surface no"
                "rmals"
            ),
        },
    }
