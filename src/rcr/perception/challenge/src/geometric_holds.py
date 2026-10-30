"""Small deterministic RGB baseline for the released, colored cup holds.

Not a general climbing-hold classifier: this assumes the simulator's cup
geometry and known mounting lattice, with image-derived camera extrinsics.
Color is used only for proposals; neither the true route nor GT masks is
an input to detection/localization.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

try:
    from .calibration import (
        CalibrationError,
        calibrate_wall,
        calibrate_wall_boundary,
        intersect_wall_plane,
        project_wall,
    )
except ImportError:
    from calibration import (
        CalibrationError,
        calibrate_wall,
        calibrate_wall_boundary,
        intersect_wall_plane,
        project_wall,
    )


def detect_colored_holds(rgb, K=None, T_camera_wall=None, min_saturation=140, min_pixels=12):
    """High-saturation instance proposals, with optional board/metric gating.

    All hue ranges are accepted, so this does not encode orange as the hold
    label. Board hole dots and robot parts can still cause false positives.
    Connected objects/occlusions remain explicit limitations.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = ((hsv[..., 1] >= min_saturation) & (hsv[..., 2] >= 35)).astype(np.uint8)
    if T_camera_wall is not None:
        border = project_wall(
            [[0, 0, 0.026], [0.8, 0, 0.026], [0.8, 1.4, 0.026], [0, 1.4, 0.026]], K, T_camera_wall
        )
        roi = np.zeros(mask.shape, np.uint8)
        cv2.fillConvexPoly(roi, np.round(border).astype(np.int32), 1)
        mask &= roi
    # Do not close globally: neighbouring cups are only 1--2 pixels apart at
    # the default distance, and even a 2x2 kernel can merge two instances.
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    result = []
    for i in range(1, count):
        x, y, w, h, area = map(int, stats[i])
        if area < min_pixels or w < 3 or h < 3:
            continue
        center = np.array([x + (w - 1) / 2.0, y + (h - 1) / 2.0])
        socket = None
        if T_camera_wall is not None:
            box_wall = intersect_wall_plane(
                [[x, y], [x + w - 1, y + h - 1]], K, T_camera_wall, 0.026
            )
            extent = np.abs(box_wall[1, :2] - box_wall[0, :2])
            if not (0.018 < extent[0] < 0.070 and 0.012 < extent[1] < 0.060):
                continue
            socket = intersect_wall_plane([center], K, T_camera_wall, 0.026)[0]
            # Center of the measured cup silhouette bbox is 1.505 mm above
            # the socket, before perspective/occlusion and pixel effects.
            socket[1] -= 0.001505
        result.append(
            {
                "mask": labels == i,
                "bbox_xywh": [x, y, w, h],
                "center_px": center,
                "centroid_px": centroids[i],
                "socket_wall_m": socket,
                "area_px": area,
            }
        )
    return result


def snap_socket_to_grid(socket_wall_m, max_distance_m=0.006):
    """Return a conditional lattice hypothesis, plus residual, without a route.

    Uses the released grid's physical origin. Never applies to offline hold
    families. A snap is a hypothesis, not evidence of a seated contact.
    """
    point = np.asarray(socket_wall_m, float)
    col = int(np.rint((point[0] - 0.0325) / 0.015))
    row = int(np.rint((point[1] + 0.004) / 0.015))
    q = np.array([0.0325 + 0.015 * col, -0.004 + 0.015 * row, 0.026])
    residual = float(np.linalg.norm(q[:2] - point[:2]))
    if residual > max_distance_m or not (0 <= col <= 49 and 0 <= row <= 92):
        return None, residual
    return q, residual


def mask_matching(predicted_masks, truth, min_visible_pixels=12, iou_threshold=0.5):
    """One-to-one IoU evaluation; truth is used only in this scoring function."""
    ids, counts = np.unique(truth, return_counts=True)
    ids = ids[(ids > 0) & (counts >= min_visible_pixels)]
    pred = [np.asarray(m, bool) for m in predicted_masks]
    matrix = np.zeros((len(pred), len(ids)), float)
    truth_areas = np.array([np.count_nonzero(truth == tid) for tid in ids])
    for i, p in enumerate(pred):
        hit_ids, hit_counts = np.unique(truth[p], return_counts=True)
        intersection_by_id = dict(zip(hit_ids, hit_counts, strict=False))
        intersection = np.array([intersection_by_id.get(tid, 0) for tid in ids])
        matrix[i] = intersection / np.maximum(np.count_nonzero(p) + truth_areas - intersection, 1)
    matches = []
    if matrix.size:
        # Maximize the number of valid matches first, then total IoU. A raw
        # maximum-IoU assignment can sacrifice two valid matches for one
        # excellent match and undercount true positives after thresholding.
        valid = matrix >= iou_threshold
        reward = valid * (min(matrix.shape) + 1.0) + matrix * valid
        ii, jj = linear_sum_assignment(-reward)
        matches = [
            {"prediction": int(i), "hold_id": int(ids[j]), "iou": float(matrix[i, j])}
            for i, j in zip(ii, jj, strict=False)
            if matrix[i, j] >= iou_threshold
        ]
    n = len(matches)
    return {
        "visible_gt": len(ids),
        "predictions": len(pred),
        "tp": n,
        "fp": len(pred) - n,
        "fn": len(ids) - n,
        "precision": n / len(pred) if pred else 0.0,
        "recall": n / len(ids) if len(ids) else None,
        "matches": matches,
    }


def _stats(values):
    values = np.asarray(values, float)
    if not len(values):
        return {"count": 0}
    return {
        "count": len(values),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def evaluate_dataset(dataset, output, min_saturation=140):
    dataset, output = Path(dataset), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for metadata_path in sorted(dataset.glob("scenes/*/metadata.json")):
        meta = json.loads(metadata_path.read_text())
        scene = metadata_path.parent
        hold_by_id = {int(h["id"]): h for h in meta["holds"]}
        for camera_name, camera in meta["cameras"].items():
            rgb = cv2.cvtColor(cv2.imread(str(scene / camera_name / "rgb.png")), cv2.COLOR_BGR2RGB)
            truth = np.load(scene / camera_name / "instances.npy")
            K = np.asarray(camera["K"], float)
            gt_T = np.asarray(camera["T_camera_wall"], float)
            record = {"scene": scene.name, "camera": camera_name}
            start = time.perf_counter()
            try:
                calibration = calibrate_wall(
                    rgb, K, marker_corners_wall_m=meta["marker_corners_wall_m"]
                )
                T = calibration.T_camera_wall
                record["calibration"] = calibration.to_dict()
                record["calibration"]["position_error_mm"] = float(
                    np.linalg.norm(-T[:3, :3].T @ T[:3, 3] + gt_T[:3, :3].T @ gt_T[:3, 3]) * 1000
                )
                relative_R = T[:3, :3] @ gt_T[:3, :3].T
                record["calibration"]["rotation_error_deg"] = float(
                    np.degrees(np.arccos(np.clip((np.trace(relative_R) - 1) / 2, -1, 1)))
                )
            except CalibrationError as e:
                record["calibration"] = {"error": str(e)}
                # Markerless is measured as an uncalibrated detection-only
                # branch; never substitute GT/cached pose into the pipeline.
                T = None
            record["calibration_seconds"] = time.perf_counter() - start
            start = time.perf_counter()
            predictions = detect_colored_holds(rgb, K, T, min_saturation=min_saturation)
            record["detection_seconds"] = time.perf_counter() - start
            raw_predictions = detect_colored_holds(rgb, min_saturation=min_saturation)
            record["ungated_color_proposals"] = mask_matching(
                [p["mask"] for p in raw_predictions], truth
            )
            score = mask_matching([p["mask"] for p in predictions], truth)
            continuous_errors, snapped_errors = [], []
            for match in score["matches"]:
                prediction = predictions[match["prediction"]]
                if prediction["socket_wall_m"] is None:
                    continue
                actual_socket = np.asarray(hold_by_id[match["hold_id"]]["socket_wall_m"], float)
                match["socket_error_mm"] = float(
                    np.linalg.norm(prediction["socket_wall_m"] - actual_socket) * 1000
                )
                continuous_errors.append(match["socket_error_mm"])
                snapped, residual = snap_socket_to_grid(prediction["socket_wall_m"])
                match["grid_snap_residual_mm"] = residual * 1000
                if snapped is not None:
                    match["snapped_socket_error_mm"] = float(
                        np.linalg.norm(snapped - actual_socket) * 1000
                    )
                    snapped_errors.append(match["snapped_socket_error_mm"])
            record["detection"] = score
            record["socket_error_mm"] = _stats(continuous_errors)
            record["snapped_socket_error_mm"] = _stats(snapped_errors)
            # Independent markerless contender; never reuses marker estimates.
            try:
                boundary = calibrate_wall_boundary(rgb, K)
                boundary_predictions = detect_colored_holds(
                    rgb, K, boundary.T_camera_wall, min_saturation=min_saturation
                )
                boundary_score = mask_matching([p["mask"] for p in boundary_predictions], truth)
                boundary_errors = []
                for match in boundary_score["matches"]:
                    p = boundary_predictions[match["prediction"]]
                    actual = np.asarray(hold_by_id[match["hold_id"]]["socket_wall_m"], float)
                    error = float(np.linalg.norm(p["socket_wall_m"] - actual) * 1000)
                    match["socket_error_mm"] = error
                    boundary_errors.append(error)
                boundary_record = boundary.to_dict()
                boundary_record["position_error_mm"] = float(
                    np.linalg.norm(boundary.camera_position_wall_m + gt_T[:3, :3].T @ gt_T[:3, 3])
                    * 1000
                )
                relative_R = boundary.T_camera_wall[:3, :3] @ gt_T[:3, :3].T
                boundary_record["rotation_error_deg"] = float(
                    np.degrees(np.arccos(np.clip((np.trace(relative_R) - 1) / 2, -1, 1)))
                )
                record["markerless_boundary"] = {
                    "calibration": boundary_record,
                    "detection": boundary_score,
                    "socket_error_mm": _stats(boundary_errors),
                }
            except CalibrationError as e:
                record["markerless_boundary"] = {"error": str(e)}
            prediction_labels = np.zeros(truth.shape, np.int16)
            for index, prediction in enumerate(predictions, 1):
                prediction_labels[prediction["mask"]] = index
            prediction_path = output / f"{scene.name}_{camera_name}_predictions.npz"
            np.savez_compressed(
                prediction_path,
                instances=prediction_labels,
                bbox_xywh=np.asarray([p["bbox_xywh"] for p in predictions], int).reshape(-1, 4),
                socket_wall_m=np.asarray(
                    [
                        p["socket_wall_m"] if p["socket_wall_m"] is not None else [np.nan] * 3
                        for p in predictions
                    ]
                ).reshape(-1, 3),
            )
            record["predictions_file"] = str(prediction_path.resolve())
            image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            for prediction in predictions:
                x, y, w, h = prediction["bbox_xywh"]
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 230, 0), 1)
            cv2.imwrite(str(output / f"{scene.name}_{camera_name}.png"), image)
            records.append(record)
    (output / "results.json").write_text(
        json.dumps(
            {
                "dataset": str(dataset.resolve()),
                "method": (
                    "high saturation all hues + component metric gating + ca"
                    "librated known socket plane"
                ),
                "gt_usage": (
                    "scoring only; RGB image and K plus measured board/cup g"
                    "eometry are inference inputs"
                ),
                "iou_threshold": 0.5,
                "minimum_visible_gt_pixels": 12,
                "minimum_hsv_saturation": min_saturation,
                "records": records,
                "calibration_position_error_mm": _stats(
                    [
                        r["calibration"]["position_error_mm"]
                        for r in records
                        if "position_error_mm" in r["calibration"]
                    ]
                ),
                "socket_error_mm": _stats(
                    [
                        m["socket_error_mm"]
                        for r in records
                        for m in r["detection"]["matches"]
                        if "socket_error_mm" in m
                    ]
                ),
                "snapped_socket_error_mm": _stats(
                    [
                        m["snapped_socket_error_mm"]
                        for r in records
                        for m in r["detection"]["matches"]
                        if "snapped_socket_error_mm" in m
                    ]
                ),
            },
            indent=2,
        )
    )
    return records


def evaluate_saturation_ablation(dataset, output):
    """Sensitivity study on these development scenes, not a held-out score."""
    records = []
    for path in sorted(Path(dataset).glob("scenes/*/metadata.json")):
        meta = json.loads(path.read_text())
        for name, camera in meta["cameras"].items():
            rgb = cv2.cvtColor(cv2.imread(str(path.parent / name / "rgb.png")), cv2.COLOR_BGR2RGB)
            gt = np.load(path.parent / name / "instances.npy")
            try:
                T = calibrate_wall(
                    rgb, camera["K"], marker_corners_wall_m=meta["marker_corners_wall_m"]
                ).T_camera_wall
            except CalibrationError:
                T = None
            for threshold in [90, 100, 110, 125, 140, 155, 170]:
                pred = detect_colored_holds(rgb, camera["K"], T, min_saturation=threshold)
                score = mask_matching([p["mask"] for p in pred], gt)
                score.pop("matches")
                records.append(
                    {
                        "scene": path.parent.name,
                        "camera": name,
                        "minimum_saturation": threshold,
                        **score,
                    }
                )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "saturation_ablation.json").write_text(
        json.dumps(
            {
                "purpose": "threshold sensitivity on development scenes; not a held-out benchmark",
                "records": records,
            },
            indent=2,
        )
    )
    return records
