"""Legal RGB cup mapping plus Microban pose in a common metric wall frame.

The lattice is a public simulator geometry prior. It is not a supplied hold map:
only image-supported cup proposals are emitted. Physical mounting origins must
be measured before enabling this prior on a real wall.
"""

import copy
import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from ._paths import simulator_asset
from .robot_pose.configured_fit import calibrate
from .robot_pose.contract import prediction
from .robot_pose.pipeline import (
    DEFAULT_TRUNK_WEIGHTS,
    DEFAULT_WEIGHTS,
    RobotPoseEstimator,
    TrunkPoseEstimator,
    prepare_packet,
)
from .src.cad_silhouette import SilhouetteRenderer, quality_gate, refine
from .src.geometric_holds import detect_colored_holds, snap_socket_to_grid
from .src.live_estimator import frame_transform


class PixelCentreCupRenderer(SilhouetteRenderer):
    def render(self, point, k, t_camera_wall, origin):
        raster_k = np.asarray(k, float).copy()
        raster_k[:2, 2] += 0.5
        return super().render(point, raster_k, t_camera_wall, origin)


class CupMapper:
    def __init__(self, *, calibration="aruco", method="grid_verified", gpu_confirmed=False):
        if method not in {"geometry", "cad", "grid", "grid_verified", "scene_grid"}:
            raise ValueError("Unknown cup mapping method")
        self.calibration = calibration
        self.method = method
        self.renderer = None
        self.scene_renderer = None
        if method != "geometry":
            if not gpu_confirmed:
                raise ValueError("CAD cup mapping uses the GPU renderer")
            mesh = simulator_asset("assets/hold/hold_visual.stl")
            self.renderer = PixelCentreCupRenderer(mesh)

    def infer(self, packet, images, robot_pose=None):
        started = time.perf_counter()
        packet, images, timing = prepare_packet(packet, images, need_joints=False)
        cameras = [
            dict(c, camera_id=p["camera_id"])
            for c, p in zip(
                calibrate(images, packet["cameras"], self.calibration),
                packet["cameras"],
                strict=False,
            )
        ]
        views = []
        for image, camera in zip(images, cameras, strict=False):
            entries = []
            for raw in detect_colored_holds(image, camera["K"], camera["T_camera_wall"]):
                point = np.asarray(raw["socket_wall_m"])
                info = dict(cad_accepted=False)
                if self.renderer is not None:
                    try:
                        fitted, history = refine(
                            point,
                            raw["mask"],
                            np.asarray(camera["K"]),
                            np.asarray(camera["T_camera_wall"]),
                            self.renderer,
                        )
                        info = quality_gate(history)
                        info["cad_accepted"] = info.pop("accepted")
                        if info["cad_accepted"]:
                            point = (point + fitted) * 0.5
                    except ValueError as error:
                        info["reason"] = str(error)
                x, y, w, h = raw["bbox_xywh"]
                entries.append(
                    dict(
                        point=point,
                        quality=info,
                        pixels=dict(origin=[x, y], mask=raw["mask"][y : y + h, x : x + w].copy()),
                    )
                )
            views.append(entries)
        first, second = views
        matches = []
        if first and second:
            distance = np.linalg.norm(
                np.array([v["point"] for v in first])[:, None]
                - np.array([v["point"] for v in second])[None],
                axis=-1,
            )
            gate = 0.020
            reward = np.where(
                distance <= gate, 1 + (1 - distance / gate) / (min(distance.shape) + 1), 0
            )
            ii, jj = linear_sum_assignment(-reward)
            matches = [
                (i, j, float(distance[i, j]))
                for i, j in zip(ii, jj, strict=False)
                if distance[i, j] <= gate
            ]
        used_a = {i for i, j, d in matches}
        used_b = {j for i, j, d in matches}
        candidates = []
        ids = [c["camera_id"] for c in packet["cameras"]]
        for i, j, residual in matches:
            candidates.append(
                dict(
                    point=(first[i]["point"] + second[j]["point"]) / 2,
                    observed_by=ids,
                    association_residual_m=residual,
                    quality=[first[i]["quality"], second[j]["quality"]],
                    pixels={0: first[i]["pixels"], 1: second[j]["pixels"]},
                )
            )
        for eye, values, used in [(0, first, used_a), (1, second, used_b)]:
            for i, value in enumerate(values):
                if i not in used:
                    candidates.append(
                        dict(
                            point=value["point"],
                            observed_by=[ids[eye]],
                            association_residual_m=None,
                            quality=[value["quality"]],
                            pixels={eye: value["pixels"]},
                        )
                    )
        scene_rejected = []
        scene_used = False
        if (
            self.method == "scene_grid"
            and robot_pose is not None
            and robot_pose["valid"]
            and robot_pose.get("joints_rad") is not None
        ):
            from .scene_cup_map import CupSceneRenderer, refine_candidates

            if self.scene_renderer is None:
                self.scene_renderer = CupSceneRenderer()
            candidates, scene_rejected = refine_candidates(
                candidates, images, cameras, robot_pose, self.scene_renderer
            )
            scene_used = True
        candidates.sort(key=lambda c: (c["point"][1], c["point"][0]))
        stamp = packet["timestamp_s"]
        holds = []
        seen = set()
        rejected = scene_rejected
        for index, candidate in enumerate(candidates):
            if (
                (self.method == "grid_verified" or (self.method == "scene_grid" and not scene_used))
                and len(candidate["observed_by"]) < 2
                and not any(q["cad_accepted"] for q in candidate["quality"])
            ):
                rejected.append(
                    dict(
                        point_m=candidate["point"].tolist(),
                        observed_by=candidate["observed_by"],
                        quality=candidate["quality"],
                        reason="Single-view proposal lacks CAD silhouette support",
                    )
                )
                continue
            point = candidate["point"]
            grid = None
            residual = None
            if self.method in {"grid", "grid_verified", "scene_grid"}:
                grid, residual = snap_socket_to_grid(point, max_distance_m=0.006)
            if grid is not None:
                col = int(round((grid[0] - 0.0325) / 0.015))
                row = int(round((grid[1] + 0.004) / 0.015))
                identifier = f"cup_c{col:02d}_r{row:02d}"
                point = grid
            else:
                identifier = f"frame_cup_{index:03d}"
            if identifier in seen:
                continue
            seen.add(identifier)
            holds.append(
                dict(
                    id=identifier,
                    type="cup",
                    model_identity="released_simulator_upright_cup",
                    mounting_transform=frame_transform(
                        identifier + "/mount", point + [0, 0.014, -0.026], stamp
                    ),
                    contact_transform=frame_transform(identifier + "/socket", point, stamp),
                    mount_reference="Midpoint of the two mounting holes",
                    entry_direction_wall=[0, 1, 0],
                    observed_by=candidate["observed_by"],
                    association_residual_m=candidate["association_residual_m"],
                    confidence=None,
                    confidence_note=(
                        "Image residuals and prior checks are not calibrated probabilities"
                    ),
                    diagnostics=dict(
                        view_quality=candidate["quality"],
                        snapped_to_public_lattice=grid is not None,
                        grid_residual_m=residual,
                        grid_origin_socket_m=[0.0325, -0.004, 0.026] if grid is not None else None,
                    ),
                )
            )
        return dict(
            schema="live-cup-map-v2",
            frame="wall",
            units="metres",
            timestamp_s=stamp,
            method=self.method,
            calibration_method=self.calibration,
            calibrations=[c["calibration"] for c in cameras],
            timing=timing,
            holds=holds,
            runtime_s=time.perf_counter() - started,
            rejected_proposals=rejected,
            estimated_robot_scene_used=scene_used,
            inputs=["RGB pair", "intrinsics", "known rectangular board and cup CAD"]
            + (
                ["public mounting lattice"]
                if self.method in {"grid", "grid_verified", "scene_grid"}
                else []
            )
            + (["RGB-estimated configured robot occlusion geometry"] if scene_used else []),
            assumptions=[
                "Known upright .8x1.4m simulator cup wall",
                "Grid prior is conditional on the released mounting geometry",
                (
                    "No native depth, true socket map, robot pose or evaluat"
                    "or camera extrinsics are read"
                ),
            ],
        )

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        if self.scene_renderer is not None:
            self.scene_renderer.close()
            self.scene_renderer = None


class LivePerception:
    def __init__(
        self,
        *,
        profile="accuracy",
        calibration="aruco",
        cup_method="grid_verified",
        sequence=False,
        fixed_cameras=False,
        weights=DEFAULT_WEIGHTS,
        trunk_weights=DEFAULT_TRUNK_WEIGHTS,
        gpu_confirmed=False,
    ):
        self.sequence = sequence
        self.map = None
        self.mapper = CupMapper(
            calibration=calibration, method=cup_method, gpu_confirmed=gpu_confirmed
        )
        cls = TrunkPoseEstimator if profile == "trunk" else RobotPoseEstimator
        options = dict(
            profile=profile,
            calibration=calibration,
            sequence=sequence,
            fixed_cameras=fixed_cameras,
            weights=weights,
            gpu_confirmed=gpu_confirmed,
        )
        if profile == "trunk":
            options["trunk_weights"] = trunk_weights
        self.robot = cls(**options)

    def infer(self, packet, images, *, refresh_map=False, reset=False):
        started = time.perf_counter()
        map_updated = self.map is None or refresh_map or reset or not self.sequence
        try:
            robot = self.robot.infer(packet, images, reset=reset)
        except (ValueError, RuntimeError, cv2.error) as error:
            robot = prediction(
                None,
                method="live_robot_pose",
                timestamp_s=packet["timestamp_s"],
                inputs=["RGB pair and normal telemetry"],
                reason=str(error),
            )
        map_error = None
        if map_updated:
            try:
                latest = self.mapper.infer(packet, images, robot_pose=robot)
            except (ValueError, RuntimeError, cv2.error) as error:
                map_error = str(error)
                map_updated = False
                latest = (
                    self.map
                    if self.map is not None and self.sequence and not reset
                    else dict(
                        timestamp_s=packet["timestamp_s"],
                        method=self.mapper.method,
                        calibration_method=self.mapper.calibration,
                        holds=[],
                        runtime_s=0.0,
                    )
                )
            if self.sequence and self.map is not None and not reset:
                # Union only stable, image-supported grid IDs. Frame-local IDs
                # cannot be associated across frames without a tracker.
                old = {h["id"]: h for h in self.map["holds"] if h["id"].startswith("cup_c")}
                old.update({h["id"]: h for h in latest["holds"]})
                latest["holds"] = list(old.values())
            self.map = latest
        return dict(
            schema="live-perception-v2",
            frame="wall",
            units="metres",
            quaternion_order="WXYZ",
            timestamp_s=packet["timestamp_s"],
            holds=copy.deepcopy(self.map["holds"]),
            robot_pose=robot,
            map_timestamp_s=self.map["timestamp_s"],
            map_updated=map_updated,
            map_method=self.map["method"],
            map_calibration=self.map["calibration_method"],
            rejected_map_proposals=self.map.get("rejected_proposals", []) if map_updated else [],
            map_refresh_error=map_error,
            map_runtime_s=self.map["runtime_s"] if map_updated else 0.0,
            runtime_s=time.perf_counter() - started,
            confirmed_contacts={},
            assumptions=[
                "Wall holds remain static during a sequence",
                "Contact loading is not inferred from planned or true contact flags",
            ],
            integration_status=(
                "Versioned perception measurements; no closed-loop climb"
                " or physical-camera validation is claimed"
            ),
        )

    def close(self):
        self.robot.close()
        self.mapper.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
