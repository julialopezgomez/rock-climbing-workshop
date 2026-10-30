"""Additional image-supported lattice hypotheses for partially occluded cups.

Preserves the established scene mapper, then tests small RGB components that
its complete-cup extent gate excluded. Every added socket must pass the same
two-view public-CAD occlusion, silhouette and uniqueness checks.
"""

import time

import cv2
import numpy as np

from .live_pipeline import CupMapper, calibrate, frame_transform, prepare_packet
from .robot_pose.frame_conventions import project
from .scene_cup_map import refine_candidates
from .src.calibration import intersect_wall_plane


class DenseCupMapper(CupMapper):
    def __init__(self, *, calibration="aruco", method="scene_grid_dense", gpu_confirmed=False):
        super().__init__(calibration=calibration, method="scene_grid", gpu_confirmed=gpu_confirmed)

    def infer(self, packet, images, robot_pose=None):
        started = time.perf_counter()
        value = super().infer(packet, images, robot_pose)
        value["method"] = "scene_grid_dense"
        if not value["estimated_robot_scene_used"]:
            value["partial_cup_recovery"] = dict(
                available=False, reason="Measured joints and a valid RGB root are required"
            )
            return value
        packet, images, _ = prepare_packet(packet, images, need_joints=False)
        cameras = [
            dict(c, camera_id=p["camera_id"])
            for c, p in zip(
                calibrate(images, packet["cameras"], self.calibration),
                packet["cameras"],
                strict=False,
            )
        ]
        known = np.asarray(
            [h["contact_transform"]["translation_m"] for h in value["holds"]]
        ).reshape(-1, 3)
        existing = [
            dict(
                point=point,
                observed_by=h["observed_by"],
                quality=[dict(cad_accepted=True)],
                pixels={},
            )
            for point, h in zip(known, value["holds"], strict=False)
        ]
        extras = []
        for eye, (image, camera) in enumerate(zip(images, cameras, strict=False)):
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            paint = ((hsv[..., 1] >= 140) & (hsv[..., 2] >= 35)).astype(np.uint8)
            roi = np.zeros(image.shape[:2], np.uint8)
            board = project(
                np.array([[0, 0, 0.026], [0.8, 0, 0.026], [0.8, 1.4, 0.026], [0, 1.4, 0.026]]),
                camera,
            )
            cv2.fillConvexPoly(roi, np.rint(board).astype(np.int32), 1)
            paint &= roi
            count, labels, stats, _ = cv2.connectedComponentsWithStats(paint)
            for label in range(1, count):
                x, y, w, h, area = map(int, stats[label])
                if area < 5 or w < 2 or h < 2:
                    continue
                center = [x + (w - 1) / 2, y + (h - 1) / 2]
                point = intersect_wall_plane([center], camera["K"], camera["T_camera_wall"], 0.026)[
                    0
                ]
                point[1] -= 0.001505
                if not (0 <= point[0] <= 0.8 and 0 <= point[1] <= 1.4):
                    continue
                if len(known) and np.linalg.norm(known[:, :2] - point[:2], axis=1).min() < 0.009:
                    continue
                extras.append(
                    dict(
                        point=point,
                        observed_by=[camera["camera_id"]],
                        quality=[
                            dict(
                                cad_accepted=False,
                                source="Partial RGB component without complete-cup extent gate",
                            )
                        ],
                        pixels={
                            eye: dict(origin=[x, y], mask=(labels[y : y + h, x : x + w] == label))
                        },
                    )
                )
        accepted, rejected = refine_candidates(
            existing + extras, images, cameras, robot_pose, self.scene_renderer
        )
        identifiers = {h["id"] for h in value["holds"]}
        added = 0
        for candidate in accepted[len(existing) :]:
            point = candidate["point"]
            col = int(round((point[0] - 0.0325) / 0.015))
            row = int(round((point[1] + 0.004) / 0.015))
            identifier = f"cup_c{col:02d}_r{row:02d}"
            if identifier in identifiers:
                continue
            identifiers.add(identifier)
            added += 1
            value["holds"].append(
                dict(
                    id=identifier,
                    type="cup",
                    model_identity="released_simulator_upright_cup",
                    mounting_transform=frame_transform(
                        identifier + "/mount", point + [0, 0.014, -0.026], packet["timestamp_s"]
                    ),
                    contact_transform=frame_transform(
                        identifier + "/socket", point, packet["timestamp_s"]
                    ),
                    mount_reference="Midpoint of the two mounting holes",
                    entry_direction_wall=[0, 1, 0],
                    observed_by=candidate["observed_by"],
                    association_residual_m=None,
                    confidence=None,
                    confidence_note="RGB/CAD residuals are not calibrated probabilities",
                    diagnostics=dict(
                        view_quality=candidate["quality"],
                        snapped_to_public_lattice=True,
                        grid_residual_m=None,
                        grid_origin_socket_m=[0.0325, -0.004, 0.026],
                        partial_cup_recovery=True,
                    ),
                )
            )
        value["holds"].sort(
            key=lambda h: (
                h["contact_transform"]["translation_m"][1],
                h["contact_transform"]["translation_m"][0],
            )
        )
        value["rejected_proposals"] += rejected
        value["partial_cup_recovery"] = dict(available=True, proposed=len(extras), added=added)
        value["runtime_s"] = time.perf_counter() - started
        return value
