"""Bounded, trimmed CAD boundary fitting for partially occluded robot images.

This experimental fallback runs only after optional full-silhouette refinement
rejects. Both candidate generation and selection use legal RGB/public geometry.
"""

import time

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from .contract import prediction
from .differentiable_fit import observations
from .frame_conventions import apply, transform
from .model_adapter import RobotRenderer
from .pipeline import prepare_packet
from .resilient_pipeline import ResilientEstimator, prepare_rgb


def boundary(mask):
    mask = np.asarray(mask, np.uint8)
    return (mask - cv2.erode(mask, np.ones((3, 3), np.uint8))) > 0


def trimmed_mean(values, fraction=0.75, clip=3.0):
    if not len(values):
        return clip * clip
    values = np.minimum(values, clip) ** 2
    keep = max(1, int(np.ceil(len(values) * fraction)))
    return float(np.partition(values, keep - 1)[:keep].mean())


class RobustBoundaryFit:
    def __init__(self):
        self.renderer = RobotRenderer(side=224, scale=2)

    def fit(self, images, cameras, joints, initial, budget=500):
        started = time.perf_counter()
        base = np.asarray(initial)
        r = self.renderer
        r.set_joints(joints)
        values = observations(images, cameras, r.side)
        for value in values:
            edge = boundary(value["mask"] > 0.5) & value["valid"]
            value["edge"] = edge
            value["distance"] = cv2.distanceTransform(
                (~edge).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
            )
        scale = np.array([0.02] * 3 + [0.12] * 3)
        evaluations = 0

        def decode(delta):
            return transform(
                Rotation.from_rotvec(delta[3:] * scale[3:]).as_matrix() @ base[:3, :3],
                base[:3, 3] + delta[:3] * scale[:3],
            )

        def score(delta, details=False):
            nonlocal evaluations
            pose = decode(delta)
            total = 0.0
            terms = []
            for camera, value in zip(cameras, values, strict=False):
                mask = r.render(pose, camera, value["origin"]) > 0.5
                edge = boundary(mask) & value["valid"]
                distance = cv2.distanceTransform(
                    (~edge).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
                )
                term = trimmed_mean(value["distance"][edge]) + trimmed_mean(distance[value["edge"]])
                # Visible positive pixels must still be covered by the CAD.
                positive = (value["mask"] > 0.5) & value["valid"]
                uncovered = float((positive & ~mask).sum() / max(positive.sum(), 1))
                term += 2 * uncovered
                terms.append(term)
                total += term
            step = np.asarray(delta) * scale
            prior = 0.02 * float((step[:3] / 0.005) @ (step[:3] / 0.005)) + 0.02 * float(
                (step[3:] / 0.04) @ (step[3:] / 0.04)
            )
            evaluations += 1
            return (total + prior, dict(views=terms, prior=prior)) if details else total + prior

        initial_loss = score(np.zeros(6))
        best = [initial_loss, np.zeros(6)]
        # The second pass starts at the first pass's measured winner.
        for _ in range(2):
            fit = minimize(
                score,
                best[1],
                method="Powell",
                bounds=[(-1, 1)] * 6,
                options=dict(maxfev=budget, xtol=0.0015, ftol=1e-4),
            )
            if fit.fun < best[0]:
                best = [float(fit.fun), fit.x]
        loss, info = score(best[1], True)
        return decode(best[1]), dict(
            initial_loss=initial_loss,
            loss=loss,
            render_evaluations=evaluations,
            runtime_s=time.perf_counter() - started,
            trimmed_fraction=0.75,
            distance_clip_px=3.0,
            selected="Minimum common robust RGB-boundary objective, including the original pose",
            **info,
        )

    def close(self):
        self.renderer.close()


class AdaptiveRobustEstimator(ResilientEstimator):
    def __init__(self, **options):
        super().__init__(**options)
        self.edge_fit = None

    def infer(self, packet, images, **options):
        started = time.perf_counter()
        value = super().infer(packet, images, **options)
        rejected = any(
            s.get("accepted") is False
            for s in value["diagnostics"].get("refinement_stages", {}).values()
        )
        if value["valid"] and rejected and value.get("joints_rad") is not None:
            clean, corrected, _ = prepare_packet(packet, images)
            corrected = [prepare_rgb(image)[0] for image in corrected]
            cameras = [
                dict(c, T_camera_wall=i["T_camera_wall"], calibration=i)
                for c, i in zip(clean["cameras"], value["diagnostics"]["calibration"], strict=False)
            ]
            if self.edge_fit is None:
                self.edge_fit = RobustBoundaryFit()
            try:
                pose, info = self.edge_fit.fit(
                    corrected, cameras, clean["joints_rad"], value["T_wall_trunk"]
                )
                endpoints = {
                    name: apply(pose, np.asarray([point]))[0].tolist()
                    for name, point in self.base.frontend.model.endpoints(
                        clean["joints_rad"]
                    ).items()
                }
                diagnostics = dict(
                    value["diagnostics"], robust_boundary=info, endpoints_wall_m=endpoints
                )
                value = prediction(
                    pose,
                    method=value["method"] + "_robust_boundary",
                    timestamp_s=value["timestamp_s"],
                    inputs=value["inputs"] + ["Trimmed multi-view CAD edge fallback"],
                    joints=value["joints_rad"],
                    diagnostics=diagnostics,
                )
            except (ValueError, RuntimeError, cv2.error) as error:
                value["diagnostics"]["robust_boundary"] = dict(accepted=False, error=str(error))
        value["diagnostics"]["runtime_s"] = time.perf_counter() - started
        return value

    def close(self):
        super().close()
        if self.edge_fit is not None:
            self.edge_fit.close()
            self.edge_fit = None
