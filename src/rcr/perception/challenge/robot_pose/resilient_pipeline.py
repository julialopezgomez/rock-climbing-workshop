"""Measured RGB preprocessing and recoverable optional CAD stages.

The frozen v1 pipeline remains unchanged. This contender preserves an autonomous
M3T estimate when an optional CAD refinement rejects its observation, and reports
that rejection rather than losing a usable earlier image estimate.
"""

import time

import cv2
import numpy as np

from .contract import prediction
from .frame_conventions import apply
from .model_adapter import RobotRenderer
from .pipeline import RobotPoseEstimator, prepare_packet
from .robust_pipeline import normalize_exposure


def prepare_rgb(image):
    corrected, exposure = normalize_exposure(image)
    gray = cv2.cvtColor(corrected, cv2.COLOR_RGB2GRAY).astype(np.float32)
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    residual = cv2.filter2D(gray, -1, kernel)[2:-2, 2:-2]
    # MAD of a second derivative on predominantly flat source backgrounds.
    # This is an input diagnostic, not a calibrated camera-noise model.
    sigma = float(np.median(np.abs(residual)) / 2.705)
    denoised = sigma > 2.0
    if denoised:
        corrected = cv2.bilateralFilter(corrected, 5, max(12.0, 4 * sigma), 1.2)
    return corrected, dict(
        exposure=exposure, noise_estimate_sigma=sigma, bilateral_applied=denoised
    )


class ResilientEstimator:
    def __init__(
        self, *, profile="max", calibration="aruco", gpu_confirmed=False, sequence=False, **options
    ):
        self.profile = profile
        self.calibration = calibration
        self.renderer = None
        self.refiner = None
        self.base = RobotPoseEstimator(
            profile="fast",
            calibration=calibration,
            gpu_confirmed=gpu_confirmed,
            sequence=sequence,
            correspondence_iterations=7,
            **options,
        )
        from .differentiable_fit import DifferentiableCAD

        self.refiner = DifferentiableCAD(
            model=self.base.frontend.model, gpu_confirmed=gpu_confirmed
        )

    def infer(self, packet, images, *, reset=False):
        started = time.perf_counter()
        images, preprocessing = zip(*(prepare_rgb(image) for image in images), strict=False)
        images = list(images)
        value = self.base.infer(packet, images, reset=reset)
        stages = {}
        if value["valid"]:
            clean, corrected, _ = prepare_packet(packet, images)
            cameras = [
                dict(c, T_camera_wall=info["T_camera_wall"], calibration=info)
                for c, info in zip(
                    clean["cameras"], value["diagnostics"]["calibration"], strict=False
                )
            ]
            pose = np.asarray(value["T_wall_trunk"])
            try:
                candidate, info = self.refiner.fit(corrected, cameras, clean["joints_rad"], pose)
                accepted = info["loss"] <= 0.25
                stages["differentiable_CAD"] = dict(info, accepted=accepted)
                if accepted:
                    pose = candidate
            except (ValueError, RuntimeError, cv2.error) as error:
                stages["differentiable_CAD"] = dict(
                    accepted=False, error=str(error), retained="autonomous M3T pose"
                )
            if self.profile == "max":
                from .exact_refine import refine_exact

                if self.renderer is None:
                    self.renderer = RobotRenderer(side=224)
                try:
                    pose, stages["exact_CAD"] = refine_exact(
                        corrected, cameras, clean["joints_rad"], self.renderer, pose
                    )
                except (ValueError, RuntimeError, cv2.error) as error:
                    stages["exact_CAD"] = dict(
                        accepted=False,
                        error=str(error),
                        retained="preceding autonomous image estimate",
                    )
            endpoints = {
                name: apply(pose, np.asarray([point]))[0].tolist()
                for name, point in self.base.frontend.model.endpoints(clean["joints_rad"]).items()
            }
            diagnostics = dict(
                value["diagnostics"], refinement_stages=stages, endpoints_wall_m=endpoints
            )
            value = prediction(
                pose,
                method="resilient_" + self.profile + "_" + self.calibration,
                timestamp_s=clean["timestamp_s"],
                joints=clean["joints_rad"],
                inputs=value["inputs"] + ["Optional robust CAD stages"],
                diagnostics=diagnostics,
            )
        else:
            value["method"] = "resilient_" + value["method"]
        value["diagnostics"].update(
            profile="resilient_" + self.profile,
            rgb_preprocessing=list(preprocessing),
            runtime_s=time.perf_counter() - started,
        )
        return value

    def close(self):
        self.base.close()
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
