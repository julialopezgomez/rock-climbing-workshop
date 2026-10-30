"""The selected two-network, three-crop image-only root estimator."""

from ._paths import WEIGHTS
from .robot_pose.trunk_ensemble import TrunkEnsemble


class TrunkEstimator(TrunkEnsemble):
    def __init__(self, calibration="aruco", sequence=False):
        self.calibration = calibration
        super().__init__(
            whole_weights=WEIGHTS / "body/best.pt",
            trunk_weights=WEIGHTS / "trunk/best.pt",
            large_weights=WEIGHTS / "trunk_large/best.pt",
            rgb_correction=True,
            sequence=sequence,
            gpu_confirmed=True,
        )

    def infer(self, packet, images):
        return super().infer(packet, images, calibration_method=self.calibration)
