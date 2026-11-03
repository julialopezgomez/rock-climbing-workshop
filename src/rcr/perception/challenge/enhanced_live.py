"""Optional partial-cup recovery and mapped-socket hypotheses for live RGB pose.

The initially frozen live and robot estimators remain unchanged. Confirmed
contacts are never inferred from the proximity prior.
"""

import time

from ._paths import WEIGHTS
from .dense_cup_map import DenseCupMapper
from .live_pipeline import CupMapper
from .live_pipeline import LivePerception as FrozenLivePerception
from .robot_pose.mapped_socket_refine import MappedSocketRefiner
from .robot_pose.pipeline import DEFAULT_TRUNK_WEIGHTS, DEFAULT_WEIGHTS, RobotPoseEstimator
from .robot_pose.resilient_pipeline import ResilientEstimator
from .robot_pose.trunk_ensemble import TrunkEnsemble


class LiveTrunkEnsemble:
    def __init__(self, calibration="aruco", **options):
        self.calibration = calibration
        self.actor = TrunkEnsemble(
            whole_weights=options.get("weights", DEFAULT_WEIGHTS),
            trunk_weights=options.get("trunk_weights", DEFAULT_TRUNK_WEIGHTS),
            large_weights=WEIGHTS / "trunk_large/best.pt",
            rgb_correction=True,
            sequence=options.get("sequence", False),
            gpu_confirmed=options.get("gpu_confirmed", False),
        )

    def infer(self, packet, images, reset=False):
        return self.actor.infer(packet, images, self.calibration, reset=reset)

    def close(self):
        self.actor.close()


class EnhancedLivePerception(FrozenLivePerception):
    def __init__(
        self,
        *,
        profile="fast",
        calibration="aruco",
        cup_method="scene_grid_dense",
        socket_refinement=False,
        resilient=False,
        **options,
    ):
        self.sequence = options.get("sequence", False)
        self.map = None
        mapper = DenseCupMapper if cup_method == "scene_grid_dense" else CupMapper
        self.mapper = mapper(
            calibration=calibration,
            method=cup_method,
            gpu_confirmed=options.get("gpu_confirmed", False),
        )
        if profile == "trunk":
            self.robot = LiveTrunkEnsemble(calibration=calibration, **options)
        else:
            cls = RobotPoseEstimator
            if resilient:
                cls = ResilientEstimator
                if profile == "max":
                    from .robot_pose.robust_edges import AdaptiveRobustEstimator

                    cls = AdaptiveRobustEstimator
            self.robot = cls(
                profile=profile,
                calibration=calibration,
                **{k: v for k, v in options.items() if k != "trunk_weights"},
            )
        self.socket_refiner = MappedSocketRefiner() if socket_refinement else None

    def infer(self, packet, images, **options):
        started = time.perf_counter()
        value = super().infer(packet, images, **options)
        if self.socket_refiner is not None:
            value["visual_robot_pose"] = value["robot_pose"]
            pose, info = self.socket_refiner.infer(value["robot_pose"], value)
            value["robot_pose"] = pose
            value["mapped_socket_hypothesis"] = info
            value["assumptions"].append(
                "Socket proximity is a conditional geometric prior; it d"
                "oes not confirm contact or loading"
            )
        value["runtime_s"] = time.perf_counter() - started
        value["robot_pose"]["diagnostics"]["runtime_s"] = value["runtime_s"]
        value["robot_pose"]["diagnostics"]["timing_scope"] = (
            "Complete live RGB/map/robot inference including map refreshes"
        )
        return value
