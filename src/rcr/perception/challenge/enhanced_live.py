"""Optional partial-cup recovery and mapped-socket hypotheses for live RGB pose.

The initially frozen live and robot estimators remain unchanged. Confirmed
contacts are never inferred from the proximity prior.
"""

import time

from .dense_cup_map import DenseCupMapper
from .live_pipeline import LivePerception as FrozenLivePerception
from .robot_pose.mapped_socket_refine import MappedSocketRefiner
from .robot_pose.pipeline import RobotPoseEstimator


class EnhancedLivePerception(FrozenLivePerception):
    def __init__(
        self,
        *,
        profile="fast",
        calibration="aruco",
        socket_refinement=False,
        **options,
    ):
        self.sequence = options.get("sequence", False)
        self.map = None
        self.mapper = DenseCupMapper(
            calibration=calibration,
            gpu_confirmed=options.get("gpu_confirmed", False),
        )
        cls = RobotPoseEstimator
        if profile == "max":
            from .robot_pose.robust_edges import AdaptiveRobustEstimator

            cls = AdaptiveRobustEstimator
        self.robot = cls(profile=profile, calibration=calibration, **options)
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
