"""Optional proximity-to-observed-socket pose hypothesis, not contact sensing.

Only RGB-mapped cups, measured joints, and an autonomous visual root are used.
No planned support, simulator socket map or contact flag is accepted. A geometric
match never confirms seating/loading; live confirmed_contacts remains empty.
"""

import numpy as np

from .contact_pose import Unobservable, estimate_contacts
from .contract import prediction
from .frame_conventions import apply, pose_errors
from .model_adapter import RobotModel


class MappedSocketRefiner:
    def __init__(
        self,
        association_gate_m=0.008,
        fit_gate_m=0.001,
        maximum_change_m=0.006,
        maximum_change_deg=2.5,
    ):
        self.model = RobotModel()
        self.association_gate_m = association_gate_m
        self.fit_gate_m = fit_gate_m
        self.maximum_change_m = maximum_change_m
        self.maximum_change_deg = maximum_change_deg

    def infer(self, visual, cup_map):
        if not visual["valid"] or visual.get("joints_rad") is None:
            return visual, dict(
                accepted=False, reason="A valid visual root and measured joints are required"
            )
        holds = [h for h in cup_map["holds"] if h.get("observed_by")]
        if len(holds) < 3:
            return visual, dict(accepted=False, reason="Fewer than three image-supported cups")
        sockets = np.asarray([h["contact_transform"]["translation_m"] for h in holds])
        points = self.model.endpoints(visual["joints_rad"])
        names = list(points)
        local = np.array(list(points.values()))
        transformed = apply(np.asarray(visual["T_wall_trunk"]), local)
        distance = np.linalg.norm(transformed[:, None] - sockets[None], axis=-1)
        matches = []
        used = set()
        for i in np.argsort(distance.min(axis=1)):
            order = np.argsort(distance[i])
            j = int(order[0])
            if j in used or distance[i, j] > self.association_gate_m:
                continue
            if len(order) > 1 and distance[i, order[1]] - distance[i, j] < 0.010:
                continue
            matches.append((int(i), j))
            used.add(j)
        info = dict(
            accepted=False,
            geometric_hypotheses=[
                dict(
                    limb=names[i],
                    hold_id=holds[j]["id"],
                    initial_distance_mm=float(distance[i, j] * 1000),
                )
                for i, j in matches
            ],
            contact_status="Unverified proximity hypotheses; seating/loading is not inferred",
        )
        if len(matches) < 3:
            return visual, dict(
                info, reason="Fewer than three unambiguous nearby socket hypotheses"
            )
        p = np.array([local[i] for i, j in matches])
        h = np.array([sockets[j] for i, j in matches])
        singular = np.linalg.svd(p - p.mean(0), compute_uv=False)
        if singular[1] < 0.015:
            return visual, dict(info, reason="Hypotheses are nearly collinear")
        try:
            pose, fit = estimate_contacts(p, h, gate_m=self.fit_gate_m)
        except (ValueError, Unobservable) as error:
            return visual, dict(info, reason=str(error))
        change = pose_errors(pose, visual["T_wall_trunk"])
        if (
            change["translation_mm"] > self.maximum_change_m * 1000
            or change["rotation_deg"] > self.maximum_change_deg
        ):
            return visual, dict(
                info,
                reason="Geometry would move too far from the autonomous image estimate",
                change=change,
            )
        residual = np.asarray(fit["residual_mm"])
        inliers = np.asarray(fit["inliers"])
        if residual[inliers].max() > self.fit_gate_m * 1000:
            return visual, dict(info, reason="Socket geometry is inconsistent", fit=fit)
        info.update(
            accepted=True,
            fit=fit,
            change=change,
            reason="Three or more congruent RGB-mapped socket proximity hypotheses",
        )
        endpoints = {
            name: apply(pose, np.asarray([point]))[0].tolist() for name, point in points.items()
        }
        diagnostics = dict(
            visual["diagnostics"], mapped_socket_prior=info, endpoints_wall_m=endpoints
        )
        value = prediction(
            pose,
            method=visual["method"] + "_mapped_socket_prior",
            timestamp_s=visual["timestamp_s"],
            inputs=visual["inputs"]
            + ["Observed RGB cup map and unverified socket-proximity geometric prior"],
            joints=visual["joints_rad"],
            diagnostics=diagnostics,
        )
        return value, info
