"""Explicit adapters to existing team types; no contact or calibration fabrication."""

import numpy as np

from rcr.types import LIMBS, Hold, Pose, Stance, WallModel

from .team_frame import ROTATION, team_pose_record


def robot_pose(prediction):
    record = team_pose_record(prediction)
    if record is None:
        return None
    return Pose(position=tuple(record["position"]), quat=tuple(record["quat"]))


def wall_model(cup_map, *, wall_id, level, verified_start_contacts):
    """Convert observed mounting points with caller-supplied verified start contacts.

    The source map contains socket centres and a separate mounting transform.
    Team Hold.position uses the mounting transform, never the socket centre.
    """
    if cup_map.get("frame") != "wall" or cup_map.get("units") != "metres":
        raise ValueError("Expected a metric research-wall cup map")
    if set(verified_start_contacts) != set(LIMBS):
        raise ValueError("Provide all four verified start contacts; no inferred loading")
    holds = []
    source_to_team = {}
    for item in cup_map["holds"]:
        position = ROTATION @ np.asarray(item["mounting_transform"]["translation_m"], float)
        identifier = item["id"]
        if identifier.startswith("cup_c"):
            col, row = identifier.removeprefix("cup_c").split("_r")
            identifier = f"c{int(col)}_r{int(row)}"
        source_to_team[item["id"]] = identifier
        holds.append(Hold(id=identifier, position=tuple(position), kind="cup"))
    observed = {h.id for h in holds}
    contacts = {
        limb: source_to_team.get(value, value) for limb, value in verified_start_contacts.items()
    }
    if any(value is not None and value not in observed for value in contacts.values()):
        raise ValueError("A verified start hold is not currently mapped; acquire another view")
    return WallModel(id=wall_id, level=level, holds=holds, start=Stance(contacts=contacts))
