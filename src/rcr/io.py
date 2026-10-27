"""Save/load for the files that cross module boundaries. Formats are documented in INTERFACES.md.

Every file is JSON with a "schema" field; loading a file with another schema raises ValueError.
Pi-safe: stdlib only, Python 3.9 or newer (the robot may decode a trajectory).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from rcr.types import ContactPlan, Hold, Stance, Trajectory, WallModel

WALL_SCHEMA = "wall/1"
CONTACT_PLAN_SCHEMA = "contact_plan/1"
TRAJECTORY_SCHEMA = "trajectory/2"


def _save(obj: object, schema: str, path: Path | str) -> None:
    data = {"schema": schema, **asdict(obj)}
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def _load(schema: str, path: Path | str) -> dict:
    return _checked(schema, json.loads(Path(path).read_text()), path)


def _checked(schema: str, data: dict, where: object) -> dict:
    data = dict(data)
    found = data.pop("schema", None)
    if found != schema:
        raise ValueError(f"{where}: expected schema {schema!r}, found {found!r}")
    return data


def _stance(d: dict) -> Stance:
    return Stance(contacts=dict(d["contacts"]))


def save_wall(wall: WallModel, path: Path | str) -> None:
    _save(wall, WALL_SCHEMA, path)


def load_wall(path: Path | str) -> WallModel:
    d = _load(WALL_SCHEMA, path)
    return WallModel(
        id=d["id"],
        level=d["level"],
        holds=[Hold(id=h["id"], position=tuple(h["position"]), kind=h["kind"]) for h in d["holds"]],
        start=_stance(d["start"]),
    )


def save_contact_plan(plan: ContactPlan, path: Path | str) -> None:
    _save(plan, CONTACT_PLAN_SCHEMA, path)


def load_contact_plan(path: Path | str) -> ContactPlan:
    d = _load(CONTACT_PLAN_SCHEMA, path)
    return ContactPlan(wall_id=d["wall_id"], stances=[_stance(s) for s in d["stances"]])


def save_trajectory(trajectory: Trajectory, path: Path | str) -> None:
    _save(trajectory, TRAJECTORY_SCHEMA, path)


def load_trajectory(path: Path | str) -> Trajectory:
    return trajectory_from_dict(json.loads(Path(path).read_text()), path)


def trajectory_to_dict(trajectory: Trajectory) -> dict:
    """The trajectory as its file's JSON object (schema included), e.g. to send over the link."""
    return {"schema": TRAJECTORY_SCHEMA, **asdict(trajectory)}


def trajectory_from_dict(data: dict, where: object = "trajectory") -> Trajectory:
    """The inverse of trajectory_to_dict; ValueError if the schema is another one."""
    d = _checked(TRAJECTORY_SCHEMA, data, where)
    return Trajectory(
        joint_names=d["joint_names"],
        times=d["times"],
        positions=d["positions"],
        planned_positions=d.get("planned_positions"),
        trunk_poses=d.get("trunk_poses"),
    )
