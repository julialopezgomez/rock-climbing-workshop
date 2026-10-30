#!/usr/bin/env python3
"""Score completed replay outputs; reference files are never inference inputs."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from rcr.perception.challenge.robot_pose.frame_conventions import apply
    from rcr.perception.challenge.robot_pose.model_adapter import RobotModel

    model = RobotModel()
    records = []
    for case in json.loads((args.dataset / "manifest.json").read_text())["cases"]:
        ident = case["id"]
        try:
            value = json.loads((args.predictions / ident / "perception.json").read_text())[
                "prediction"
            ]
            pose = value.get("robot_pose", value)
            if not pose["valid"]:
                raise ValueError(pose.get("reason", "Invalid prediction"))
            reference = json.loads(
                (args.dataset / "scenes" / ident / "evaluation.json").read_text()
            )
            estimated, truth = (
                np.asarray(pose["T_wall_trunk"]),
                np.asarray(reference["T_wall_trunk"]),
            )
            distance = float(np.linalg.norm(estimated[:3, 3] - truth[:3, 3]) * 1000)
            angle = float(
                np.rad2deg(Rotation.from_matrix(estimated[:3, :3] @ truth[:3, :3].T).magnitude())
            )
            joints = pose.get("joints_rad")
            endpoint = None
            if joints is not None:
                own = model.endpoints(joints)
                target = model.endpoints(reference["joint_positions_rad"])
                endpoint = max(
                    float(np.linalg.norm(apply(estimated, p) - apply(truth, target[name])) * 1000)
                    for name, p in own.items()
                )
            records.append(
                {
                    "case": ident,
                    "position_mm": distance,
                    "rotation_deg": angle,
                    "worst_endpoint_mm": endpoint,
                    "passed_root": distance < 5 and angle < 5,
                    "passed_endpoints": endpoint is not None and endpoint < 5,
                }
            )
        except Exception as error:
            records.append(
                {
                    "case": ident,
                    "passed_root": False,
                    "passed_endpoints": False,
                    "error": str(error),
                }
            )
    result = {
        "cases": len(records),
        "root_passes": sum(r["passed_root"] for r in records),
        "endpoint_passes": sum(r["passed_endpoints"] for r in records),
        "records": records,
        "scope": (
            "Saved-case regression under local 5mm/5degree root and all-endpoints-5mm criteria"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return (
        0
        if result["cases"] > 0
        and result["root_passes"] == result["endpoint_passes"] == result["cases"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
