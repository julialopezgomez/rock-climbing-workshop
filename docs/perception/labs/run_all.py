#!/usr/bin/env python3
"""Run all examples and its exact-data checks using the current Python interpreter."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def execute(script, output, *options):
    command = [sys.executable, str(HERE / script), "--out", str(output), *options]
    subprocess.run(command, check=True, capture_output=True, text=True)
    result = json.loads((output / "results.json").read_text())
    print(script, result.get("lab"), "OK", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE.parent / "outputs")
    args = parser.parse_args()
    results = {}
    for script in [
        "stereo.py",
        "cad_fit.py",
        "keypoint_pose.py",
        "contact_registration.py",
        "real_stereo.py",
    ]:
        name = Path(script).stem
        results[name] = execute(script, args.out / name)
    zero = execute("stereo.py", args.out / "checks/stereo_exact", "--noise", "0")
    assert zero["valid_points"] == 500 and zero["p90_depth_error_mm"] < 1e-8
    zero = execute(
        "keypoint_pose.py",
        args.out / "checks/pnp_exact",
        "--noise",
        "0",
        "--outliers",
        "0",
    )
    assert zero["position_error_mm"] < 1e-5 and zero["rotation_error_deg"] < 1e-5
    zero = execute(
        "contact_registration.py",
        args.out / "checks/registration_exact",
        "--noise",
        "0",
    )
    assert zero["sample_consensus"]["position_error_mm"] < 1e-8
    assert zero["sample_consensus"]["rotation_error_deg"] < 1e-8
    import numpy as np
    from contact_registration import rigid_registration
    from stereo import triangulate

    line = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    try:
        rigid_registration(line, line)
    except ValueError:
        pass
    else:
        raise AssertionError("Collinear geometry must be rejected")
    _, valid = triangulate(
        np.array([[2.0, 0.0], [0.0, 0.0]]),
        np.array([[2.0, 0.0], [1.0, 0.0]]),
        np.eye(3),
        0.1,
    )
    assert not valid.any(), "Zero and negative disparity must be rejected"
    cad = results["cad_fit"]
    assert cad["selected"]["cost"] == min(c["cost"] for c in cad["candidates"])
    assert cad["translation_error_px"] < 3 and cad["yaw_error_deg"] < 5
    result = {
        "passed": True,
        "scope": "Simplified examples; independent of competition benchmarks",
        "examples": results,
        "checks": [
            "Exact stereo",
            "Exact PnP",
            "Outlier-resistant exact registration",
            "Reject collinear contacts",
            "Reject nonpositive disparity",
            "CAD candidate selection and fit",
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("All simplified examples and checks passed.")


if __name__ == "__main__":
    main()
