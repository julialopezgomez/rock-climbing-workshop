"""Convert research wall X/right Y/up Z/out to the team's simulator wall basis."""

import numpy as np
from scipy.spatial.transform import Rotation

ROTATION = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def team_pose_record(prediction):
    if not prediction["valid"]:
        return None
    source = np.asarray(prediction["T_wall_trunk"], float)
    r = ROTATION @ source[:3, :3]
    t = ROTATION @ source[:3, 3]
    q = Rotation.from_matrix(r).as_quat()[[3, 0, 1, 2]]
    return {
        "position": t.tolist(),
        "quat": q.tolist(),
        "frame": "team_wall",
        "units": "metres",
        "axes": "x right, y into wall, z up",
        "timestamp_s": prediction["timestamp_s"],
    }
