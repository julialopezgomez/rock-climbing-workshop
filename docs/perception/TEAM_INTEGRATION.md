# Team integration

The operational API accepts explicit RGB packets, camera calibration, capture
clock metadata and measured joints where applicable. The legacy images-only
`rcr.perception.wall_model(images)` and `robot_pose(images)` remain stubs: their
signatures cannot carry these measurements. Existing shared dataclasses,
serialization formats and third-party submodules are unchanged.

Use `scripts/perception/run.py` for capture, replay and persistent inference.
For a caller owning capture/telemetry, use the estimator classes selected in
`challenge.cli.make_estimator`; call `infer(packet, rgb_images)` and `close()`.
The packet examples are in `data/perception/robot/scenes/`. Do not pass
`evaluation.json`, simulator truth or depth into inference.

## Frames and reference points

`INTERFACES.md` defines the team wall frame: metres/radians, origin at bottom
left, x right, y into the wall, z up, WXYZ quaternions. Raw robot/cup predictions
use x right, y up, z out at the same physical origin. The basis conversion is

```
R_team_research = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
```

`challenge.team_frame` applies this rotation to the complete transform.
`team_pose` in CLI output is already converted. Do not convert it twice.
The separate RCR stereo hold generator has a bottom-centre origin and CAD
backplate anchors. Its hold JSON is a submission to that image task; it is not
a team `WallModel` and must not be interpreted as socket centres or bolt midpoints.

## Convert a measured robot/cup result

```python
from rcr.perception.challenge.team_api import robot_pose, wall_model

# result is the raw prediction from the combined live estimator.
pose = robot_pose(result["robot_pose"])  # existing Pose, or None if invalid
wall = wall_model(
    result,
    wall_id=route_id,
    level=route_level,
    verified_start_contacts=measured_start_contacts,
)
```

The caller supplies the route identity/level and all four start-contact entries
(`left_hand`, `right_hand`, `left_foot`, `right_foot`). Each value is a mapped hold
ID or `None` for a verified free limb. The adapter uses each cup's **mounting
transform**, converts IDs to `c<col>_r<row>`, and rejects a start hold absent from
the map. Socket proximity does not establish seating or loading.

For live consumers, accept only records with `valid` and `fresh` true. The
launcher suppresses `team_pose` when output age exceeds the configured limit.
A valid robot pose can coexist with a failed map refresh; inspect
`prediction.map_refresh_error`, map timestamp and mapped holds before using the
wall. A controller bridge and contact verification remain caller responsibilities.

## Physical prerequisites

Measure camera calibration, wall-marker geometry and the rigid marker mount
before marker operation. Learned configured profiles additionally require all
19 named joints with acquisition times mapped into the camera host's monotonic
clock. See [RUNBOOK.md](RUNBOOK.md) for calibration and telemetry contracts.
Only RGB is captured. Physical USB capture and physical accuracy are unvalidated
because no cameras were attached during this preparation.
