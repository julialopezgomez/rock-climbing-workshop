# Perception

The operational implementation is in `challenge/`. See the
[runbook](../../../docs/perception/RUNBOOK.md) and
[report](../../../docs/perception/perception_report.pdf).

It provides source-specific stereo hold JSON, configured robot pose, an image-only
trunk profile, RGB cup mapping, a measured-marker baseline, and two-RealSense
color capture/calibration. The selected models and replay examples are included;
`scripts/perception/setup.py` installs a private environment and pinned resources.

Run `scripts/perception/run.py --help` inside that environment. Inputs are explicit
RGB packets with calibration and timing; configured profiles require measured
joints. `challenge.team_api` converts measured outputs to existing team types.
Read the [integration notes](../../../docs/perception/TEAM_INTEGRATION.md) for
frames, mount/socket reference points, start-contact requirements and validity.

The legacy `wall_model(images)` and `robot_pose(images)` signatures remain stubs.
They do not carry camera calibration, telemetry or capture timing. Pipeline callers
can continue loading saved wall models while the explicit runtime supplies camera
results. A controller bridge must agree on these additional inputs.

No physical cameras were attached during preparation. Saved simulator regression,
synthetic RGB calibration and marker tests establish software behavior; they do
not establish physical-camera accuracy or a camera-driven climb.
