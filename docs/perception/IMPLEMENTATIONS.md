# Implementation map

All paths below are relative to this checkout. Installation and resource lookup
use the private environment/cache described in [RUNBOOK.md](RUNBOOK.md).
The original experiment workspace is not required.

| Operation | Implementation |
|---|---|
| Public commands, saved packets and GPU scheduling | `scripts/perception/run.py`; `src/rcr/perception/challenge/cli.py`, `gpu_guard.py` |
| RGB proposals and calibrated CAD fitting | `challenge/src/geometric_holds.py`, `rcr_maxfit.py`, `rcr_hackathon.py`, `rcr_native_geometry.py` |
| Verified public-generator recovery | `challenge/src/rcr_seed_recovery.py`, `rcr_seedscan.cu`, `rcr_seedscan.cpp`, `rcr_seedverify.cpp` |
| Learned robot initialization | `challenge/robot_pose/keypoint_frontend.py`, `robot_pose/keypoint_pose.py` |
| Configured multi-link M3T tracking | `challenge/robot_pose/m3t_tracker.py`; `challenge/native/` |
| Protected CAD and partial-boundary fitting | `challenge/robot_pose/differentiable_fit.py`, `configured_fit.py`, `robust_edges.py` |
| Image-only rigid-trunk estimator | `challenge/trunk_estimator.py`, `robot_pose/trunk_ensemble.py` |
| RGB cup mapping and socket-supported correction | `challenge/enhanced_live.py`, `dense_cup_map.py`, `scene_cup_map.py`, `robot_pose/mapped_socket_refine.py` |
| Two-camera RGB capture and joint matching | `challenge/realsense_capture.py` |
| Chessboard calibration and capture-profile binding | `challenge/camera_calibration.py`, `capture_calibration.py` |
| Measured trunk-marker baseline | `challenge/markers.py`; `config/perception/marker.example.json` |
| Persistent live processing and output-age gate | `challenge/online.py` |
| Existing team dataclass conversion | `challenge/team_api.py`, `team_frame.py` |

Here `challenge/` abbreviates `src/rcr/perception/challenge/`.
The three trained landmark models and their schemas are included under
`challenge/assets/weights/`. Public source commits, CAD checksums and provenance
are recorded in `assets/downloads.json`, `assets/provenance.json` and the setup
script. The native dependency lock is `scripts/perception/m3t-linux-64.lock`.
Setup downloads public dependencies and builds the supplied native adapter;
compiled executables and multi-GB environments are not committed.

## What is reproduced

`data/perception/holds/` contains five original stereo pairs, calibration and
post-inference references. `data/perception/robot/` contains five RGB packets,
measured joint configurations and separate evaluation records. Run the commands
in the [README](README.md) to produce predictions; run
`scripts/perception/verify_replays.py` afterwards to score robot predictions.
The included examples are regression cases, not a new blind benchmark.

The selected inference paths are supplied. Historical training runs, complete
benchmark datasets, and discarded research-package environments are not included.
The report separates earlier experimental results from this checkout's checks.

`labs/` retains optional small CPU geometry references from the earlier handoff.
It is not the selected estimator or the main entrypoint. Its installation is
independent of the operational runtime.
