# Perception

`challenge/` contains the selected stereo hold estimator, configured robot
estimator (`max` for saved frames), and persistent robot/cup tracking (`fast`).

See the [operating guide](../../../docs/perception/README.md) for isolated setup,
replay, RGB camera calibration, measured-joint timing and physical model selection.
The [team adapters](../../../docs/perception/TEAM_INTEGRATION.md) preserve existing
SI/WXYZ types and require verified start contacts for a `WallModel`.

One trained checkpoint and one example per challenge are included. Alternative
models and historical experiments remain on `perception_challenge_full`.
All retained robot profiles require the 19 measured joints. No image-only robot
or rigid robot-marker estimator is included.

Legacy images-only `wall_model(images)` and `robot_pose(images)` remain stubs;
use the explicit packet/runtime API for calibration and telemetry. Physical
camera accuracy and controller integration remain unvalidated.
