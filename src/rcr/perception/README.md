# Perception

`challenge/` contains the selected stereo hold estimator, configured robot
estimator (`max` for saved frames), and persistent robot/cup tracking (`fast`).

See the [operating guide](../../../docs/perception/README.md) for isolated setup,
replay, RGB camera calibration, measured-joint timing and physical model selection.
The [team adapters](../../../docs/perception/TEAM_INTEGRATION.md) preserve existing
SI/WXYZ types and require verified start contacts for a `WallModel`.

The hold path fits Cube, Microban, Sidepull and Fingerpocket meshes in both RGB
views, estimates volume geometry, and recovers support/collision consistency.
It uses declared wall/camera calibration and no longer searches generator seeds.
Two hold regression fixtures and one robot fixture are included, with one trained
robot checkpoint. Alternative
models and historical experiments remain on `perception_challenge_full`.
All retained robot profiles require the 19 measured joints. No image-only robot
or rigid robot-marker estimator is included.

Legacy images-only `wall_model(images)` and `robot_pose(images)` remain stubs;
use the explicit packet/runtime API for calibration and telemetry. Physical
camera accuracy and controller integration remain unvalidated.
