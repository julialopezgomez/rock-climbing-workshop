# perception

**Purpose:** say what the robot faces: where the holds are, where the robot starts, and where the robot is.

| | |
|---|---|
| Input | RGB images from the 2 USB wall cameras, one on each side, plugged into our laptop [Rules §5.1]. Only the RGB part of the streams may be used |
| Output | `WallModel` (holds + start stance) via `wall_model(images)`; `Pose` of the robot base in the wall frame via `robot_pose(images)` |
| Runs on | laptop |

## For now
Real perception is parked (see ROADMAP). The pipeline loads a **hand-written wall model** with `rcr.io.load_wall`
(e.g. `data/examples/wall_l1.json`). Both functions stay stubs.

## Where real detection plugs in
- Camera images go into `wall_model()` / `robot_pose()`. Implementation files go next to `__init__.py`, and `__init__`
  delegates to them, so callers do not change.
- **With markers** [Rules §5.2, §6.1]:
  - ArUco DICT_4X4_50 markers 1–4 sit on the wall corners. Their centres are given in INTERFACES.md: 0.100 m code, 0.136 m with the border.
  - Detecting them gives each camera's extrinsics in the wall frame.
  - Optional marker ID 5 on the robot (any size, anywhere) gives `robot_pose` directly.
  - The robot can instead be detected without a marker, with the wall markers still giving the extrinsics.
- **Markerless** [Rules §6.2]: no markers at all, and the organisers move the cameras slightly before each attempt, so the camera pose
  must be estimated by us. Time factor ×1, against ×1.25 with markers [Rules §8].
- **In the official sim**, perception works from the camera images MuJoCo renders [Rules §7].
- A live `robot_pose` can be sent to the robot over the link (`LaptopLink.send_pose`) if the controller needs it.

## v0 could be
- hand-written walls built from grid indices (hold x = 0.0175 + 0.015·i, z = 0.010 + 0.015·j) [Rules §4];
- later, ArUco 1–4 for extrinsics plus hand-entered hold grid positions;
- later still, holds detected in the images.

## Open decisions
- Marker vs markerless (score factor ×1.25 vs ×1).
- Whether we use ID 5 on the robot, and its size and placement.
- The detection library. OpenCV's `aruco` would go in the `perception` dependency group, never on the Pi.
- Whether the robot pose is needed live by the controller at all.
