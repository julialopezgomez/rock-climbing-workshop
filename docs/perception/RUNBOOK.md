# Operating the perception runtime

Run commands from the repository root. `scripts/perception/setup.py` installs
only the private runtime; activate it or use `.venv-perception/bin/python`.
`RCR_PERCEPTION_HOME` optionally selects a different cache. If set, use the same
value for setup, inference and verification.

## 1. Check the installation

```bash
.venv-perception/bin/python scripts/perception/run.py doctor --robot
```

The setup requires Git/Git-LFS, a C++ compiler, EGL/OpenGL runtime libraries and
CUDA 12.8 development tools for `--cuda`. Its native OpenCV/M3T dependencies use
an explicit conda-forge package lock. Without `--robot`, the core supports the
hold estimator and camera/marker path; `--cameras` adds the RealSense SDK.

The default extension target is `8.9+PTX` (Ada). For another supported card, set
the correct `--cuda-arch` during setup, or `CUDA_ARCH` when building Docker.
For example an Ampere build can use `--cuda-arch 8.6+PTX`. The Python/Torch pins
remain unchanged. The binaries/cache must be rebuilt when changing this target.

GPU jobs check ambient utilisation/free memory, serialize through a cache lock,
and check again every 30 minutes when externally busy. They never stop another
workload. `RCR_GPU_LOCK` can select a shared lock path on a multi-project machine.

## 2. Run challenge stereo images

```bash
.venv-perception/bin/python scripts/perception/run.py holds \
  --stereo path/to/stereo.png --calibration path/to/rig.json \
  --out outputs/perception/prediction
```

The PNG contains equal-sized left/right panels. The calibration follows the
included `data/perception/holds/000000/calibration.json`: K and
`T_camera_scene` for each camera, image size and source scene geometry.
Output is `holds.json`, a compatible scene envelope and diagnostic/recovery JSON.
Positions are metres; rotations are intrinsic XYZ Euler radians; the reference
point is the CAD backplate anchor. Volumes in the scene envelope are not claimed
to be reconstructed exactly.

Default mode performs independent CAD fitting and then source recovery. Use
`--cad-only` for the independent RGB/CAD result. `--seed-backend cpu` replaces
only the seed scan; MuJoCo rendering still uses the configured graphics backend.
The exact result requires the unedited public native distribution. Do not use
that assumption for arbitrary edits, Cube holds or camera photographs.

`reproduce --out DIRECTORY` runs five included examples and compares saved
reference JSON only after inference. Its summary retains failures.

## 3. Capture color frames

```bash
.venv-perception/bin/python scripts/perception/run.py capture --list
.venv-perception/bin/python scripts/perception/run.py capture \
  --serial LEFT_SERIAL --serial RIGHT_SERIAL \
  --width 1280 --height 720 --fps 30 --count 40 \
  --out outputs/camera_calibration
```

Only RGB streams are enabled. Color lens distortion is rectified using SDK
projection. Packets contain pixel-centre K, camera serials, frame times and
source intrinsics. The pair gate is 15 ms in a common host-monotonic clock.
Frame matching does not guarantee simultaneous exposure.

For Docker, USB access typically requires explicit host device access:

```bash
docker run --rm --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  --device=/dev/bus/usb:/dev/bus/usb \
  -v "$PWD:/app" rcr-perception capture --list
```

Device permissions depend on the host. No real USB capture was possible here.

## 4. Measure camera calibration

Generate a 9x6-inner-corner chessboard, with 20 mm squares, on landscape A4:

```bash
python3 scripts/perception/make_chessboard.py --out outputs/chessboard.svg
```

Print at 100%, measure the actual square size, and use that measurement below.
Capture at least 10 successful pairs (prefer 20--40), varying distance and tilt
while keeping the entire board visible in both images. Avoid blur and nearly
identical poses. Keep the board static briefly for each accepted pair.

```bash
.venv-perception/bin/python scripts/perception/run.py calibrate \
  --dataset outputs/camera_calibration --square 0.020 \
  --out outputs/calibrated_cameras
```

The program estimates per-camera intrinsics/residual distortion and stereo
relative geometry from these SDK-rectified color frames. It reports accepted
and rejected pairs and monocular/stereo RMS. Inspect the residuals; repeat poor
coverage. The result does not establish the cameras' pose relative to the wall.

Apply it to the identical serials, order, image size and capture profile using
`--camera-calibration outputs/calibrated_cameras/calibration.json` on `capture`
or `live`. The runtime checks that binding. Do not apply these residual
coefficients directly to raw, unrectified sensor images.

## 5. Locate the physical robot with a marker

Mount marker ID 5 rigidly on the trunk. Measure its **black code** size and
`T_trunk_marker`, mapping marker coordinates into the model trunk. Copy
`config/perception/marker.example.json`, replace its dimensions/transform, and
set `measured` to true only after calibration. The example deliberately refuses
to run as a measurement.

Wall markers use DICT_4X4_50 IDs 1--4. The default board is 0.8 x 1.4 m with 0.1 m black
codes, 0.018 m white borders and the released corner placement. For a different
measured arrangement, provide `wall.marker_corners_wall_m` in the marker config,
as an ID-to-four-3D-corners mapping in research wall coordinates (right, up,
out), ordered TL/TR/BR/BL. Two observed wall markers per used view are required.

```bash
.venv-perception/bin/python scripts/perception/run.py live \
  --serial LEFT_SERIAL --serial RIGHT_SERIAL \
  --estimator markers --marker-config config/perception/my_marker.json \
  --camera-calibration outputs/calibrated_cameras/calibration.json \
  --out outputs/perception/marker_live
```

This mode needs no GPU or encoder readings. It jointly fits marker observations
from the calibrated views and exports the trunk pose in the team wall basis.
It does not confirm contacts or loading. A marker attached to an articulated
link requires a time-varying kinematic transform and is not supported by this
fixed-trunk configuration.

## 6. Run the learned CAD estimator and cup map

For measured joints, supply a JSONL log updated by the telemetry bridge. Every
record needs `clock: "host_monotonic"`, `timestamp_s` and `joints_rad` containing
all 19 model names. The measurement time must be mapped into the laptop's clock;
a Pi's independent monotonic clock or packet receipt time is not equivalent.
The adapter interpolates only nearby measured samples and rejects stale data.

```bash
.venv-perception/bin/python scripts/perception/run.py live \
  --serial LEFT_SERIAL --serial RIGHT_SERIAL \
  --profile fast --task live --telemetry-log outputs/encoders.jsonl \
  --camera-calibration outputs/calibrated_cameras/calibration.json \
  --out outputs/perception/learned_live
```

`fast` retains the persistent tracker. `max` adds the selected protected exact
CAD and partial-boundary search for offline/high-compute use. `trunk` estimates
only the root without joints; it does not recover articulation. The included
networks were trained on the released simulator; physical appearance transfer
is unvalidated. Keep the measured-marker mode as a calibration/reference option.

`--calibration boundary` removes wall-marker dependence but retains the tested
board/appearance assumptions. `--no-socket-prior` disables the conditional
image-supported socket correction. Cup mapping assumes the released upright cup,
wall and lattice geometry; measure the physical setup before using those priors.

## 7. Read the output safely

`live` writes `predictions.jsonl` and atomically replaces `latest.json`.
Every result retains its acquisition timestamp and output age. Outputs older
than `--max-output-age` (default 0.25 s) do not expose a usable `team_pose`.
`Ctrl-C` closes the owned capture pipelines and estimator.

`replay --input input.json` or `--dataset DIRECTORY` processes saved packets;
it preserves failed frames with explicit reasons. Inputs follow the included
robot packets: RGB paths, camera K/size/ID/time, common clock and, for known-joint
profiles, exactly 19 measured joint angles. Do not supply simulator truth/depth
as actor data.

The raw research pose uses x right, y up, z out. `team_pose` converts it into
the team's x right, y into wall, z up frame, in metres with WXYZ quaternion.
`team_api.robot_pose` returns the existing team `Pose` dataclass. For team
`WallModel`, `team_api.wall_model` requires separately verified start contacts
and uses mounting transforms rather than socket centres. It rejects unobserved
start holds and never infers loading.

The legacy images-only perception entrypoints lack calibration, capture times
and telemetry. Use this explicit packet/runtime API; adapting a controller call
site requires agreement on those inputs, not a silent guessed measurement.

## Robot geometry for physical use

The private runtime pins Rhoban's original climbing model to reproduce the
included simulator benchmark. Current team `main` uses the cut-foot fork pinned
in `pyproject.toml`. These models must not be interchanged silently.
For physical CAD fitting, point `RCR_MICROBAN_MODEL` at the measured robot's
complete MJCF installation, including its referenced meshes:

```bash
export RCR_MICROBAN_MODEL=/absolute/path/to/team-model/mjcf/robot_climb.xml
```

The runtime hashes configured geometry before rebuilding M3T exports. The
included networks and benchmark numbers remain tied to the original synthetic
appearance. Recheck overlays and pose/endpoint accuracy after any geometry
change. Marker-only rigid-trunk operation does not load the articulated CAD.
Use the default original model for the included regression commands.

## Rebuild the report

```bash
bash docs/perception/build.sh
```

This requires a standard TeX installation with Latin Modern, microtype,
booktabs and hyperref. Figures are included. To regenerate the repository robot
figure from new scored outputs, install Matplotlib in the private environment
and run `scripts/perception/plot_replays.py --help`. The older
`prepare_figures.py` imports historical experiment archives; it is not part of
runtime installation or report compilation.
