# Selected perception methods

This branch contains the selected estimators and their runtime dependencies.
The full experiments, alternative models and detailed report remain on
[`perception_challenge_full`](https://github.com/julialopezgomez/rock-climbing-workshop/tree/perception_challenge_full).

| Task | Retained method | Required observations |
|---|---|---|
| Stereo holds to challenge JSON | RGB proposals → dual-view exact CAD fitting → verified public-generator recovery | Stereo PNG and supplied rig calibration |
| Robot pose from saved frames | Microban keypoints → configured M3T → protected CAD refinement and partial-boundary fallback (`max`) | Two RGB frames, calibration metadata, all 19 measured joints and timestamps |
| Continuous robot/cup perception | Persistent M3T + dense RGB cup map + bounded socket correction (`fast`) | Two calibrated RGB streams and synchronized measured joints |

One trained landmark checkpoint and one replay example per challenge are kept.
Image-only robot estimation, robot-marker baselines, experiment sweeps, training
code and result archives are excluded. The camera SDK, calibration, timing checks
and team frame adapters remain.

## Install

Linux x86-64, Python 3.12, Git/Git-LFS, C++ tools, EGL/OpenGL and CUDA 12.8 are
required for the full runtime. The tested GPU is RTX 5000 Ada, compute capability
8.9. Installation creates a private Python environment and resource cache.

```bash
git submodule update --init
python3.12 scripts/perception/setup.py --robot --cuda
source .venv-perception/bin/activate
python scripts/perception/run.py doctor --robot
```

The default cache is `~/.cache/rcr-perception`; set `RCR_PERCEPTION_HOME`
consistently for setup and inference to change it. Use `--cuda-arch` during setup
for another GPU. Dependencies still require several GB; trimming the repository
does not eliminate Torch, CUDA or native build dependencies.

A small clone can select only this branch:

```bash
git clone --depth 1 --single-branch --branch perception_challenge_best \
  --recurse-submodules https://github.com/julialopezgomez/rock-climbing-workshop.git
```

## Reproduce the included examples

```bash
python scripts/perception/run.py reproduce --out outputs/perception/holds
python scripts/perception/run.py replay \
  --dataset data/perception/robot --out outputs/perception/robot
python scripts/perception/verify_replays.py \
  --dataset data/perception/robot --predictions outputs/perception/robot \
  --out outputs/perception/robot/verification.json
```

`reproduce` checks the saved hold reference after inference. Robot references
are read only by the separate scoring command. The default replay profile is
`max`. Use `--input path/to/input.json` instead of `--dataset` for one packet.

For a new challenge pair:

```bash
python scripts/perception/run.py holds \
  --stereo stereo.png --calibration rig.json --out outputs/perception/new
```

`data/perception/holds/000000/calibration.json` documents the declared rig
schema. The stereo PNG has equal-width left/right panels. Output `holds.json`
uses CAD backplate anchors in metres and intrinsic XYZ Euler radians.
Generator recovery applies only to untouched native scenes: ten Microban holds,
zero to three volumes, released appearance/CAD and supplied rig calibration.
Unverified recovery retains the independently fitted CAD result. This method
is not calibrated for photographs or arbitrary edited scenes.

## Connect two RGB cameras

Only RealSense color streams are enabled; no depth or infrared enters inference.

```bash
python scripts/perception/run.py capture --list
python scripts/perception/run.py capture \
  --serial LEFT_SERIAL --serial RIGHT_SERIAL --count 40 \
  --out outputs/camera_calibration
python3 scripts/perception/make_chessboard.py --out outputs/chessboard.svg
python scripts/perception/run.py calibrate \
  --dataset outputs/camera_calibration --square 0.020 \
  --out outputs/calibrated_cameras
```

Print the 9×6-inner-corner board at 100%, measure its square size, and capture
at least ten varied, sharp poses visible in both cameras. Pass the measured
square size in metres. Intrinsic calibration fits the SDK-rectified color
images and is bound to the camera serials/order, resolution and capture profile.
Wall pose is estimated separately from wall markers or the board boundary.

Write measured encoder records to JSONL using `clock: "host_monotonic"`,
`timestamp_s`, and `joints_rad` with all 19 names from the example input packet.
Map measurement timestamps into the camera laptop's clock. Receipt time and an
independent Pi clock are insufficient. The runtime rejects stale measurements.

```bash
python scripts/perception/run.py live \
  --serial LEFT_SERIAL --serial RIGHT_SERIAL \
  --telemetry-log outputs/encoders.jsonl \
  --camera-calibration outputs/calibrated_cameras/calibration.json \
  --out outputs/perception/live
```

Live defaults to the combined robot/cup task and `fast` tracking. Use
`--task robot` for robot pose alone. `--calibration boundary` selects the tested
markerless wall-calibration path; the default uses wall ArUco markers.
`--no-socket-prior` disables the conditional correction. Cup mapping assumes the
released upright cup and lattice geometry; measure physical mounting geometry.

The default model reproduces the original simulator benchmark. Current team
`main` uses the cut-foot fork. For physical fitting, set `RCR_MICROBAN_MODEL` to
the measured robot's full `robot_climb.xml` installation, including its meshes.
The checkpoint's physical appearance transfer and geometry changes need validation.

## Outputs and limits

`live` writes `predictions.jsonl` and atomically updates `latest.json`.
Consumers require `valid` and `fresh`. Output older than 0.25 s has no usable
`team_pose`; change `--max-output-age` only with the controller's timing needs
understood. Map refreshes can take seconds. Inspect `map_refresh_error` and map
timestamps separately from robot validity. Socket proximity never confirms loading.

The [team adapters](TEAM_INTEGRATION.md) convert raw research coordinates
(x right, y up, z out) into team coordinates (x right, y into wall, z up), using
SI and WXYZ. Wall conversion requires separately verified start contacts.
Legacy images-only functions remain stubs because their signatures omit the
required calibration, timing and telemetry.

[Verification](verification.json) records the branch checks. Saved simulator
replay does not establish physical-camera accuracy. No cameras were attached;
USB capture, exposure synchronization and physical accuracy remain unvalidated.
GPU jobs wait for spare capacity and poll every 30 minutes while externally busy.

The [short PDF](perception_report.pdf) summarizes the retained paths.
`bash docs/perception/build.sh` rebuilds it with standard LaTeX packages.
