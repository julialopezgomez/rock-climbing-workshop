# Selected perception methods

The [31 October scored-benchmark update](SCORED_BENCHMARK.md) adds the released
hold/volume scorer and an optional native-generator recovery mode. It scores
zero on 40 supplied scenes, 100 development/regression scenes and 25 separate
held-out scenes. Native recovery requires an unedited scene from the pinned
sampler and a declared hold count; Cube and manually edited scenes use the
general four-mesh CAD path below. The current upstream default scores holds
only; the update reports both scoring modes.

This branch contains the selected estimators and their runtime dependencies.
The full experiments, alternative models and detailed report remain on
[`perception_challenge_full`](https://github.com/julialopezgomez/rock-climbing-workshop/tree/perception_challenge_full).

| Task | Retained method | Required observations |
|---|---|---|
| Stereo holds to challenge JSON | Four-mesh CAD/support fit; optional verified native-generator recovery | Stereo PNG and declared calibration; native mode also requires the generator hold count |
| Robot pose from saved frames | Microban keypoints → configured M3T → protected CAD refinement and partial-boundary fallback (`max`) | Two RGB frames, calibration metadata, all 19 measured joints and timestamps |
| Continuous robot/cup perception | Persistent M3T + dense RGB cup map + bounded socket correction (`fast`) | Two calibrated RGB streams and synchronized measured joints |
| Real recording review, experimental | Hole-lattice camera localization → cup CAD fitting → SAM 3 masks → rigid-trunk fitting | Four RGB bags; declared board/grid geometry; optional isolated SAM 3 and robot environments |

One trained landmark checkpoint, two hold fixtures and one robot fixture are kept.
Image-only robot estimation, robot-marker baselines, experiment sweeps, training
code and result archives are excluded. The camera SDK, calibration, timing checks
and team frame adapters remain.

## Install

For hold inference, use Linux x86-64, Python 3.12 and working EGL/OpenGL drivers.
MuJoCo renders the CAD hypotheses on the GPU; this path does not require Torch
or a CUDA compiler. Installation creates a private environment and downloads four
checksum-verified meshes. The tested GPU is RTX 5000 Ada.

```bash
python3.12 scripts/perception/setup.py
source .venv-perception/bin/activate
python scripts/perception/run.py doctor
```

For robot tracking, additionally run `setup.py --robot`; this needs Git/Git-LFS,
C++ tools and a compatible Torch/CUDA installation. `--cuda` checks the installed
CUDA compiler; `--cuda-arch` selects the robot extension target (tested: 8.9).
`setup.py --cameras` installs color-only RealSense capture support.

The default cache is `~/.cache/rcr-perception`; set `RCR_PERCEPTION_HOME`
consistently for setup and inference to change it. Robot dependencies require
several GB. They are optional for the stereo hold challenge.

A small clone can select only this branch:

```bash
git clone --depth 1 --single-branch --branch perception_challenge_best \
  --recurse-submodules https://github.com/julialopezgomez/rock-climbing-workshop.git
```

## Reproduce the included examples

```bash
python scripts/perception/run.py reproduce --workers 2 --out outputs/perception/holds
python scripts/perception/run.py replay \
  --dataset data/perception/robot --out outputs/perception/robot
python scripts/perception/verify_replays.py \
  --dataset data/perception/robot --predictions outputs/perception/robot \
  --out outputs/perception/robot/verification.json
```

`reproduce` checks the saved hold reference after writing predictions. Its exit
status reports whether inference and scoring completed; inspect `strict_poses`,
matches, missed holds and false positives for accuracy. Robot references
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
Supported types are `cube_hold`, `microban_hold`, `sidepull_hold` and
`fingerpocket_hold`. The mesh manifest pins the public source at
[`51efe48`](https://github.com/jackybaltes/rcr-perception/tree/51efe48b9765a0de7c3a4658a8b7bb781faf3459).
The source's `sidehpull_hold.stl` filename is intentional. Geometry uses Blender
axis conversion, a centred backplate origin and metric dimensions.

Set `wall_width_m` and `wall_height_m` in calibration; optional command-line
`--wall-width` and `--wall-height` override them. The supplied scenes use a
3 m × 1.4 m wall, not the former 0.8 m width. Tetrahedral volume base height is
`min(0.1 * wall_width, wall_height)` unless `volume_base_height_m` is declared.
Camera intrinsics/extrinsics must describe each actual stereo panel.

### Supplied sample archives

For the declared sample rig (64 mm baseline, 1 m distance, 0.7 m camera height,
74° vertical field of view), prepare the two ZIP archives and run all cases:

```bash
python scripts/perception/prepare_samples.py \
  --samples samples --out data/local/perception_samples
python scripts/perception/run.py reproduce \
  --dataset data/local/perception_samples --workers 4 \
  --out outputs/perception/samples
```

The preparation script writes separate calibration and scoring files. Change its
rig flags for images generated with other settings. It does not infer calibration
from the labelled hold positions. `--only CASE_ID ...` selects a short rerun.

### What the estimator does

Color components provide stereo proposals. Each proposal is compared with all
four CAD meshes at the three public yaw choices (0°, −45°, +45°). Flat RGB face
regions estimate the volumes. Both cameras then constrain wall/volume support,
translation and inclined-pose refinement. Fingerpocket candidates receive an
additional stereo RGB consistency test using the candidate CAD-rendered depth.
No sensor depth enters this test.

The final recovery aligns volume-mounted backplates to their estimated support
faces, applies the source's 1.73 mm clearance, and resolves convex-hull conflicts
by moving the weaker fit along its support plane. Unresolved fits are rejected.
This replaces the obsolete ten-Microban seed search in the general image path.
The separate [scored benchmark command](SCORED_BENCHMARK.md#run-it) implements
updated seed recovery as an explicit option. The current proposals and geometry
priors are specific to rendered challenge scenes; photographs need validation.

Outputs are `image_holds.json` (raw image fits), `holds.json` (recovered poses),
`prediction.json` (viewer scene envelope with estimated volumes) and `debug.json`
(fit losses, geometry and recovery changes). Preserve the debug file when a
viewer scene is accepted but the pose score is poor: those are separate checks.

### Measured results

The [benchmark record](benchmark.json) preserves all 40 per-scene summaries.
Matching uses correct type and ≤50 mm position error; a strict pose additionally
requires <5 mm and <15°. These are local diagnostics, not an official score.

| Supplied 40 scenes, 1,200 holds | Typed matches | Strict poses | False positives | Viewer accepted |
|---|---:|---:|---:|---:|
| Frozen image fit | 1,160 | 999 | 7 | 23/40 |
| Final support/hull recovery | 1,159 | 1,005 | 8 | 40/40 |

Four scenes guided image-fit development; the other 36 were evaluated after
freezing those files. The viewer compatibility correction was added after
inspecting full-set validation failures. Final results are regression results,
not a new blind test. The correction improves compatibility and some poses,
while one former match moves outside the matching tolerance.

The supplied archives contain 429 Microban, 349 Sidepull, 422 Fingerpocket and
**zero Cube** holds. Three additional controls extend only the source sampler's
type list to include Cube: 47/48 matches, 37/48 strict; Cube alone gives 11/12
matches and 10/12 strict. These are separate synthetic controls. The dominant
remaining error is Fingerpocket yaw; 141 matched holds are within 5 mm but fail
the rotation threshold. Small images, partial visibility and uncertain volume
faces still cause missing or misplaced instances.

## Connect two RGB cameras

### Process the supplied real recordings

The separate [real-image workflow](REAL_IMAGES.md) reads the four `.bag` files
directly, estimates camera placement from the perforated board and saves cup
maps, an experimental rigid-trunk hypothesis, overlays, HTML and a LaTeX/PDF report.
It uses RGB only. The four-type synthetic hold estimator above remains separate.

### Capture and calibrate a live pair

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
After reproducing the two fixtures, `python scripts/perception/report_figures.py
--predictions outputs/perception/holds` rebuilds the scoring overlays.
