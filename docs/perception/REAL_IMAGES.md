# Physical RGB recordings

[Ours] `scripts/perception/process_recordings.py` processes four independent
RealSense color recordings: `l_without_rob.bag`, `r_without_rob.bag`,
`l_with_rob.bag` and `r_with_rob.bag`. Input may be their directory or the supplied
`real_examples.tar`. This is a separate physical-cup path; the four-type image
challenge uses a different coordinate origin and JSON contract.

## Run

For camera localization and cup mapping, install the isolated core and RGB SDK:

```bash
python3.12 scripts/perception/setup.py --cameras
source .venv-perception/bin/activate
python scripts/perception/process_recordings.py \
  --input samples/real_examples.tar --out outputs/perception/real
```

To reproduce the tested robot stages, supply a private SAM 3 environment and its
local checkpoint. `setup.py --robot` prepares the existing robot environment;
SAM 3 remains an optional external dependency and is not installed into the core.

```bash
python scripts/perception/process_recordings.py \
  --input samples/real_examples.tar --out outputs/perception/real_full \
  --sam3-python /path/to/sam3/bin/python \
  --sam3-checkpoint /path/to/sam3.pt \
  --robot-python /path/to/robot-environment/bin/python
```

GPU stages use the existing capacity guard and never stop unrelated processes.
Without the optional environments, camera/cup processing still runs and the
unavailable robot stages are recorded explicitly. A complete run from an empty
output folder was tested with the supplied archive's extracted recordings.
No real photographs or bag files are committed to this branch.

## Camera placement from the board

The workflow reads **color-camera intrinsics from each bag**. The separate
`calib_l/r` exports contain the device's internal infrared calibration, including
its roughly 50 mm internal baseline. That baseline is not the distance between
the two externally placed cameras. No depth or infrared frames are read.

The main board is partially cropped and has no detected ArUco corner markers.
The previous outline detector fails on these images. The replacement detects
the dense hole array, indexes a local patch, expands its homography and fits a
metric planar camera pose to thousands of hole observations. Each recording
receives its own camera pose; the right camera moved approximately 0.25 m between
empty-wall and robot captures.

The teammate's manual-corner and printed-corner tools are preserved. Manual
PnP requires supplied corner coordinates. The automatic printed-corner mode
does not find a complete four-corner set in any of these recordings; its texture
mask is a diagnostic and does not establish the missing outer corners.

The declared geometry defaults are 0.8 m × 1.4 m, 15 mm hole spacing and 10 mm
first-hole offsets from the left/top edges. Override these with measured values:

```text
--wall-width 0.8 --wall-height 1.4 --pitch 0.015
--first-hole-left 0.010 --first-hole-top 0.010
```

The 10 mm offsets are provisional physical assumptions. The simulator's known
horizontal margin is 17.5 mm; that value does not independently establish the
physical board's margin. Repeated holes also permit integer index offsets when
outer rows are missing. Outputs therefore retain `grid_origin_verified=false`.

| Recording | Camera x (m) | y (m) | z (m) | Grid RMS (pixels) |
|---|---:|---:|---:|---:|
| Left, empty | −0.455 | 0.818 | 1.624 | 0.449 |
| Right, empty | 1.003 | 1.068 | 1.460 | 0.778 |
| Left, robot | −0.455 | 0.821 | 1.624 | 0.496 |
| Right, robot | 1.060 | 0.895 | 1.632 | 0.391 |

Axes are x right, y up, z out, with a provisional bottom-left board origin.
Across three extra frames per recording, camera differences from the selected
frame were 0.31–4.13 mm and 0.008–0.131°. This is short-recording repeatability,
not absolute accuracy. Uniformly scaling the declared geometry preserves pixel
fit while changing metric positions; the tests demonstrate that dependence.

## Cup map

The board is rectified at an initial socket plane. Orange/blue components are
split along the board's horizontal direction to separate touching cups. The
existing public cup CAD is fitted in both RGB views, with depth fitted for paired
observations and a declared socket-plane prior for single-view observations.
Scales 1.0, 1.15 and 1.25 were compared on 12 complete components; scale 1.0 had
the lowest median boundary loss and was retained.

SAM 3 supplies robot masks that exclude robot parts from cup proposals. The
empty-wall map remains a static prior during the robot-present pass. New
robot-present candidates require two-view support and are checked against the
empty images, preventing a robot part from becoming a new hold.

| Pass | Left instances | Right instances | Fitted hypotheses | Image-supported |
|---|---:|---:|---:|---:|
| Empty wall | 61 | 48 | 62 | 60 |
| Robot present, SAM exclusion | 54 | 46 | 62 | 52 |

The fused map retains 60 previously observed cups; 51 are re-associated with
robot-present observations. One additional supported robot-present proposal is
rejected by the static-map consistency check. Hidden or unobserved holds are
not completed from a route. Counts are diagnostic support counts, not measured
precision or recall. Mounting/socket offsets still depend on the public CAD.

## Robot pose

The existing whole-body network was tested at full resolution and several crop
sizes. It misses the robot in the left view and detects it weakly in the right.
The dedicated trunk network was also tried separately with RGB and grayscale
crops; it did not supply a usable two-view initializer. No real-data training
was performed. SAM 3 segments the robot in both images.

The rope crosses the printed marker. Normal decoding, including dictionary and
scale trials, does not identify the robot marker reliably. White-label and dark
border geometry provide an initialization, with uncertain orientation.

The experimental alternative extracts only the **rigid root-body meshes** from
the public robot model. It fits them to both SAM masks and RGB edges from several
initial poses. A trimmed boundary loss tolerates boundaries adjoining moving
links. The label initializes the fit; no measured marker-to-trunk transform is
used. Moving links, joints and endpoints are excluded from this registration.

The selected root hypothesis is approximately `(0.460, 0.594, 0.088)` m in the
provisional wall frame. Both rendered views and all candidate fits are saved.
No surveyed root pose was supplied, so no translation/rotation accuracy is claimed.

## Outputs and remaining measurements

Open `overview.html` or `real_images_report.pdf`. The directory also contains:

- `decoded/`: selected RGB, bag metadata, camera transforms, grid overlays and repeatability.
- `wall_map.json`: static fused cup hypotheses and observation status.
- `without_rob_map.json`, `with_rob_map.json`: raw fit/support diagnostics.
- `sam.json`, `robot_masks.npz`: segmentation provenance and masks.
- `keypoints.json`: learned frontend trials, including failed detections.
- `label_hypothesis.json`: occluded label geometry and alternative square fits.
- `trunk.json`: provisional root transform, WXYZ quaternion, candidate fits and mesh hash.
- `summary.json`: complete stage status and limitations.

The selected robot images were recorded **17.021 s apart**; the empty-wall images
are 105.056 s apart. A stationary-scene assumption supports this offline test.
It does not establish synchronized exposure or a valid live state. There is no
encoder log or independent pose/hold reference. The configured whole-body path
therefore remains unavailable, and outputs retain `valid_for_control=false` and
`team_pose=null`.

Confirm board/grid dimensions and offsets, capture a synchronized pair, and
measure a reference robot pose before using these estimates for control. Supply
synchronized encoders when the configured estimator or endpoint poses are needed.
Shared team types, the four-type benchmark, simulator submodules and existing
`max`/`fast` robot implementations are unchanged.
