# Ten-minute perception submission

## Selected version: image_only_best_20261101_1202

Restored on 3 November 2026. The runtime, selected weights and launcher match
all 13 hashes recorded for this version. The saved
[baseline manifest](baselines/image_only_best_20261101_1202.json) records the
source hashes, model hashes and inference settings.

Restoration checks passed: 35 targeted tests and a fresh four-scene grouped run
(one scene from each archive) in 44.22 seconds. All four predicted JSON documents
are identical to the saved version, and the original image bytes are preserved.

```bash
cd /workspace/students/rock-climbing-workshop
export RCR_GPU_LOCK=/workspace/development/outputs/perception/gpu_experiment.lock
./perception_submit_image_only.sh /workspace/perception/samples \
  --seconds 475 --out "outputs/image_only_$(date +%Y%m%d_%H%M%S)"
```

The command reads `real/*.zip` and `sim/*.zip`, then writes identically named
ZIPs under the output folder's `real/` and `sim/` directories. Each ZIP preserves
the original images and adds their predicted JSON siblings. Input answer JSON
files are ignored. The timestamp gives each run a fresh output directory.

Scene-seed search and generator matching are disabled. Four workers share one
475-second budget across preparation, fitting, retries and packaging. For 80
images, initial tasks receive 23 seconds each. Remaining time retries unfinished
fits, up to 45 seconds per retry. A retry replaces an earlier result only when
it completes; otherwise the saved estimate is retained.

### Method

A fixed paint classifier separates hold paint from background features in
real-style images. Connected components are associated across the stereo views
with a 2.5-pixel epipolar tolerance. CPU CAD estimates initialize multiview CAD
fitting, which retains original RGB for photometric comparisons. Stereo color
clustering estimates volume surfaces. A local crop classifier rejects background
hypotheses and chooses among fitted CAD types. Uncertain wall Fingerpocket yaw
estimates are reduced by half. Support and collision checks finish the scene.

This version uses the original paint components and volume detector. The later
center network, wider stereo gate, additional volume-color search, separate
support-refitting pass and final candidate-selection filter are not active.
All four CAD meshes remain available; the supplied archives do not establish
Cube accuracy.

The packaged paint and crop models use development labels rendered from
previously recovered scenes: 24 real-style images for paint and 48 stereo images
for crop classification. Inference reads RGB, declared calibration, CAD and fixed
weights. It reads no scene seeds, reference JSON or earlier reconstructed answers.
Model cards and training commands are in
[paint_forest.json](../../src/rcr/perception/challenge/assets/weights/paint_forest.json)
and [hold_crop_model.json](../../src/rcr/perception/challenge/assets/weights/hold_crop_model.json).

### Recorded benchmark for this version

The original 1 November run completed all 80 scenes in **383.33 seconds**,
including 16 completed refinement retries, with zero runtime errors. These are
the saved version's benchmark results, not a new full-batch run on 3 November.
The actual output ZIPs were checked with the upstream JavaScript scorer and
preserve the input images. Lower scores are better.

| Archive | Scenes | Mean hold score |
|---|---:|---:|
| real_v0 | 20 | 9.3142 |
| real_v3 | 20 | 38.4312 |
| synth_v0 | 20 | 8.2754 |
| synth_v3 | 20 | 19.8883 |
| All | 80 | 18.9773 |

There are 2,317 position matches among 2,400 reference holds, with 83 missing
holds, 35 extras and 11 matched type errors. Matched position error has a
0.560 mm median and 1.721 mm 90th percentile. Inclined and occluded holds remain
difficult: 38 inclined holds are missing. Volume matches are 109/120.

The 32-image reserved-subset mean is 23.0554. Those images were excluded from
training for this version but had been evaluated in earlier experiments; they
are not a fresh blind test. Scores use reconstructed references, not independently
supplied organizer labels. A later audit found a suspected reference-geometry
discrepancy in `real_v3/0014`; that case and its score remain included unchanged.
The `real` archives use textured rendering. These results do not establish
physical-camera accuracy.

The saved delivery and evaluation remain under
`outputs/image_only_best_20261101_1202/`:

```text
real/real_v0.zip
real/real_v3.zip
sim/synth_v0.zip
sim/synth_v3.zip
evaluation/scores.json
image_only_provenance.json
```

Local output archives and experiment backups are not included in Git.

## Earlier generator-assisted runs

The commands and zero-score results below used seed recovery. They are retained
as historical results and do not apply to the image-only mode above.

### Real/sim folders

Put the release ZIPs in two directories:

```text
/workspace/perception/samples/
    real/
        real_batch.zip
    sim/
        sim_batch.zip
```

These directories are ready in this workspace. Each ZIP may contain several
stereo images. Run both groups together:

```bash
cd /workspace/students/rock-climbing-workshop
export RCR_GPU_LOCK=/workspace/development/outputs/perception/gpu_experiment.lock
./perception_submit_folders.sh /workspace/perception/samples --out outputs
```

The result preserves the group and archive names:

```text
outputs/
    real/real_batch.zip
    sim/sim_batch.zip
    perception_manifest.json
```

Each output ZIP contains the original images, byte for byte, and their predicted
JSON files at the original internal directory paths. For example,
`real_batch/scene_cam1_0001.png` is paired with
`real_batch/scene_0001.json`. Input JSON files are ignored and replaced by
predictions. Image basenames may repeat in different archives. Existing output
archives are protected; use a fresh `--out` directory for another attempt.

The command counts stereo images across both folders, including every ZIP.
Its total budget is `min(480, 24 × image_count)` seconds from command startup,
including input preparation and output packaging. The image-fit cap is
`min(24, total_budget / image_count)` seconds. Twenty images therefore allow
24 seconds per fit; forty allow 12. The generator search shares the batch budget
and is not run independently for each image.

Four image jobs run concurrently. The GPU fitter prioritizes well-supported
components across all three paint colours and saves completed hold fits before
its timeout. Eight worker processes generate and verify native-scene candidates.
A partial fit replaces the current submission only after native verification
accepts a complete scene. The manifest records errors and unresolved cases;
output ZIPs update throughout the run. `--seconds 360` reduces the shared limit
if less time remains. Increasing `--seconds` cannot exceed the eight-minute cap.

### Timed folder tests

The final code completed both groups within the same budget. Scores below use
the supplied upstream scoring function, applied to the JSONs inside the actual
delivered ZIPs. Zero is the minimum error in both modes.

| Run | Images | Per-image fitting cap | Total elapsed | Hold score | With volumes |
|---|---:|---:|---:|---:|---:|
| Initial strict cap | 10 real + 10 sim | 24 s | 148.02 s | 0 | 0 |
| Optimized | 10 real + 10 sim | 24 s | **118.38 s** | **0** | **0** |
| Optimized | 20 real + 20 sim | 12 s | **169.35 s** | **0** | **0** |

The initial run killed one optional GPU fit and retained a correct earlier
prediction, leaving an uncertainty warning. Both optimized runs completed with
no errors or uncertain outputs. The 40-image run used partial CAD anchors in ten
of its eleven GPU refinements; native recovery then reconstructed each complete
scene. Its slowest saved GPU fit took 10.64 s from task creation to prediction.
The slowest fit in the 20-image run took 13.65 s, below its 24 s cap.

Repeated raster results are reused without changing the fitting objective.
Completed hold pairs survive the internal fitting cutoff. Candidate generation
and verification run in separate worker processes, while each OpenCV worker uses
one thread to avoid competing thread pools. A cached 20-case check produced
identical predictions and diagnostics with serial and eight-worker verification;
that stage took 17.95 s and 8.34 s respectively on the GPU renderer.

The 20-case subset deliberately includes both prior CPU failure cases and overlaps
the 40-case set. These are regression tests of this distribution, not 60 independent
scenes. All original image bytes, archive paths and predicted JSON names were
checked. [Detailed results](grouped_submission_results.json) retain errors from
the initial trial, per-image times, scores and source hashes.

An eight-second-budget interruption test finished in 3.97 s with all
20 original images and explicit empty JSON fallbacks packaged into the four
archives. It returned exit code 2 and a partial manifest.

## Single submission ZIP

Run from the team repository. The runtime and model assets must already be
installed before the competition starts.

```bash
cd /workspace/students/rock-climbing-workshop
./perception_submit.sh --check
```

After downloading the competition images, run:

```bash
./perception_submit.sh /path/to/release.zip --out outputs/perception/tomorrow
```

Use a new output directory for each attempt. The input can be one ZIP, several
ZIP arguments, an image directory, or individual stereo images. For example:

```bash
./perception_submit.sh /path/to/release/*.zip --out outputs/perception/tomorrow
```

**Upload `outputs/perception/tomorrow/submission.zip` through the requested
Discord private chat.** It contains only the JSON files. Individual files are
also in `outputs/perception/tomorrow/json/`. A sample image named
`real_zero_volumes_cam1_0001.png` produces `real_zero_volumes_0001.json`.

The default computation budget is **540 seconds**, leaving approximately one
minute for upload. Four CPU image fits run concurrently. Native recovery then
searches for a complete generator scene. Cases that fail verification or have
paint-mask loss above 0.30 receive GPU CAD fitting and another recovery pass.
The command writes an
initial ZIP, replaces each prediction as it finishes, and refreshes the ZIP
during refinement. The initial files for unfinished images are explicit empty
predictions. Completed estimates survive interruption or a refinement timeout.
The command does not send anything to Discord.

If downloading consumes part of the window, shorten the run, for example with
`--seconds 420` for seven minutes of computation. The budget starts when the
command starts; it does not include the preceding download or the later upload.
Submit the release ZIPs together so the seed scan is shared across images.

Inspect the final terminal summary. `manifest.json`, outside the upload ZIP,
lists completed estimates, empty fallbacks, uncertain cases, errors and elapsed
time. `complete` describes output coverage; a verified native reconstruction
remains conditional on the declared generator. `complete_with_warnings` calls
out errors or unresolved uncertainty while retaining the available JSON. A partial
run returns exit code 2. The ZIP remains available, but empty predictions will
lose points. A busy GPU causes an immediate preflight failure; the command does
not spend the submission window waiting for another workload.

## Input assumptions

**[Source]** The supplied competition script at
[`11295d1`](https://github.com/jackybaltes/rcr-perception/tree/11295d1091a8e038164e93dc8483139e821b6cab)
uses 30 holds, zero or three volumes, a 3×1.4 m wall, 64 mm stereo baseline,
0.7 m camera height, 1 m camera distance and 74° vertical field of view. The
sample stereo images have two 699×318 panels. Calibration is calculated from
these declared settings, independently of reference hold positions.

The command defaults to those settings. If the organizer changes them, the CLI
provides `--native-hold-count`, `--wall-width`, `--wall-height`, `--camera-height`,
`--distance`, `--baseline` and `--fov`. The current volume prior is zero or three;
other volume distributions require changing that prior before using recovery.

The supplied “real” images are the public renderer's matte, textured appearance
mode. They are not physical-camera photographs. The estimator identifies the
appearance from RGB, extracts coloured hold silhouettes in the matte mode, and
fits the public CAD geometry. It then searches the public generator and checks
candidate geometry against the observed holds. Textured volumes are recovered
through the generator; an exact flat-colour volume mask is not assumed.

The scorer and sampled geometry agree with the previously pinned `b8c607e`
runtime; the new `11295d1` source changes the rendered appearance. The existing
pinned generator is retained and was checked against all 40 supplied scene
documents. Those label-based compatibility checks are separate from inference.

Reference JSON files are ignored even if present in an input archive. No seed
or answer cache is supplied to inference. Native recovery assumes an unedited
scene from this distribution. For manually edited scenes or Cube holds, use the
general image estimator described in [the operating guide](README.md).

## Prepare the machine before release

The repository launcher selects `.venv-perception` or `.perception-core`.
For a new machine, install the private runtime in advance:

```bash
python3.12 scripts/perception/setup.py --benchmark
./perception_submit.sh --check
```

This needs Linux, Python 3.12, EGL/OpenGL, Node.js 20+, npm, CUDA development tools
and an NVIDIA GPU. The runtime cache defaults to `~/.cache/rcr-perception`; the
existing workspace's `.perception` cache is detected automatically. If using a
custom cache, set `RCR_PERCEPTION_HOME` consistently for setup and inference.
The competition run performs no downloads or package installations.
On the shared workspace, use its common GPU lock when launching:

```bash
export RCR_GPU_LOCK=/workspace/development/outputs/perception/gpu_experiment.lock
```

`--check` validates imports, mesh checksums, native resources and GPU capacity.
The timed rehearsals below test estimation accuracy. Keep the GPU available at
release time, and allow time to download the images and upload the resulting ZIP.

## Earlier full-fit measurements

These measurements precede the grouped, bounded-anchor changes above. Their
source hashes and outputs are preserved separately.

The final launcher processed all four supplied archives together after every
reference JSON had been removed. The actual upload ZIP was evaluated with the
unmodified scoring functions from source `11295d1`. Lower scores are better;
zero means that every scored position, orientation and type matches.

| Supplied group | Scenes | Hold score | Score with volumes |
|---|---:|---:|---:|
| Synthetic, no volumes | 10 | 0 | 0 |
| Synthetic, three volumes | 10 | 0 | 0 |
| Matte rendering, no volumes | 10 | 0 | 0 |
| Matte rendering, three volumes | 10 | 0 | 0 |

The complete 40-image run took **242.31 seconds** on the workspace RTX 5000 Ada
with four workers, including preflight, image preparation and packaging. It
recovered all 1,200 holds and 60 volumes and produced all 40 correctly named
JSON files without errors. Eleven cases received GPU refinement. These supplied
samples informed development, so this is a regression result.

A separate frozen test used **12 previously unseen matte-rendered scenes**, six
with no volumes and six with three. It scored **zero in both modes**, recovering
360 holds and 18 volumes in **136.78 seconds**. Five cases received GPU fitting.
The reference labels were first read after inference completed. This is held-out
validation of the same generator and camera settings; future images remain
unseen, and the sample count does not establish a universal success rate.

### Changes supported by the experiments

| Method on the same 40 supplied scenes | Summed hold score | Exact scenes |
|---|---:|---:|
| Previous CAD frontend and native recovery | 6,073.65 | 10/40 |
| Matte normalization, GPU CAD, original strict verifier | 1,263.64 | 30/40 |
| CPU CAD proposals and revised verification | 155.32 | 38/40 |
| CPU CAD, recovery and selective GPU refinement | 0 | 40/40 |

The previous detector relied on bright paint and missed most matte holds.
Hue-based components restore useful silhouettes. The red reference square
merges with some red holds, so local appearance subtraction separates small
hold regions from that smooth square. Textured volumes also invalidate the
old flat-colour volume mask. Recovery therefore tries the declared zero and
three volume counts and verifies their rendered holds against both views.

Some correct scenes still have incomplete paint masks because of shading and
the square. The revised verifier permits higher mask loss when at least ten
fitted anchors agree. The coarse CPU fit then made two wrong reconstructions.
Repeating the accurate GPU CAD fit for uncertain cases corrected both. The
0.30 fallback threshold was chosen using these supplied examples; it reruns
additional already-correct cases to protect against those failures.

The selected command combines those changes and preserves a verified output
while a more expensive fit is running. It estimates the full native scene,
including hidden holds, under the generator assumption. Arbitrary edited scenes
and physical photographs require the general perception workflow and separate
validation.

The final delivery code also passed an eight-second-budget test. It returned a
partial result in 5.50 seconds, with all 40 correctly named empty fallback JSONs
and a valid ZIP. No inference was launched at that budget. Progress messages,
confidence reporting and archive metadata handling were added after the timed
accuracy runs; estimation and candidate selection stayed unchanged.

Per-scene scores, output hashes and the frozen inference source hashes are in
[`submission_results.json`](submission_results.json). The unchanged earlier
experiments are documented in [SCORED_BENCHMARK.md](SCORED_BENCHMARK.md).
