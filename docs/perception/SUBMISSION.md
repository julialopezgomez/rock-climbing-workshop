# Ten-minute perception submission

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

## Measured results

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
