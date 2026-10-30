# Perception: run, reproduce and connect cameras

[Ours] This branch contains the selected estimator implementations, trained
Microban landmark checkpoints, native build setup, replay data and camera
entrypoints. The runtime is under `src/rcr/perception/challenge/`; it does not
load code, predictions or CAD from the original experiment workspace.

- [Operational report (PDF)](perception_report.pdf)
- [Commands and input/output contracts](RUNBOOK.md)
- [Implementation map](IMPLEMENTATIONS.md)
- [Team frame/type adapters](TEAM_INTEGRATION.md)
- [Recorded verification](verification.json)

## Install

Use Linux x86-64, Python 3.12 and an NVIDIA GPU/runtime compatible with CUDA 12.8
for the complete learned pipeline. The tested card is RTX 5000 Ada (32 GB).
From the repository root:

```bash
git submodule update --init
python3.12 scripts/perception/setup.py --robot --cuda
.venv-perception/bin/python scripts/perception/run.py doctor --robot
```

Setup creates `.venv-perception` and a separate cache under
`~/.cache/rcr-perception`. It installs pinned Python packages, downloads the
pinned public CAD, builds M3T and the source-recovery binaries, and installs the
pinned climbing model. It preserves the team's base environment and submodules.
The native and CUDA dependencies require several GB of disk/network transfer.
Use `--cuda-arch` for a GPU other than the tested Ada target; see the runbook.

For RGB capture and the measured-marker pose baseline without the learned
robot stack:

```bash
python3.12 scripts/perception/setup.py --cameras
```

A [Docker recipe](../../scripts/perception/Dockerfile) is also provided:

```bash
docker build -f scripts/perception/Dockerfile -t rcr-perception .
docker run --rm --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -v "$PWD:/app" rcr-perception doctor --robot
```

The runtime/build stages were tested directly in isolated environments. The
Docker recipe itself has not been built here because a Docker daemon is not
available in this container. USB camera use additionally needs the host's device
permissions; the runbook gives the device-mapping command.

## Reproduce the selected methods

```bash
.venv-perception/bin/python scripts/perception/run.py reproduce \
  --out outputs/perception/holds

.venv-perception/bin/python scripts/perception/run.py replay \
  --dataset data/perception/robot --profile max \
  --out outputs/perception/robot

.venv-perception/bin/python scripts/perception/verify_replays.py \
  --dataset data/perception/robot \
  --predictions outputs/perception/robot \
  --out outputs/perception/robot/verification.json
```

Five original stereo examples reproduce 50/50 source hold dictionaries exactly.
The selected robot estimator passes all five root and endpoint checks on the
included examples. These are saved-case regression checks, distinct from the
historical 200-scene hold/robot evaluations in the report.

`reproduce` uses RGB/CAD fitting followed by verified public-generator recovery.
It assumes untouched native10-Microban-hold scenes and supplied calibration.
For independently estimated CAD poses use `holds --cad-only`. Generator
recovery is not a physical-camera method.

## Connect cameras

```bash
.venv-perception/bin/python scripts/perception/run.py capture --list
```

Follow [RUNBOOK.md](RUNBOOK.md) to capture color pairs, calibrate a printed board,
record measured joint telemetry, and run persistent estimation. The physical
baseline uses a measured trunk marker; the learned/markerless profiles use the
included trained networks and CAD. Camera intrinsics, wall marker geometry and
marker-to-trunk mounting must be measured for the real setup.

No RealSense devices are visible in the development container. Color-only SDK
capture, timing, geometry and calibration paths have software tests; physical
USB capture, exposure synchronization and real-camera accuracy remain to be
measured. Low reported residuals alone do not establish contact acquisition.
