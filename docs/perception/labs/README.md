# Small geometry references

These optional CPU scripts isolate geometry used in the runtime. They supply
simplified observations and do not reproduce the full benchmark. Use the
[operational runtime](../README.md) for challenge images and cameras.

From `docs/perception`:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r labs/requirements-lock.txt
.venv/bin/python labs/run_all.py
```

| Script | Input and output |
|---|---|
| `stereo.py` | Corresponding image points to metric 3D, with disparity validation |
| `cad_fit.py` | A polygon silhouette to planar position/yaw through multiple starts |
| `keypoint_pose.py` | Supplied noisy keypoints and toy joints to a camera-frame pose |
| `contact_registration.py` | Tentative point correspondences to robust rigid alignment |
| `real_stereo.py` | The saved stereo pair to a color/stereo baseline |

Each command supports `--help` and writes JSON and a figure. `run_all.py`
checks exact-data recovery, invalid disparity, unobservable registration and
candidate selection. Saved reference outputs are in `../reference_outputs/`.
The PnP example's camera frame is distinct from the runtime wall frame. The
planar CAD example has three unknowns; the selected runtime fits full 6D CAD.
