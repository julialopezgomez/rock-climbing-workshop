# Known-working Microban replay snapshot

Recorded on 2026-11-01 and preserved on Ruchira's Microban fork.

- Microban fork: `https://github.com/rayruchira/microban.git`
- Microban branch: `ruchira-replay-gains`
- Microban commit: `ea61e7f2606337dd623bb39968c6cdb1024d0452`
- Commit subject: `Allow higher Kp values in replay`
- Checkout location: `third_party/microban`
- `src/moves/replay.py` permits `kp` through `POSITION_GAIN_MAX` (`16383`) instead of limiting it to `1000`.

The local-tools replay runtime also had these working settings, which are outside this Git repository:

- Speeds: `0.5, 1, 1.25, 1.5, 2, 2.5, 3`
- Gain ranges: `kp`, `ki`, and `kd` accept `0..16383`
- Defaults: `kp=1000`, `ki=100`, `kd=10`

To restore the Microban state:

```bash
cd /Users/ruchiraray/Documents/DAIR/rock-climbing-workshop/third_party/microban
git fetch origin
git switch ruchira-replay-gains
```

The parent workshop branch `ruchira/replay-updates` records this exact submodule commit and fork URL.
