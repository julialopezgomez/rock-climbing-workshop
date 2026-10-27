# link/raspberry_pi

The Pi's side of the link. See [../README.md](../README.md).

| File | What |
|---|---|
| [robot_link.py](robot_link.py) | `RobotLink`: the trajectory, start/stop and pose in; state out. Every method is non-blocking, for the 50 Hz loop |
| [\_\_main\_\_.py](__main__.py) | the receiver: `python -m rcr.link.raspberry_pi`. It saves each trajectory, prints start and stop, and sends state back. It drives no motors |

## Putting it on the Pi
From the laptop, in this repository:
```bash
uv run scripts/link_pi.py deploy user@172.20.10.8        # copies ~/rcr/src/rcr/... (asks for the password)
uv run scripts/link_pi.py deploy user@172.20.10.8 --run  # ... and starts the receiver
```

On the Pi, by hand:
```bash
cd ~/rcr && PYTHONPATH=src python3 -m rcr.link.raspberry_pi
#   --save FILE  where trajectories go (default ~/rcr/trajectory.json)
#   --rate HZ    state sent back (default 50)
#   --port N     TCP port (default 5005; UDP pose port 5006)
```

**What it needs:** our `rcr/__init__.py`, `types.py`, `io.py` and `link/`, on Python 3.11 or newer. It uses the stdlib
only: nothing to install.

**Copying by hand instead of `deploy`:** keep the folder layout, `~/rcr/src/rcr/link/raspberry_pi/...`. These files
import `rcr.types`, `rcr.io` and `rcr.link.protocol`, so they need the rest of `rcr` next to them.
