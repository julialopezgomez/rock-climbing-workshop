# link

**Purpose:** the **only** thing that crosses laptop ↔ robot during a run.

| Direction | What | When | Transport |
|---|---|---|---|
| laptop → robot | `Trajectory` | before the climb (and again if the laptop replans) | TCP, answered |
| laptop → robot | start / stop | once each | TCP, answered |
| laptop → robot | robot `Pose` from the cameras (optional) | live, if the controller uses it | UDP |
| robot → laptop | `RobotState` | live, for monitoring and recording | UDP |

| | |
|---|---|
| Runs on | both: `LaptopLink` on the laptop ([laptop.py](laptop.py)); `RobotLink` on the Pi ([raspberry_pi/](raspberry_pi/), Pi-safe, every method non-blocking because it is called from the 50 Hz loop) |
| Wire format | [protocol.py](protocol.py), shared by both sides |

## Use it

**The setup:** the laptop and the Pi on the same Wi-Fi.

1. **Put our code on the Pi, once and after every change:**
   ```bash
   uv run scripts/link_pi.py deploy pi@172.20.10.8       # rsync over ssh: asks for the Pi's password
   ```
   This copies `rcr/{__init__,types,io}.py`, `controller/` and `link/` to `~/rcr/src` on the Pi.

2. **Start the receiver on the Pi:**
   ```bash
   cd ~/rcr && PYTHONPATH=src python3 -m rcr.link.raspberry_pi     # on the Pi (ssh pi@172.20.10.8)
   ```
   Or `deploy ... --run` does steps 1 and 2 in one command, using Microban's Python if it is there.
   - It prints `listening on <address>:5005`.
   - It saves each trajectory to `~/rcr/trajectory.json` and prints start and stop.
   - It sends a state back 50 times a second. **It drives no motors**: until our Move exists, it stands in for it.

3. **From the laptop:**
   ```bash
   uv run scripts/link_pi.py check --pi 172.20.10.8      # evaluate the link
   uv run scripts/link_pi.py send --pi 172.20.10.8       # send data/examples/trajectory.json (or give a file)
   uv run scripts/link_pi.py start --pi 172.20.10.8      # / stop
   ```
   Set `RCR_PI=172.20.10.8` once and `--pi` can be left out.

   **From Python, one call:**
   ```python
   from rcr import link

   link.send_trajectory("data/examples/trajectory.json", "172.20.10.8")  # or a Trajectory
   ```
   It returns once the Pi has the whole trajectory and has checked it. Otherwise it raises `LinkError`, saying why.

   **For a whole run:**
   ```python
   with link.LaptopLink("172.20.10.8") as robot:
       robot.send_trajectory(trajectory)
       robot.start()
       state = robot.robot_state()  # the newest state from the robot, or None
       robot.stop()
   ```

**On the robot side** (our Move, later):
```python
from rcr.link import RobotLink

robot_link = RobotLink()  # listens on 5005 (TCP) and 5006 (UDP)
# each 50 Hz tick, none of these waits:
robot_link.trajectory()  # the newest trajectory, or None
robot_link.run_requested()  # between the laptop's start and stop
robot_link.pose()  # the newest camera pose, or None
robot_link.send_state(state)  # to the laptop, if one connected
```

## The protocol (D5) [Ours]
- **Commands over TCP:** the trajectory, start and stop.
  - The laptop connects to the Pi's port 5005. The robot answers every command, accepting or refusing it with a reason.
  - So `send_trajectory` returning means the trajectory arrived whole, in order, and passed the robot's checks:
    times increase, and there is one row per time with one value per joint.
  - A message is a 4-byte length, then JSON. The trajectory is `rcr.io`'s file format, schema included.
  - It is ~1.1 MB with the replay fields, sent once.
  - TCP is used because UDP would lose or reorder pieces of a trajectory, and a lost stop is not acceptable.
- **Live streams over UDP:** the state (robot → laptop) and the camera pose (laptop → robot).
  - Only the newest one matters, so a lost one is just replaced by the next. The sender never waits.
  - Each datagram carries a sequence number, so the laptop counts losses (`states_lost`).
  - Each also carries a session: a restarted Pi starts a new one.
- **The laptop speaks first, also over UDP:**
  - it sends a small "subscribe" to the Pi's UDP port when it connects, then every second;
  - the Pi sends the state back to wherever that came from, from the same port, so it arrives as a reply;
  - so the laptop needs no open port: the macOS firewall, which blocks incoming connections for our Python, lets it
    through, and so does a router that translates addresses;
  - found on the real Pi: before this, the state was blocked.
- **Ports:** TCP 5005 and UDP 5006 on the Pi. The laptop listens on nothing.
- **Safety:**
  - start is refused before a trajectory has arrived;
  - a new trajectory is refused while running, so stop first;
  - a bad message is refused, never fatal;
  - the laptop reconnects once by itself if the Pi restarted.
- **Pi-safe:** `socket`, `threading`, `json` and `struct` from the stdlib, plus `rcr.types` and `rcr.io`. It runs on
  Python 3.9 or newer: our test Pi has 3.9 (Debian 11), Microban's image 3.12. That is why `types.py` and `io.py`
  start with `from __future__ import annotations`, and the Pi's files avoid `zip(strict=)` and catch `socket.timeout`.
- **Nothing time-critical crosses the link.** The 50 Hz loop lives on the robot. `RobotLink`'s methods only read
  what its background threads received, so a late or lost message never stalls a tick.
- **Tests:** `tests/test_link.py` runs both ends over loopback (127.0.0.1): the protocol, the receiver and the `check`
  script. `run_pipeline.py --check` sends the example trajectory through the real link, also over loopback.

## Measured
| Where | Round trip (median) | Trajectory (1.15 MB) | State back |
|---|---|---|---|
| Loopback, one MacBook | 0.2 ms | 0.02 s | 50 per second, 0 lost |
| MacBook ↔ Raspberry Pi (Debian 11, Python 3.9) over a phone hotspot, 27 Oct | 4.8 ms (4.4–8.4) | 0.35 s | 51 per second, 0 of 507 lost |

## Design points
- **Microban provides SSH/rsync before a run, but nothing during it** [Microban Makefile]. So the run-time link is ours.
- **On the Pi it plugs in as a Microban input source and/or inside our Move:**
  - start/stop becomes an `InputSource` that sets `active_moves`;
  - the trajectory and pose are read inside the Move.
  - See [../robot/README.md](../robot/README.md).
- **Autonomy:** no teleoperation [Rules §2.4]. The laptop sends a plan and start/stop, never live motion commands.
- **Allowed links:** any wireless link or the Ethernet tether along the safety rope is allowed [Rules §2.4, §5.3].
- **Networks that block it:** many venue networks block traffic between devices ("client isolation"). A phone hotspot
  works (172.20.10.x is an iPhone's), as does our own 2.4 GHz router, the robot README's recommendation.
- **macOS firewall:** nothing to allow; the state comes back as replies to the laptop's subscribes.

## Still open
- What happens on link loss mid-climb: continue the trajectory, or stop? The robot keeps `run_requested()` as it
  was, and the Move decides. That decision is for the controller and robot (D4 / P3).
- The state rate during a real climb: the receiver sends 50 per second, but the Move may send fewer.
- USB-Ethernet as a backup: the same code works over it (the address changes).
