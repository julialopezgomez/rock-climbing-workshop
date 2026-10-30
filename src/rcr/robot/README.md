# robot

**Purpose:** the real-robot backend, which is Microban ([Rhoban/microban](https://github.com/Rhoban/microban)). This
README explains how our code meets Microban. The only code here is what we add to Microban's robot.

## The robot model and its climbing limbs [Microban]
- **Model:** the `microban_model` package, climbing variant (`robot_climb.urdf` / `robot_climb.xml`), from **our fork**
  ([MohammadrezaDindarloo/microban_model](https://github.com/MohammadrezaDindarloo/microban_model), branch
  `cut-foot`), pinned in `pyproject.toml`. It is Rhoban's 8e42ef7 (the organisers' simulator's and Microban `dev_v2`'s
  version) with only the climbing feet changed, as on our real robot: the end of each foot's bottom plate cut off, the
  drawn foot, its collision pieces (51, CoACD) and its mass (29.98 → 25.74 g) updated in the MJCF and the URDF
  [Ours]. `override-dependencies` in `pyproject.toml` makes the organisers' simulator load it too (it asks for
  Rhoban's), so their sim, the optimiser and the robot have the same feet. [`microban.py`](microban.py) finds it.
- **Limbs:** the climbing forearms and feet each end in a **12.5 mm ball** that sits in a hold's socket.
  [`hooks.py`](hooks.py) reads each ball's link, centre and radius from the model files (URDF `<limb>_ball_frame`,
  MJCF `<limb>_ball_coll`); nothing is typed in. The optimiser takes them from there; a test checks they are where
  the organisers' sim has them.
- **Ours [Ours]:** only `hooks.NECK`, the direction the forearm or foot comes into each ball (measured on the model),
  a hint for the optimiser's first solve. And `observation.py`, which converts, adds nothing.

## What Microban gives us [Microban]
- **Hardware:**
  - Raspberry Pi Zero 2W: weak CPU, **2.4 GHz-only Wi-Fi**, **no Ethernet port**;
  - 19 Dynamixel XL330 servos (6 per leg, 3 per arm, head);
  - BMI088 IMU;
  - ≈ 0.82 kg.
- **`make run`:** rsyncs the repo to the Pi over SSH, then runs `src/main.py` there. That builds a `Scheduler` at 50 Hz with a
  set of `Move`s and an `InputSource`, and the loop then repeats each tick:
  - the Observer reads motor positions/velocities and the IMU;
  - the InputSource says which moves are active;
  - the active moves write `command.target_angles` (joint name → rad);
  - the targets are sent to the servos, which run their own P loop.
- **`make sim`:** runs the same Scheduler and Moves in MuJoCo. The backend is swapped at the bus level (`ControllerProtocol`).
- **Pi env:** Python ≥ 3.12 venv with numpy, placo, onnxruntime, rustypot and the IMU library.
- **No run-time data link:** SSH/rsync before a run, and only keyboard/gamepad input during it.

## How our controller becomes a Move (later, P3.2)
We wrap our controller in a thin Move and do **not** rewrite Microban's Move or Scheduler:

```python
# a Microban move (dev_v2: a class with step(obs), docs/moves.md); location depends on D1
from microban_model import JOINTS
from rcr.robot import to_state


class ClimbMove:
    def start(self, obs):
        self.t0 = obs["time"]

    def step(self, obs):
        state = to_state(obs, JOINTS, self.t0, self.link.pose())  # Microban's obs -> RobotState
        self.link.send_state(state)  # non-blocking
        return self.controller.step(state).positions  # {joint: rad}; the others keep their target
```

`to_state` ([observation.py](observation.py)) is the same conversion the sim uses: Microban's observation is the
organisers' simulator's. Start/stop from the laptop instead of the keyboard is still to do.

### Pitfalls to handle in the wrapper (found in Microban's code)
1. **`main.py` ramps all motors to `NEUTRAL_POSE` (standing) at start.** On the wall that is wrong: the start ramp must go
   to the trajectory's first pose. This means we need our own entry point instead of Microban's `main.py`.
2. **`MotorCommand` starts each tick from `NEUTRAL_POSE`.** Joints we do not target are driven to neutral.
3. **Overcurrent safety:**
   - if the estimated total current is ≥ 15 A for 2 ticks, the Scheduler **disables torque on every motor**, and the robot drops onto the rope;
   - the estimate is tuned for walking (`PROXY_KP = KP_RL` = 125, while `main.py` sets kp = 400);
   - recalibrate it for climbing loads; do not just remove it.
4. **Stopping a Move:**
   - the default `on_stop` goes INACTIVE immediately, so targets snap back to neutral;
   - on a wall we need a defined stop, e.g. holding the current pose.
5. **No teleoperation** [Rules §2.4]. On the day, the keyboard/gamepad input sources must not drive moves; `LinkInputSource`
   replaces them.
6. `make sync` rsyncs Microban's repo only, so our files need their own sync (see below).
7. Always `make shutdown` before cutting power, or the SD card can be corrupted [Microban usage.md].

## Where Microban sits (D1): a submodule [Ours]
**Chosen: the submodule** `third_party/microban`, on branch `dev_v2` (29 Oct), what the competition robot runs.
- Every clone gets the same version, and no Microban code (GPL-3.0) enters our history.
- Microban is not pip-installable: it is used via `PYTHONPATH=src`.
- The robot model is no longer in Microban, but in `microban_model` (above).

Still open: where `ClimbMove` and our entry point sit on the Pi (next to Microban on its `PYTHONPATH`, see below).

## What goes to the Pi
| To the Pi | Stays on the laptop |
|---|---|
| `src/rcr/__init__.py`, `types.py`, `io.py` | `perception/`, `planner/`, `optimiser/`, `sim/` |
| `src/rcr/controller/` | `scripts/`, `tests/`, `data/` (the trajectory goes over the link) |
| `src/rcr/link/` (the robot side runs there) | all the docs |
| later: `ClimbMove`, `LinkInputSource`, our entry point (location per D1) | |

- These files need **no new dependencies on the Pi**: they use the stdlib and numpy, which Microban's venv already has.
- They run with Microban's `.venv/bin/python`, with our `src/` added to `PYTHONPATH`.
- Deploy with an rsync of exactly these paths, like Microban's `make sync`: `uv run scripts/link_pi.py deploy user@<pi>`
  does it (to `~/rcr/src` on the Pi). The Pi-safe test in `tests/` keeps these imports honest.

## Physical link
| | Wi-Fi 2.4 GHz | USB-Ethernet adapter (OTG) | USB gadget mode |
|---|---|---|---|
| Pi side | built in; already set up by the image's `network-config` [Microban] | `network-config` already declares `eth0` with DHCP [Microban]; needs an OTG adapter + USB-Ethernet dongle. *Check* that the data micro-USB port is free with the Pollen hat | Ethernet over a USB cable; needs `dwc2`/`g_ether` set up on the Pi (untested) |
| Cable | none | along the safety rope, never load-bearing, with strain relief on the robot [Rules §2.3, §5.3] | same, plus USB length limits |
| Risk | crowded 2.4 GHz band (4 walls, many teams); venue Wi-Fi unknown | a cable dragging on a 0.8 kg climber; extra mass; laptop needs an Ethernet port | setup effort; same cable issues |

**Recommendation [Ours]:**
- Use **Wi-Fi on our own 2.4 GHz access point** (a travel router, or a hotspot forced to 2.4 GHz) rather than the venue
  network, with the robot preconfigured for it in `network-config`.
- Keep a USB-Ethernet adapter as the **tested backup** (P3.4).
- This works because nothing time-critical crosses the link: once started, the robot climbs on its own, and a Wi-Fi drop only costs
  monitoring (if the link-loss policy is "continue").

## Run sequence on the day
**At the team table**
1. Power the robot. Check it is on our access point (`ping`, `ssh`). Deploy our Pi files.
2. Get the wall model for this wall: hand-written from the hold layout (perception later). Check the hold positions.
3. On the laptop: plan → optimise → trajectory. Dry-run it in the organisers' sim and confirm that it climbs.

**At the wall**

4. Place the robot at `wall.start` (bottom of the wall; the ground start is parked).
   - Attach the safety rope to the anchor point.
   - Tie any cable along the rope, with strain relief on the robot.
5. Start our program on the Pi over `ssh`. It enables torque and ramps to the **trajectory's first pose** (pitfall 1), then waits.
   If the robot has to be held in pose while it is placed, do this before step 4.
6. Send the trajectory from the laptop, and check that the robot reports it received it (state back).
7. Send **start**. From here there is no human input [Rules §2.4]. The robot runs at 50 Hz and streams its state back, while the laptop
   records the state and the camera images.
8. At the end (last hold reached, fall, or timeout), send **stop**, or let the robot stop itself. `make stop` disables torque.
9. Before cutting power, run `make shutdown` and wait 10–15 s.
