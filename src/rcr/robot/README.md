# robot

**Purpose:** the real-robot backend, which is Microban ([Rhoban/microban](https://github.com/Rhoban/microban)). This
README explains how our code meets Microban. The only code here is what we add to Microban's robot.

## What we add to Microban: the climbing hooks [Ours]
Microban as published has no climbing limbs, and the organisers will publish theirs [Rules §2.5]. Until then each hand
and foot carries a hook: a short bent finger, fixed to a Microban link, ending in a ball that rests in a hold's socket.
They are defined once, in [`hooks.py`](hooks.py), and the optimiser's model and the sim both take them from there.

When the real limbs are known, change [`hooks.py`](hooks.py) and nothing else:
- **the shape:** `HOOKS`, which link each hook is fixed to, where on it, and its finger segments;
- **the size:** `BALL_RADIUS`, `FINGER_RADIUS`, `BALL_INSET`;
- **the mass:** `MASS`, `SEGMENT_MASS_SHARE`.

Then run the tests, which check that the optimiser and the sim agree, and solve the plan again: the motions depend on the
hooks. The contact point on a hold follows the ball's radius. Two other guesses also depend on the real limbs and holds:
the hold's push direction and friction (`normals`, `mu` in `rcr.optimiser.transition.TransitionConfig`), and the hold's
socket (`SOCKET_RADIUS` in `rcr.optimiser.collision`, `CupHold` in `rcr.sim`).

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
# Microban's Move; ~30 lines, location depends on D1
class ClimbMove(Move):
    def step(self, obs, command):
        # Microban's RobotState -> rcr.types.RobotState
        state = to_rcr_state(obs.robot_state, self.t0, self.link.pose())
        command.target_angles.update(self.controller.step(state).positions)
        self.link.send_state(state)  # non-blocking


# start/stop from the laptop instead of the keyboard
class LinkInputSource(InputSource):
    def read(self):
        return UserInput(active_moves={"climb"} if self.link.run_requested() else set())
```

`to_rcr_state` maps the fields as follows:
- `motor_positions` → `joint_positions`;
- `motor_velocities` → `joint_velocities`;
- `body_quat` → `imu_quat`;
- `gyro` → `gyro`;
- `time_s` − t0 → `time`.

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
**Chosen: the submodule** `third_party/microban`, pinned to `d0a128d` (27 Oct). Every clone reads the same files,
nothing of Microban is copied into our history, and the climbing variant will be a pin bump. The optimiser and the sim
find it through `rcr/robot/microban.py`; `MICROBAN_URDF` or `MICROBAN_XML` still points them at another checkout.
Still open: where `ClimbMove` and our entry point sit on the Pi (next to Microban on its `PYTHONPATH`, see below).

The options, as compared before choosing:

Microban's software is **GPL-3.0**, and its `pyproject.toml` has **no build system**, so it is not pip-installable. In
every option, it is used via `PYTHONPATH=src`. The organisers' climbing variant will probably come as Microban changes, so
the choice must make switching to it easy.

| Option | For | Against |
|---|---|---|
| **Submodule** (`third_party/microban`, pinned commit) | exact version; upstream/climbing-variant updates are a commit bump; no Microban code in our history; the sim gets its MJCF and servo model | submodule friction (`clone --recursive`, bumps); our `ClimbMove` and entry point live in our repo and must be put on the Pi's `PYTHONPATH` next to Microban |
| **Fork** (our fork of Rhoban/microban) | `ClimbMove` + entry point sit next to Microban's moves; Microban's `make run` / `make sim` workflow keeps working; upstream PRs possible | two repos to keep in step; merging upstream / the climbing variant into the fork |
| **Copy** (vendor `src/` into our repo) | one repo; edit anything | loses upstream fixes; our repo becomes a GPL-3.0 combined work if we publish it (we have no licence yet); harder to switch to the climbing variant |

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
3. On the laptop: plan → optimise → trajectory. Dry-run it in our sim and confirm that it climbs.

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
