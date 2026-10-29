# sim

**Purpose:** a backend with the same role as the robot: it receives joint targets and returns robot state.

| | |
|---|---|
| Input | `WallModel` (at construction); `JointTargets` every tick |
| Output | `RobotState` every tick (20 ms of sim time per `step`) |
| Runs on | laptop. MuJoCo is in the `sim` dependency group and is never imported by `controller/` or `link/` |

- The controller must not know which backend it runs on. The loop (`run_pipeline.py --backend sim`) calls
  `state = sim.step(controller.step(state))`.
- Microban does the same thing at the motor-bus level: its Scheduler and Moves run unchanged on `RobotController` or
  `MuJoCoController` [Microban].

## Our emulation of the official simulator
The official simulator is **not released yet** (website: "T.B.D."). It will be MuJoCo (CPU), with models of the robot, walls
and holds, and robots relying on rendered camera images [Rules §7]. Until then, we build our own emulation from the rules:

- **Wall + holds from the `WallModel`** [Rules §3–4]:
  - a 0.800 × 1.400 × 0.010 m plywood board, vertical, with 52 × 93 holes (17.5 mm side margin) [Rules v1.1 Figure 2];
  - each hold the organisers' hold mesh (`data/holds/hold.stl`, Figure 3), bolted two pitches apart, at `Hold.position`.
- **ArUco 1–4 + 2 cameras** [Rules §5.1–5.2]:
  - DICT_4X4_50 textures at the corner positions in INTERFACES.md;
  - two cameras, one on each side, both seeing the whole wall (placement is a guess until the official sim);
  - rendered RGB images feed `perception` later.
- **Microban with its climbing limbs:**
  - the climbing model's MJCF (`microban_model`, `robot_climb.xml`) [Microban];
  - each limb's ball marked by a body named after the limb and a site `<limb>_tip` (`rcr.robot.hooks`);
  - the model lets only the climbing forearms' and feet's collision pieces collide with the rest of the robot.
- **A swappable grip model:** how a limb holds a hold is chosen in one place, e.g. v0 "weld when close, release on command",
  later friction-only contact, so it can be changed when the real holds and limbs are known.
- **Servo limits and delays** [Microban sim]:
  - BAM XL330 servo model (`better-actuator-models`) with the 1.75 A current limit, i.e. ≈ 0.64 N·m;
  - kp as on the robot;
  - delays as in Microban's `sim_main.py` defaults: actuation 2 physics steps (10 ms), velocity 1 tick, gyro 3 ticks, orientation 4 ticks;
  - 5 ms physics step, 50 Hz control.
- **Scoring emulation** [Rules §8]: highest hold touched, and a timer until the last hold.
- **Optional:** a safety-rope catch, so falls end the run without blowing up the sim.

**Note** [Microban]: Microban's own `MuJoCoController` opens a viewer (`launch_passive`) and implements the bus protocol,
not our `step()`. We may reuse its model loading and servo model, but `Sim` needs a headless mode for batch runs.

## When the official sim is released
Only the inside of `Sim` changes: the same `reset()` and `step(targets) -> RobotState`. Planner, optimiser and controller
are untouched. If its interface differs (e.g. it owns the loop), a thin adapter lives here.

## Open decisions
- Reuse Microban's sim code vs write our own (depends on D1).
- The grip model v0.
- The camera placement.

## v0, implemented (prep fork)

Built with `MjSpec` from the Microban model, so the clone stays untouched and the generated XML
(`build/scene.xml`, plus `build/robot_climb.xml` for placo) is a one-path swap for the official sim.

- **Wall** — the M3 grid (52 × 93 holes, 17.5 mm side and 10 mm top/bottom margins [Rules v1.1 Figure 2];
  the §3 text still says 10 mm all round) is a generated texture, not 4836 hole geoms; the board uses stiff contact
  (`solref 0.002`), because MuJoCo's default lets the robot sink ~10 mm into it.
- **Holds** — the organisers' own mesh, `data/holds/hold.stl` (from their rules repo, the file Figure 3 is drawn
  from), placed by `CupHold` in `geometry.py` with a translation only (its axes are the wall frame's), with the
  origin at the socket centre, **26 mm out and 14 mm below** the bolt midpoint (`Hold.position`) [Rules v1.1 Figure 3].
  A 3 mm back plate is bolted through holes **two pitches (0.030 m) apart**; the bucket has a **spherical socket of
  12.7 mm radius** that takes the robot's ball end. A ball rests at the bottom of the socket (`grasp_point`): on the
  socket centre if it is as big as the socket, lower by the clearance if it is smaller. Ours is 10 mm until the limbs
  are published, so it sits 2.7 mm below the centre. Holds sit on `contype/conaffinity 2` and never collide with the
  robot: the weld *is* the grip. The colour stays the orange of the organisers' render; green in Figure 3 is not a rule.
  **Rules v1.1, read together:** §4 says "side by side, 15 mm apart" (the grid pitch) and Figure 3 says "two pitches
  apart (30 mm)"; we read them as bolts in holes (i − 1, j) and (i + 1, j), so `Hold.position` is hole (i, j). Hold x
  positions are unchanged from the draft1 version of this sim.
- **Layout** — `levels.organisers_wall()` reproduces their wall: rows every 3 or 4 pitches
  alternating (≈0.0525 m), one to four cups per row, scattered across columns 15–36
  (x ≈ 0.24–0.56 m). Rows below j≈22 were behind the robot in that frame and follow the same
  spacing and density. `levels.generate(1|2|3)` still gives the rulebook levels.
- **Limbs** — `LimbTip`, read side-on from their video. Hands: past the end of the forearm, a
  finger bent towards the wall *with the arm raised* (the climbing pose), ending in a grey ball
  (frames 880–940). Feet: the same hook on the stock foot plate, past the toe and bent down out of
  the sole, so the ball drops into the cup and the plate stays above the rim [Ours].
  The ball is what sits in a cup; the sim's tip site is its centre. Dimensions [Ours], read against
  the forearm, until the climbing limbs are published [Rules §2.5]. The hooks are defined once, in
  `rcr.robot.hooks` (shared with the optimiser); `LimbTip` and `LIMB_PARENTS` take them from there.
- **Actuators** — Microban's are `<motor>`, i.e. **torque**, not position servos; `data.ctrl` is N·m.
  The scene clamps them to **0.64 N·m** (1.75 A × 0.366 N·m/A), which the stock XML leaves unset.
  `ServoGains` then runs the P loop the real servos run.

### Grip: `Sim.expect(stance)`
Our contact model is a weld, so something must decide when a ball leaves a cup. **The caller says, the
sim does not guess.** Inference was tried and does not work: constraint force is the wrong signal (four
welds carry tens of newtons internally, a servo pulls with about six), and so is the commanded pose
(it disagrees with the welded contacts by more than the signal). So whoever runs the loop calls
`expect(stance)` when the plan moves on; capture still requires the ball to actually arrive within
20 mm of the cup, slowly.

### Measured
- Four-limb hang: all limbs held, trunk drifts ~9 mm over 5 s, grip forces sum to the robot's weight
  (8.27 N), peak torque ~20 % of the limit at rest.
- All four ArUco markers detect from both cameras.

### Setup
```bash
git submodule update --init               # Microban, in third_party/microban (once)
uv sync --group sim                       # mujoco + opencv
```
`MICROBAN_XML` (or `MICROBAN_URDF`) points it at another Microban checkout instead. Without Microban, `run_pipeline.py --check` reports the sim as `skip` and `tests/test_sim.py` skips (as on CI).

### Using it
```python
sim = Sim(wall)  # builds build/scene.xml (+ build/robot_climb.xml for placo)
state = sim.reset()  # hanging on wall.start, settled, time = 0
sim.expect(next_stance)  # when the plan moves on: the limb that changes lets go
state = sim.step(JointTargets(...))  # one 20 ms tick; absent joints go to Microban's NEUTRAL_POSE
sim.score  # Score(height, top_time) [Rules §8]
sim.images()  # {"cam_left": rgb, "cam_right": rgb}
```
- `reset()` solves a start pose with a small IK (`start.py`) that puts each ball on its cup, the trunk
  upright and facing the wall [Ours]. It raises if `wall.start` is unreachable.
- `reset(stance)` hangs the robot on any `Stance` instead (free limbs = `None`), or raises if a ball
  cannot reach its cup. `tests/test_sim.py` has a `STANCES` list to edit and `verify()`, which also
  rejects a pose that hits the wall or itself (`wall_penetration`, `self_penetration`);
  `uv run mjpython tests/test_sim.py stances` applies them one by one in the viewer.
- `RobotState`: the IMU comes from Microban's IMU site sensors (`orientation`, `angular-velocity`);
  `base_pose` is the trunk, i.e. what the cameras would give.
- `step()` rejects joint names it does not know. The servo PD runs every 2 ms physics step.
- **`tests/test_sim.py` is the one place to try it.** `uv run mjpython tests/test_sim.py [l1|l2|l3|organisers|file]`
  opens the robot on a wall in the viewer (macOS needs `mjpython`); `uv run pytest tests/test_sim.py` runs
  the checks. Its `run()` loop is where the controller plugs in later (`targets = controller.step(state)`).

**Wall frame:** y points *into* the wall, so the robot and cameras are at y < 0. The earlier
"y out of the wall" with x right and z up was left-handed, which MuJoCo rendered mirrored (marker 3 to
the right of marker 4, and `right_hand` on the robot's left).

### Worth knowing before writing the controller
Playing joint angles open-loop will not place a limb accurately: the robot is a floating base in a
closed chain, and with 0.64 N·m servos each limb sags ~15 mm, so the base ends up away from wherever
the kinematics assumed. `RobotState.base_pose` is there for closing that loop.

### Open
- Not modelled yet: Microban's servo model (BAM) and its actuation/sensor delays (P2.3), a safety-rope
  catch. Physics runs at 2 ms, not Microban's 5 ms.
- Four 3-DOF welds over-constrain the robot, so reported grip forces include large internal
  components. Only the vertical sum is trustworthy.
- A friction backend (ball resting in a cup, no weld) needs real limb geometry; it would also remove
  the `contype 2` hack and `expect()`.
