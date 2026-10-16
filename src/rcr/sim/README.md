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
  - a 0.800 × 1.400 × 0.010 m board, vertical;
  - each hold a placeholder shape inside its Ø0.060 × 0.100 m envelope, at `Hold.position`.
- **ArUco 1–4 + 2 cameras** [Rules §5.1–5.2]:
  - DICT_4X4_50 textures at the corner positions in INTERFACES.md;
  - two cameras, one on each side, both seeing the whole wall (placement is a guess until the official sim);
  - rendered RGB images feed `perception` later.
- **Microban with placeholder climbing limbs:**
  - start from Microban's MJCF (`src/model/mjcf/`) [Microban];
  - the climbing limbs are not published, so we add simple hooks/grippers and document their joint names.
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
- Whether `Sim` also exposes rendered images (`images()`) for perception. Add it when perception needs it.
