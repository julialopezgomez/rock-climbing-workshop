# Roadmap

Today is Fri 16 Oct 2026. The event runs Thu 29 Oct – Sun 1 Nov, which leaves **13 days**. Each task below is one GitHub issue,
labelled with its module (or `integration` / `decision`).

| Phase | Dates | Done when |
|---|---|---|
| 1. Modules alone | Fri 16 – Tue 20 Oct | every module's v0 runs alone against `data/examples/`, and `run_pipeline.py --check` shows `ok` for it |
| 2. Connected in sim | Wed 21 – **Fri 23 Oct (gate)** | `run_pipeline.py --backend sim` climbs the L1 example wall end to end in our emulated sim, with no manual help |
| 3. Robot, robustness, L2/L3 | Sat 24 – **Wed 28 Oct (freeze)** | the real-robot path works laptop → Pi → Move; L1 is robust to sim perturbations; L2/L3 climb in sim |
| Event | Thu 29 Oct – Sun 1 Nov | |

## Decisions

Each decision is a `decision` issue. The issue closes with the choice and a one-line reason, recorded in the module README.

| ID | Decision | Module | Decide by | Why then |
|---|---|---|---|---|
| D1 | Where Microban sits: submodule / fork / copy | robot | **Mon 19 Oct** | the sim needs Microban's MJCF and servo model |
| D2 | Planner method | planner | **Tue 20 Oct** | Phase 2 builds on it; informed by the P1 v0 |
| D3 | Optimiser method (and what feasibility checks) | optimiser | **Tue 20 Oct** | same |
| D4 | Controller method | controller | **Tue 20 Oct** | same; must fit 20 ms on a Pi Zero 2W |
| D5 | Link: transport, format, rates; physical link (Wi-Fi / USB-Ethernet) | link | **Sat 24 Oct** | Phase 3 implements it |

## Phase 1: each module's v0 alone (Fri 16 – Tue 20 Oct)

| ID | Task | Module | Done when |
|---|---|---|---|
| P1.1 | Send our questions to the organisers (see bottom) | integration | email sent; answers recorded in the relevant READMEs |
| P1.2 | Hand-written walls: helper that builds a `WallModel` from grid indices (i, j); a realistic L1 wall for Microban's size | perception | L1 wall file loads with `rcr.io`, every hold is on the [Rules] grid |
| P1.3 | Planner v0 for L1 | planner | `--check` shows `planner.plan ok`; the plan starts at `wall.start` and only uses existing holds |
| P1.4 | Feasibility v0 | optimiser | `--check` shows the planner receiving a working `feasibility`; returns a verdict for each example transition |
| P1.5 | Optimiser v0: contact plan → trajectory | optimiser | `--check` shows `optimiser.optimise ok`; the output saves and loads as `trajectory/1` |
| P1.6 | Controller v0: trajectory → joint targets | controller | `--check` shows `controller.Controller.step ok`; Pi-safe test green; one step well under 1 ms on a laptop |
| P1.7 | Sim v0: MuJoCo scene with wall and holds from a `WallModel` plus the Microban model, headless `reset`/`step` | sim | `--check` shows `sim.Sim.step ok` (with `--group sim`) |
| P1.8 | Link v0: in-process loopback (no network) implementing both sides | link | `--check` shows both link stages `ok`; the sim run can go through the link API |

## Phase 2: everything connected in our sim; L1 end to end (gate Fri 23 Oct)

| ID | Task | Module | Done when |
|---|---|---|---|
| P2.1 | Placeholder climbing limbs on the Microban model | sim | limbs reach holds in the scene; joint names documented in INTERFACES.md |
| P2.2 | Swappable grip model v0 | sim | a limb can attach to and release a hold; the model is selected in one place |
| P2.3 | Servo limits and delays: XL330 servo model, 1.75 A current limit, Microban's delay defaults | sim | a pose needing more than 0.64 N·m sags or fails in sim |
| P2.4 | Scoring emulation: highest hold touched and timer [Rules §8] | sim | each sim run prints height climbed and time |
| P2.5 | `run_pipeline.py --backend sim`: closed loop Controller ↔ Sim through the link | integration | the loop runs at 50 Hz sim time, and the controller code is unchanged from P1.6 |
| P2.6 | Feasibility accounts for torque (0.64 N·m limit, margin) | optimiser | infeasible transitions in sim are rejected by `feasibility` |
| P2.7 | **Gate: L1 climb end to end in sim** | integration | the L1 wall is climbed to the top hold from `wall.start`, with no manual help |

## Phase 3: real robot, robustness, L2/L3 (freeze Wed 28 Oct)

| ID | Task | Module | Done when |
|---|---|---|---|
| P3.1 | Apply D1: Microban in place; deploy our Pi files next to it | robot | one command puts `types.py`, `io.py`, `controller/`, `link/` on a Pi |
| P3.2 | Wrap the controller as a Microban Move (`ClimbMove`) plus start/stop as an `InputSource`; handle the pitfalls in robot/README | robot | Microban's Scheduler runs our controller (in Microban's sim or on a Pi) with no edits to Move/Scheduler |
| P3.3 | Implement the chosen link transport (D5), both sides | link | laptop sends trajectory + start/stop, robot sends state back; robot side non-blocking and Pi-safe |
| P3.4 | Test laptop ↔ any Raspberry Pi over Wi-Fi 2.4 GHz and USB-Ethernet | link | trajectory up, start/stop, state back on both; latency and drop rate written in link/README |
| P3.5 | Run-day dry run following robot/README's run sequence | robot | the checklist runs top to bottom on a Pi (on Microban if available) |
| P3.6 | Robustness in sim: random delays, CoM offset, kp/sag, grip failures | sim | L1 still climbs over N randomised runs (N in the issue) |
| P3.7 | Handle servo sag under load (≈ τ / kp) | controller | tracking error under load is reduced in sim; still Pi-safe |
| P3.8 | ArUco 1–4 + two cameras rendered in the sim scene | sim | rendered images show all four markers at the [Rules] positions |
| P3.9 | Hand-written L2 (gaps) and L3 (lateral) walls | perception | files load and are on the grid |
| P3.10 | Planner handles L2 and L3 | planner | valid plans for the P3.9 walls |
| P3.11 | L2 and L3 climbs in sim | integration | both climbed to the top in sim |
| P3.12 | **Freeze**: tag `main`; only bug fixes after | integration | tag pushed Wed 28 Oct, CI green |

## Parked (not before the freeze unless the organisers' answers change it)
- Real perception: holds and robot pose from camera images (ArUco, `perception.wall_model` / `robot_pose`).
  If the official sim ranking requires perceiving the wall from rendered images ([Rules §7]), this item moves to Phase 3.
- Markerless perception (time factor ×1 vs ×1.25 with markers) [Rules §8].
- On-board vision.
- Ground start (×1 vs ×1.1 at the bottom of the wall) [Rules §8]; `"ground"` is reserved in `Stance`.
- Official simulator: when it is released, only `rcr.sim.Sim`'s internals change (sim/README).

## Questions for the organisers (P1.1)
1. **Simulator:** release date and contents (robot model, walls, holds, cameras, grip/contact model)?
2. **Climbing limbs:** when are they published (CAD, MJCF, joint names)? What is the 19th servo on the climber?
3. **On-site network:** is there a 2.4 GHz Wi-Fi we can use, may we bring our own access point, and can we run an
   Ethernet cable along the safety rope [Rules §5.3]?
4. **Sim ranking:** how and on which hardware are the official sim runs executed? Do we hand over code, and with which interface?
   May a run use a hand-written wall model, or must the wall be perceived from the rendered images [Rules §7]?
