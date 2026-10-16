# link

**Purpose:** the **only** thing that crosses laptop ↔ robot during a run.

| Direction | What | When |
|---|---|---|
| laptop → robot | `Trajectory` | before the climb (and again if the laptop replans) |
| laptop → robot | start / stop | once each |
| laptop → robot | robot `Pose` from the cameras (optional) | live, if the controller uses it |
| robot → laptop | `RobotState` | live, for monitoring and recording |

| | |
|---|---|
| Runs on | both: `LaptopLink` on the laptop; `RobotLink` on the Pi (Pi-safe, every method non-blocking because it is called from the 50 Hz loop) |

## Design points
- **Nothing time-critical crosses the link.** The 50 Hz loop lives on the robot, and a late or lost message must
  never stall a tick. This makes 2.4 GHz Wi-Fi viable, with wired as a backup.
- **Planning and optimisation never run on the Pi.** If we replan, the laptop does it and sends the result.
- **Microban provides SSH/rsync before a run, but nothing during it** [Microban Makefile]. So the run-time link is ours.
- **On the Pi it plugs in as a Microban input source and/or inside our Move:**
  - start/stop becomes an `InputSource` that sets `active_moves`;
  - the trajectory and pose are read inside the Move.
  - See [../robot/README.md](../robot/README.md).
- **Autonomy:** no teleoperation [Rules §2.4]. The laptop sends a plan and start/stop, never live motion commands.
- **Allowed links:** any wireless link or the Ethernet tether along the safety rope is allowed [Rules §2.4, §5.3].

## Open choices (D5, decide by Sat 24 Oct)
- **Transport:**
  - plain TCP/UDP sockets (stdlib, Pi-safe);
  - HTTP;
  - ZeroMQ (needs pyzmq on the Pi, which is not in Microban's env);
  - or a degenerate zero-code v0: `scp` the trajectory file before the run and start/stop over `ssh`.
- **Format:** JSON via `rcr.io` (already Pi-safe) or something binary. Size is not an issue: 19 joints × 50 Hz × 60 s
  ≈ 57k numbers ≈ 0.5 MB of JSON, sent once.
- **Rates:** robot state back at a few Hz up to 50 Hz; camera pose at camera rate (if used).
- **Physical:** Wi-Fi 2.4 GHz or USB-Ethernet; see robot/README for the recommendation.
- What happens on link loss mid-climb (continue the trajectory? stop?).
