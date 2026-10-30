"""Replay a joint trajectory (trajlevel3.npz by default) on a real Microban, the servos in position control with
configurable PID and trajectory feedforward gains (fixed for the run), while viser shows the robot as measured
and, as a translucent green shadow, the target it is sent (the trajectory's frame, the ramp; before the move runs: the
first frame), on the same fixed base.

The npz holds `q`, (frames, 26): a free-flyer base (7, ignored: the robot's base is not driven) then the 19 joints in
the robot's joint order (FILE_JOINTS), one frame every --dt s.

1. The `replay` move starts, the gains are set through Microban's public API, and the robot holds where it is.
2. "Go to initial position" ramps the joints smoothly (--ramp s) to the first frame, and holds it.
3. "Start motion" (at the first frame only) streams the frames, the frame index taken from the time elapsed (a late PC
   tick skips ahead, it does not slow the replay down), at the Speed chosen on the page (SPEEDS, or --speed: 1 real
   time, 0.5 half speed: each frame held twice as long, 1.5: frames skipped to go 1.5 times as fast). The last frame
   is held at the end; "Go to initial position" again for another replay (or, mid-motion, to abort it).
Ctrl+C stops the move: the torque goes off (--hold keeps it on).

Needs the updated Microban position-gain API and src/moves/replay.py deployed on the robot:
run `make deploy HOST=radxa@<robot>` from that Microban checkout.
All gains are XL330 register values: Kp 0..1000; Ki, Kd and both feedforward gains 0..16383.
First-order feedforward acts on the servo's internal velocity trajectory, second-order on its acceleration.

    uv run python -m rcr.controller.robot_replay --host microban-1a2b.local --kp 1000 --ki 1 --kd 200
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pinocchio as pin
import viser
from pinocchio.visualize import ViserVisualizer

from rcr.controller.robot_link import LINK_PORT, MOVE_ACTIVE, RobotLink, wait_for_schema
from rcr.robot.microban import mjcf_path

DT = 0.02  # the robot's control tick (50 Hz)
TRUNK_HEIGHT = 0.168
TRAJECTORY = Path(__file__).resolve().parents[3] / "data/trajectories/level3-2steps-highergains.npz"
SHADOW_COLOR = (0.2, 0.85, 0.35, 0.35)  # RGBA of the target robot
SPEEDS = (0.5, 1.0, 1.25, 1.5)  # the replay speeds offered on the page (with --speed)

# Default, maximum, and meaning, in XL330 register units. The old KP=2000 request was clipped to 1000 by the
# remote move's slider range; preserve that effective value and reject out-of-range requests explicitly.
GAIN_OPTIONS = {
    "kp": (1000, 1000, "position P gain"),
    "ki": (2000, 16383, "position I gain"),
    "kd": (1500, 16383, "position D gain"),
    "feedforward_1st_gain": (0, 16383, "internal trajectory velocity feedforward gain"),
    "feedforward_2nd_gain": (0, 16383, "internal trajectory acceleration feedforward gain"),
}

# The joints of the npz's columns 7.. (the robot's joint order, not the model's).
FILE_JOINTS = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "left_shoulder_pitch", "left_shoulder_roll", "left_elbow",
    "right_shoulder_pitch", "right_shoulder_roll", "right_elbow",
    "head",
]


class Ramp:
    """From one pose to another over duration (smoothstep); duration 0: at once."""

    def __init__(self, start: np.ndarray, end: np.ndarray, duration: float):
        self.start, self.end, self.duration, self.t = start, end, duration, 0.0

    def step(self) -> np.ndarray:
        self.t += DT
        s = min(self.t / self.duration, 1.0) if self.duration > 0 else 1.0
        s = s * s * (3 - 2 * s)
        return self.start + s * (self.end - self.start)

    @property
    def done(self) -> bool:
        return self.duration <= 0 or self.t >= self.duration


class MoveStopped(Exception):
    """The move stopped on the robot (dashboard, em, error)."""


def load_trajectory(path: Path, joints: list[str], lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """The npz's joint trajectory (frames, joints) in the robot's joint order [rad], clipped to the slider range."""
    q = np.load(path)["q"]
    if q.ndim != 2 or q.shape[1] != 7 + len(FILE_JOINTS):
        raise SystemExit(f"{path}: expected q of shape (frames, {7 + len(FILE_JOINTS)}), got {q.shape}")
    missing = set(joints) - set(FILE_JOINTS)
    if missing:
        raise SystemExit(f"{path} has no column for {sorted(missing)}")
    q = q[:, 7:][:, [FILE_JOINTS.index(j) for j in joints]]
    return np.clip(q, lower, upper)


def check_gain_sliders(move_meta: dict, gains: dict[str, int]) -> None:
    """Require the new move's gain controls and reject values the robot would otherwise silently clip."""
    sliders = move_meta.get("sliders", {})
    missing = gains.keys() - sliders.keys()
    if missing:
        raise SystemExit(f"The robot's replay move is missing gain sliders {sorted(missing)}: "
                         "deploy the updated Microban API and src/moves/replay.py.")
    for name, value in gains.items():
        lower, upper = sliders[name][:2]
        if not lower <= value <= upper:
            raise SystemExit(f"{name}={value} is outside this move's slider range [{lower}, {upper}].")


def send_target(link: RobotLink, move: str, joints: list[str], target: np.ndarray, gains: dict[str, int]) -> None:
    """Resend sliders each tick (UDP); the move uses set_* and the scheduler writes only changed gains."""
    for name, value in (*gains.items(), *zip(joints, target)):
        link.command("slider", move=move, slider=name, value=float(value))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", required=True, help="the robot's host name or IP")
    parser.add_argument("--link-port", type=int, default=LINK_PORT)
    parser.add_argument("--move", default="replay", help="the robot-side move (Microban src/moves/replay.py)")
    parser.add_argument("--trajectory", type=Path, default=TRAJECTORY, help="the npz to replay")
    parser.add_argument("--dt", type=float, default=0.02, help="time between two frames of the trajectory [s]")
    parser.add_argument("--speed", type=float, default=1.0, help=f"replay speed (0.5: half speed, 1.5: 1.5 times as fast), on the page: {', '.join(f'{x:g}' for x in SPEEDS)}")
    parser.add_argument("--ramp", type=float, default=3.0, help="duration of the ramp to the first frame [s]")
    parser.add_argument("--hold", action="store_true", help="on exit, leave the move running (torque on)")
    parser.add_argument("--port", type=int, default=8080, help="viser port")
    for name, (default, upper, description) in GAIN_OPTIONS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=int, default=default,
                            help=f"{description}, register units 0..{upper} (default: {default})")
    args = parser.parse_args()
    if args.speed <= 0:
        raise SystemExit("--speed must be > 0")
    gains = {name: getattr(args, name) for name in GAIN_OPTIONS}
    for name, value in gains.items():
        upper = GAIN_OPTIONS[name][1]
        if not 0 <= value <= upper:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and {upper}")
    gain_text = (f"Kp {gains['kp']}, Ki {gains['ki']}, Kd {gains['kd']}, "
                 f"FF1 {gains['feedforward_1st_gain']}, FF2 {gains['feedforward_2nd_gain']}")

    # ---- the robot ----
    link = RobotLink(args.host, args.link_port)
    print(f"connecting to {link.addr[0]}:{link.addr[1]} ...")
    wait_for_schema(link, timeout=5.0)
    meta = link.schema.get("meta", {})
    move_meta = link.check_move(
        args.move, "deploy the updated Microban gain API and src/moves/replay.py "
        "(make deploy HOST=radxa@<robot> from the Microban checkout)."
    )
    check_gain_sliders(move_meta, gains)
    joints = link.labels("q")
    slider_lo, slider_hi = np.array([move_meta["sliders"][j][:2] for j in joints]).T
    traj = load_trajectory(args.trajectory, joints, slider_lo, slider_hi)
    duration = (len(traj) - 1) * args.dt
    print(f"{args.trajectory.name}: {len(traj)} frames, {duration:.1f} s at {args.dt} s per frame")

    # ---- viser: the robot as measured, and its target as a shadow ----
    model, _, collision_model, visual_model = pin.buildModelsFromMJCF(
        str(mjcf_path(climber=meta.get("variant") == "climber"))
    )
    q_idx = np.array([model.idx_qs[model.getJointId(j)] for j in joints])
    q_disp = pin.neutral(model)
    q_disp[2] = TRUNK_HEIGHT
    server = viser.ViserServer(port=args.port)
    viz = ViserVisualizer(model, collision_model, visual_model)
    viz.initViewer(viewer=server, loadModel=True)
    server.scene.add_grid("/ground", width=1.0, height=1.0, cell_size=0.05)
    shadow = ViserVisualizer(model, collision_model, visual_model)
    shadow.initViewer(viewer=server)
    shadow.loadViewerModel(rootNodeName="target", visual_color=SHADOW_COLOR)
    q_shadow = q_disp.copy()
    q_shadow[q_idx] = traj[0]
    shadow.display(q_shadow)

    status = server.gui.add_markdown("starting...")
    initial_button = server.gui.add_button("Go to initial position", disabled=True)
    start_button = server.gui.add_button("Start motion", disabled=True)
    speed = server.gui.add_dropdown("Speed", [f"{x:g}" for x in sorted({*SPEEDS, args.speed})],
                                    initial_value=f"{args.speed:g}")
    show_shadow = server.gui.add_checkbox("Show target (shadow)", True)
    requests = set()  # from the buttons (viser's thread), handled in the control loop

    @initial_button.on_click
    def _(_):
        requests.add("initial")

    @start_button.on_click
    def _(_):
        requests.add("start")

    @show_shadow.on_update
    def _(_):
        shadow.displayVisuals(show_shadow.value)

    print(f"viser running at http://localhost:{args.port}")

    next_tick = time.perf_counter()

    def tick(target: np.ndarray | None) -> np.ndarray:
        """One control tick: read the robot, send the gains and target (if any), display, wait. Returns q measured."""
        nonlocal next_tick
        link.poll()
        q = link.frame.get("q", np.full(len(joints), np.nan))
        if target is not None:
            if link.move_state(args.move) != MOVE_ACTIVE:
                raise MoveStopped
            send_target(link, args.move, joints, target, gains)
        ok = ~np.isnan(q)
        q_disp[q_idx[ok]] = q[ok]
        viz.display(q_disp)
        if target is not None and show_shadow.value:
            q_shadow[q_idx] = target
            shadow.display(q_shadow)
        next_tick += DT
        sleep = next_tick - time.perf_counter()
        if sleep > 0:
            time.sleep(sleep)
        else:
            next_tick = time.perf_counter()
        return q

    def max_error(target: np.ndarray, q: np.ndarray) -> float:
        return float(np.rad2deg(np.nanmax(np.abs(target - q))))

    started = False
    try:
        # 1. start the move
        print(f"starting {args.move} ({gain_text})")
        link.command("move_start", move=args.move)
        started = True
        t0 = time.monotonic()
        q = tick(None)
        while link.move_state(args.move) != MOVE_ACTIVE:
            if time.monotonic() - t0 > 2.0:
                raise SystemExit(f"{args.move} did not start (see the robot's answer above)")
            q = tick(None)

        # 2. hold where the robot is, then follow the buttons
        target = np.where(np.isnan(q), traj[0], q)
        phase = "holding"  # holding, to initial, at initial, playing, done
        ramp = None
        play_t0, play_speed, frame_i, errors = 0.0, 1.0, 0, []
        last_status = 0.0
        print("holding: press Go to initial position")
        while True:
            if "initial" in requests and phase != "at initial":
                ramp, phase = Ramp(target, traj[0], args.ramp), "to initial"
                print(f"ramping to the first frame over {args.ramp:.1f} s")
            if "start" in requests and phase == "at initial":
                phase, play_t0, play_speed, errors = "playing", time.monotonic(), float(speed.value), []
                print(f"playing at speed {play_speed:g} ({duration / play_speed:.1f} s)")
            requests.clear()

            if phase == "to initial":
                target = ramp.step()
                if ramp.done:
                    phase = "at initial"
                    print("at the first frame: press Start motion")
            elif phase == "playing":
                frame_i = min(int((time.monotonic() - play_t0) * play_speed / args.dt), len(traj) - 1)
                target = traj[frame_i]
                if frame_i == len(traj) - 1:
                    phase = "done"
                    print(f"replay done in {time.monotonic() - play_t0:.1f} s, max tracking error "
                          f"{np.nanmax(errors):.1f}°, mean {np.nanmean(errors):.1f}°: holding the last frame")
            q = tick(target)
            if phase == "playing":
                errors.append(max_error(target, q))

            for button, disabled in ((initial_button, phase in ("to initial", "at initial")),
                                     (start_button, phase != "at initial")):
                if button.disabled != disabled:
                    button.disabled = disabled
            if time.monotonic() - last_status > 0.2:
                last_status = time.monotonic()
                progress = f" · frame {frame_i + 1} / {len(traj)} · ×{play_speed:g}" if phase in ("playing", "done") else ""
                status.content = (f"**{phase}**{progress} · max error {max_error(target, q):.1f}° · "
                                  f"{gain_text}")
    except MoveStopped:
        print(f"{args.move} stopped on the robot")
        started = False
    except KeyboardInterrupt:
        pass
    finally:
        if started and not args.hold:
            print("\nmove_stop: the torque goes off")
            link.command("move_stop", move=args.move)
            time.sleep(0.2)
            link.poll()  # its ack
        link.close()
        server.stop()


if __name__ == "__main__":
    main()
