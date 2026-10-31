"""Replay a joint trajectory (trajlevel3.npz by default) on a real Microban, the servos in position control with
configurable P, I and D gains (fixed for the run; no feedforward), while viser shows the robot as measured
and, as a translucent green shadow, the target it is sent (the trajectory's frame, the ramp; before the move runs: the
first frame), on the same fixed base.

The npz holds `q`, (frames, 26): a free-flyer base (7: x y z, quaternion w x y z; not driven) then the 19 joints
in the robot's joint order (FILE_JOINTS), one frame every --dt s. An npz with `joint_names` instead gives the joints
in `q_joints`.

1. The `replay` move starts, the gains are set through Microban's public API, and the robot holds where it is.
2. "Go to initial position" ramps the joints smoothly (--ramp s) to the first frame, and holds it.
3. "Start motion" (at the first frame only) streams the frames, the frame index taken from the time elapsed (a late PC
   tick skips ahead, it does not slow the replay down), at the Speed chosen on the page (SPEEDS, or --speed: 1 real
   time, 0.5 half speed: each frame held twice as long, 1.5: frames skipped to go 1.5 times as fast). The last frame
   is held at the end; "Go to initial position" again for another replay (or, mid-motion, to abort it).
Ctrl+C stops the move: the torque goes off (--hold keeps it on).

Every tick of a replay is logged: the target sent, the joints as measured (`q`) and as the robot echoes its goal
(`q_target`), each servo's present current (`current` [A], 1 mA steps; its sign is the motor's, not the joint's: Microban
does not apply MOTOR_SIGN to it [Microban]), the IMU's trunk orientation (`imu_quat`, w x y z) and the trajectory's
base orientation. When the replay
ends (or is aborted) the log is saved to runs/replay/replay-<date>-<time>.npz and the page plots, per joint, the measured
position against the target, and the base rotation error (the IMU's trunk against the trajectory's base, roll / pitch /
yaw of the error in the trunk frame). The IMU's yaw is arbitrary, so the trajectory is first turned about z to match
the IMU's yaw at the replay's start [Ours]. `--plot LOG.npz` shows a saved log's plots again, without a robot.

The goal sent to each servo is the trajectory's target alone: no feedforward torque offset, and the servos' own
velocity / acceleration feedforward gains stay at 0 (the replay move holds them there).

Needs the updated Microban position-gain API and src/moves/replay.py deployed on the robot:
run `make deploy HOST=radxa@<robot>` from that Microban checkout.
All gains are XL330 register values, 0..16383.

    uv run robot-replay --host microban-1a2b.local --kp 1000 --ki 1 --kd 200
    uv run robot-replay --plot runs/replay/replay-20261031-153000.npz
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pinocchio as pin
import viser
from pinocchio.visualize import ViserVisualizer
from viser import uplot

from rcr.controller.robot_link import LINK_PORT, MOVE_ACTIVE, RobotLink, wait_for_schema
from rcr.robot.microban import mjcf_path

DT = 0.02  # the robot's control tick (50 Hz)
TRUNK_HEIGHT = 0.168
TRAJECTORY = Path(__file__).resolve().parents[3] / "data/trajectories/wall-1-ik.npz"
SHADOW_COLOR = (0.2, 0.85, 0.35, 0.35)  # RGBA of the target robot
SPEEDS = (0.5, 1.0, 1.25, 1.5, 2.0)  # the replay speeds offered on the page (with --speed)
LOG_DIR = Path(__file__).resolve().parents[3] / "runs/replay"  # runs/ is git-ignored
TARGET_COLOR, MEASURED_COLOR = "#2a9d4b", "#1f5fbf"
AXIS_COLORS = ("#d1495b", "#2a9d4b", "#1f5fbf", "#555555")  # roll, pitch, yaw, total
CURRENT_COLOR = "#e07b00"
MAX_CURRENT_A = 1.75  # the XL330's firmware current limit [A] (BAM_MAX_CURRENT) [Microban]
# x in seconds, not a date (uPlot's default); the current on its own axis, on the right
CHART_SCALES = {"x": uplot.Scale(time=False), "y": uplot.Scale(auto=True), "A": uplot.Scale(auto=True)}

# Default, maximum, and meaning, in XL330 register units. Kp keeps its old effective default
# of 1000 (a KP=2000 request used to be clipped to 1000 by the robot moves); out-of-range
# requests are rejected explicitly.
GAIN_OPTIONS = {
    "kp": (2000, 16383, "position P gain"),
    "ki": (2000, 16383, "position I gain"),
    "kd": (1000, 16383, "position D gain"),
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


def load_trajectory(
    path: Path, joints: list[str], lower: np.ndarray, upper: np.ndarray
) -> tuple[np.ndarray, np.ndarray | None]:
    """The npz's joint trajectory (frames, joints) in the robot's joint order [rad], clipped to the slider range, and
    its base orientation (frames, 4: w x y z; None when the npz has no base)."""
    data = np.load(path)
    base_quat = None
    if "joint_names" in data:
        names, q = list(data["joint_names"]), data["q_joints"]
    else:
        names, q = FILE_JOINTS, data["q"]
        if q.ndim != 2 or q.shape[1] != 7 + len(FILE_JOINTS):
            raise SystemExit(f"{path}: expected q of shape (frames, {7 + len(FILE_JOINTS)}), got {q.shape}")
        base_quat, q = q[:, 3:7], q[:, 7:]
    missing = set(joints) - set(names)
    if missing:
        raise SystemExit(f"{path} has no column for {sorted(missing)}")
    q = q[:, [names.index(j) for j in joints]]
    return np.clip(q, lower, upper), base_quat


def quat_matrix(q: np.ndarray) -> np.ndarray:
    """The rotation matrix of a quaternion (w, x, y, z)."""
    return pin.Quaternion(*map(float, q)).normalized().toRotationMatrix()


def yaw_alignment(imu_quat: np.ndarray, ref_quat: np.ndarray) -> np.ndarray | None:
    """Rz (3, 3) turning the reference base about z onto the IMU's yaw, at the first tick where both quaternions are
    known (None if never): the IMU's yaw is arbitrary, its roll and pitch are referenced to gravity [Ours]."""
    ok = np.flatnonzero(~np.isnan(imu_quat).any(axis=1) & ~np.isnan(ref_quat).any(axis=1))
    if len(ok) == 0:
        return None
    m = quat_matrix(imu_quat[ok[0]]) @ quat_matrix(ref_quat[ok[0]]).T
    return pin.utils.rotate("z", np.arctan2(m[1, 0], m[0, 0]))


def base_rotation_error(imu_quat: np.ndarray, ref_quat: np.ndarray) -> np.ndarray:
    """(ticks, 3) [rad]: the rotation from the reference base to the measured trunk, log3(R_ref^T R_imu), in the trunk
    frame (x: roll, y: pitch, z: yaw). The reference is first turned about z to match the IMU's yaw at the first tick
    where both are known (the IMU's yaw is arbitrary; roll and pitch are referenced to gravity) [Ours]. NaN where a
    quaternion is missing."""
    err = np.full((len(imu_quat), 3), np.nan)
    yaw = yaw_alignment(imu_quat, ref_quat)
    if yaw is None:
        return err
    for i in np.flatnonzero(~np.isnan(imu_quat).any(axis=1) & ~np.isnan(ref_quat).any(axis=1)):
        err[i] = pin.log3((yaw @ quat_matrix(ref_quat[i])).T @ quat_matrix(imu_quat[i]))
    return err


def save_log(log: dict[str, list], info: dict) -> Path:
    """The replay's log as an npz in LOG_DIR: one row per tick, and the run's settings (info)."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / time.strftime("replay-%Y%m%d-%H%M%S.npz")
    np.savez(path, **{k: np.asarray(v, dtype=float) for k, v in log.items()},
             **{k: np.asarray(v) for k, v in info.items()})
    return path


def has_current(data) -> bool:
    """Whether the log has the servos' currents (logs from before they were logged do not)."""
    return "current" in data and not np.isnan(data["current"]).all()


def summarize(data) -> str:
    """Per-joint RMS and max tracking error [°] and, if logged, mean and max |current| [A]; the base rotation
    error [°]."""
    joints = list(data["joints"])
    err = np.rad2deg(np.abs(data["q"] - data["target"]))
    current = np.abs(data["current"]) if has_current(data) else None
    lines = [f"{'joint':<22}{'rms °':>8}{'max °':>8}" + (f"{'mean A':>9}{'max A':>8}" if current is not None else "")]
    for k, j in enumerate(joints):
        if np.isnan(err[:, k]).all():
            continue
        line = f"{j:<22}{np.sqrt(np.nanmean(err[:, k] ** 2)):8.2f}{np.nanmax(err[:, k]):8.2f}"
        if current is not None and not np.isnan(current[:, k]).all():
            line += f"{np.nanmean(current[:, k]):9.3f}{np.nanmax(current[:, k]):8.3f}"
        lines.append(line)
    if current is not None:
        total = np.nansum(current, axis=1)
        lines.append(f"total |current| (mean / max A): {np.mean(total):.2f} / {np.max(total):.2f}")
    rot = np.rad2deg(base_rotation_error(data["imu_quat"], data["base_quat_ref"]))
    if not np.isnan(rot).all():
        lines.append("base rotation error (rms / max °): " + ", ".join(
            f"{a} {np.sqrt(np.nanmean(rot[:, k] ** 2)):.2f} / {np.nanmax(np.abs(rot[:, k])):.2f}"
            for k, a in enumerate(("roll", "pitch", "yaw"))))
    return "\n".join(lines)


def plot_log(server: viser.ViserServer, data, title: str) -> viser.GuiFolderHandle:
    """A folder of uPlot charts on the page: the base rotation error, then each joint, measured against target."""
    t = np.asarray(data["t"], dtype=float)
    folder = server.gui.add_folder(title)
    with folder:
        rot = np.rad2deg(base_rotation_error(data["imu_quat"], data["base_quat_ref"]))
        if not np.isnan(rot).all():
            server.gui.add_uplot(
                data=(t, *rot.T, np.linalg.norm(rot, axis=1)),
                series=(uplot.Series(label="t [s]"),
                        *(uplot.Series(label=a, stroke=c, width=1.5)
                          for a, c in zip(("roll", "pitch", "yaw", "total"), AXIS_COLORS, strict=True))),
                scales=CHART_SCALES, title="base rotation error [°] (IMU vs trajectory)", aspect=1.6)
        else:
            server.gui.add_markdown("_no base rotation error: the log has no IMU or trajectory base orientation_")
        current = has_current(data)
        axes = (uplot.Axis(), uplot.Axis(scale="y", label="°"),
                *((uplot.Axis(scale="A", side=1, label="A", grid={"show": False}),) if current else ()))
        for k, joint in enumerate(data["joints"]):
            series = [uplot.Series(label="t [s]"),
                      uplot.Series(label="target", stroke=TARGET_COLOR, width=1.5, dash=(6, 3)),
                      uplot.Series(label="measured", stroke=MEASURED_COLOR, width=1.5)]
            columns = [t, np.rad2deg(data["target"][:, k]), np.rad2deg(data["q"][:, k])]
            if current:
                series.append(uplot.Series(label="current [A]", stroke=CURRENT_COLOR, width=1, scale="A"))
                columns.append(np.asarray(data["current"][:, k], dtype=float))
            server.gui.add_uplot(data=tuple(columns), series=tuple(series), scales=CHART_SCALES, axes=axes,
                                 title=f"{joint} [°" + (", A]" if current else "]"), aspect=1.6)
    return folder


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
    parser.add_argument("--host", help="the robot's host name or IP (required unless --plot)")
    parser.add_argument("--plot", type=Path, metavar="LOG",
                        help="only show the plots of a saved replay log (runs/replay/*.npz), no robot")
    parser.add_argument("--link-port", type=int, default=LINK_PORT)
    parser.add_argument("--move", default="replay",
                        help="the robot-side move (default: replay, Microban src/moves/replay.py)")
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
    if args.plot is not None:
        show_log(args.plot, args.port)
        return
    if args.host is None:
        parser.error("--host is required (or --plot LOG)")
    if args.speed <= 0:
        raise SystemExit("--speed must be > 0")
    gains = {name: getattr(args, name) for name in GAIN_OPTIONS}
    for name, value in gains.items():
        upper = GAIN_OPTIONS[name][1]
        if not 0 <= value <= upper:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and {upper}")
    gain_text = f"Kp {gains['kp']}, Ki {gains['ki']}, Kd {gains['kd']}"

    # ---- the robot ----
    link = RobotLink(args.host, args.link_port)
    print(f"connecting to {link.addr[0]}:{link.addr[1]} ...")
    wait_for_schema(link, timeout=5.0)
    meta = link.schema.get("meta", {})
    move_meta = link.check_move(
        args.move, f"deploy the updated Microban gain API and src/moves/{args.move}.py "
        "(make deploy HOST=radxa@<robot> from the Microban checkout)."
    )
    joints = link.labels("q")
    check_gain_sliders(move_meta, gains)
    slider_lo, slider_hi = np.array([move_meta["sliders"][j][:2] for j in joints]).T
    traj, traj_base_quat = load_trajectory(args.trajectory, joints, slider_lo, slider_hi)
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
    plots = None  # the folder of the last replay's plots

    @initial_button.on_click
    def _(_):
        requests.add("initial")

    @start_button.on_click
    def _(_):
        requests.add("start")

    @show_shadow.on_update
    def _(_):
        shadow.displayVisuals(show_shadow.value)

    server.gui.configure_theme(control_width="large")  # room for the plots
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

    def field(name: str, size: int) -> np.ndarray:
        """The last frame's field (NaN when the robot does not send it)."""
        return link.frame.get(name, np.full(size, np.nan)).astype(float)

    def new_log() -> dict[str, list]:
        return {k: [] for k in ("t", "robot_time", "frame", "target", "q", "q_target", "current",
                                "imu_quat", "base_quat_ref")}

    def record(log: dict[str, list], t: float, frame: int, target: np.ndarray, q: np.ndarray) -> None:
        """One tick of the replay: what was sent, what the robot measured."""
        log["t"].append(t)
        log["robot_time"].append(np.nan if link.frame_time is None else link.frame_time)
        log["frame"].append(frame)
        log["target"].append(target)
        log["q"].append(q)
        log["q_target"].append(field("q_target", len(joints)))
        log["current"].append(field("current", len(joints)))  # [A]; NaN if the robot does not send it
        log["imu_quat"].append(field("imu_quat", 4))
        log["base_quat_ref"].append(np.full(4, np.nan) if traj_base_quat is None else traj_base_quat[frame])

    def finish(log: dict[str, list], how: str, plot: bool = True) -> None:
        """Save the replay's log, print its summary, and plot it on the page."""
        nonlocal plots
        if not log["t"]:
            return
        path = save_log(log, {"joints": joints, "trajectory": str(args.trajectory), "dt": args.dt,
                              "speed": play_speed, "how": how,
                              "variant": meta.get("variant", "classic"), **gains})
        data = np.load(path)
        print(f"replay log ({how}, {len(log['t'])} ticks): {path}\n{summarize(data)}")
        log.clear()
        log.update(new_log())
        if plot:
            if plots is not None:
                plots.remove()
            plots = plot_log(server, data, f"Last replay ({how}, ×{play_speed:g})")

    started = False
    phase, log, play_speed = "starting", {"t": []}, 1.0
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
        log = new_log()
        last_status = 0.0
        print("holding: press Go to initial position")
        while True:
            if "initial" in requests and phase != "at initial":
                if phase == "playing":
                    finish(log, "aborted")
                ramp, phase = Ramp(target, traj[0], args.ramp), "to initial"
                print(f"ramping to the first frame over {args.ramp:.1f} s")
            if "start" in requests and phase == "at initial":
                play_speed, errors = float(speed.value), []
                phase, play_t0 = "playing", time.monotonic()
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
            if phase == "playing" or (phase == "done" and log["t"]):  # done: its last frame, once
                errors.append(max_error(target, q))
                record(log, time.monotonic() - play_t0, frame_i, target, q)
                if phase == "done":
                    finish(log, "done")

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
        if phase == "playing":
            finish(log, "move stopped", plot=False)
    except KeyboardInterrupt:
        if phase == "playing":
            finish(log, "interrupted", plot=False)
    finally:
        if started and not args.hold:
            print("\nmove_stop: the torque goes off")
            link.command("move_stop", move=args.move)
            time.sleep(0.2)
            link.poll()  # its ack
        link.close()
        server.stop()


def show_log(path: Path, port: int) -> None:
    """Print a saved replay log's summary and plot it on a viser page, until Ctrl+C."""
    data = np.load(path)
    print(f"{path}: {len(data['t'])} ticks ({data['how']}, ×{float(data['speed']):g})\n{summarize(data)}")
    server = viser.ViserServer(port=port)
    server.gui.configure_theme(control_width="large")
    plot_log(server, data, path.name)
    print(f"plots at http://localhost:{port} (Ctrl+C to quit)")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
