"""Plot replay logs (runs/replay/*.npz, saved by robot-replay) on a viser page.

    uv run scripts/plot_replay.py                        # the newest log
    uv run scripts/plot_replay.py --last 3               # the 3 newest, one folder each
    uv run scripts/plot_replay.py runs/replay/replay-20261031-110008.npz --joints left_knee
    uv run scripts/plot_replay.py --list                 # the logs, with their settings and errors

For each log, the terminal prints its summary (per-joint RMS / max tracking error, base rotation
error), and the page shows, in a folder, the base rotation error (IMU vs trajectory, yaw aligned at
the first tick) and each joint's measured position against its target (see
rcr.controller.robot_replay).

Above the plots, "Replay" plays the motion back in time (a time slider, Play / Pause, speed): the
robot as measured (its joints, its trunk oriented by the IMU) and, as a green shadow, the target
(the joints sent, the base oriented as in the trajectory, turned to the IMU's yaw). A sphere at each
joint shows its tracking error at that time, green (0°) to amber (--error-scale and above), or, with
"Spheres show" on current, the servo's |current| (green 0 A to amber at --current-warn). In either
mode, a servo whose |current| is at or above --current-warn (default 1.5 A, 85 % of the XL330's
1.75 A limit [Microban]; the threshold is [Ours]) is drawn red and larger: close to the current
limit. The panel lists those servos, the base rotation error and the worst joints (by the quantity
shown, the ones close to the limit first) with their current; the terminal lists, per servo, how
long it was close to the limit. With several logs, "Log" picks the one replayed.

The holds are found from the commanded trajectory (the log's npz, or --trajectory): forward
kinematics of the four balls (rcr.robot.hooks) in the wall frame, where a ball stays still
(< HOLD_SPEED for HOLD_MIN_TIME) it is in a hold, and stays within HOLD_MERGE are one hold [Ours].
The terminal lists them in the wall frame [m]; the viewer draws them, and the wall face (y = 0), in
the frame of the reference robot (the shadow): through the trajectory's base pose at the current
frame, so the target's balls sit in them while in contact. The holds in contact at that frame are
highlighted.
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pinocchio as pin
import viser
from pinocchio.visualize import ViserVisualizer

from rcr.controller.robot_replay import (
    FILE_JOINTS,
    LOG_DIR,
    MAX_CURRENT_A,
    SHADOW_COLOR,
    TRUNK_HEIGHT,
    base_rotation_error,
    has_current,
    plot_log,
    quat_matrix,
    summarize,
    yaw_alignment,
)
from rcr.robot import hooks
from rcr.robot.microban import mjcf_path

REPO = Path(__file__).resolve().parents[1]

SPEEDS = ("0.1", "0.25", "0.5", "1", "2")
# joint sphere colours: the gradient from 0 to the scale, then red for a servo close to the current
# limit (red means that only)
GOOD, HIGH = np.array([40, 170, 80]), np.array([245, 165, 0])
LIMIT_COLOR, NO_DATA_COLOR = (230, 25, 25), (150, 150, 150)
SPHERE_RADIUS = 0.012
LIMIT_SPHERE_SCALE = 1.8  # a servo close to the current limit: a larger sphere
CURRENT_WARN = 1.5  # [A] "close to the limit": about 85 % of MAX_CURRENT_A [Ours]
WORST_SHOWN = 6  # joints listed in the panel
HOLD_SPEED = 0.01  # [m/s] a ball slower than this is in a hold [Ours]
HOLD_MIN_TIME = 0.3  # [s] for at least this long [Ours]
HOLD_MERGE = 0.015  # [m] contacts closer than this are the same hold [Ours]
HOLD_RADIUS = 0.015  # [m] as drawn
HOLD_COLOR, HOLD_ACTIVE_COLOR = (150, 110, 70), (255, 140, 0)
WALL_COLOR, WALL_MARGIN = (190, 190, 200), 0.1  # [m] the wall face drawn around the holds


@dataclass
class Hold:
    position: np.ndarray  # [m] the ball centre in the hold, wall frame
    contacts: list[tuple[str, int, int]]  # (limb, first frame, end frame (excluded))


def load_reference(path: Path, model: pin.Model) -> np.ndarray:
    """The trajectory's frames as the model's configurations (frames, nq): base x y z, quaternion
    x y z w (pinocchio's order), joints. Raises ValueError if the npz has no base."""
    q_file = np.load(path).get("q")
    if q_file is None or q_file.ndim != 2 or q_file.shape[1] != 7 + len(FILE_JOINTS):
        raise ValueError(f"{path} has no q (frames, {7 + len(FILE_JOINTS)}) with a base")
    idx = np.array([model.idx_qs[model.getJointId(j)] for j in FILE_JOINTS])
    q = np.tile(pin.neutral(model), (len(q_file), 1))
    q[:, :3], q[:, 3:7], q[:, idx] = q_file[:, :3], q_file[:, [4, 5, 6, 3]], q_file[:, 7:]
    return q


def find_holds(model: pin.Model, reference: np.ndarray, dt: float) -> list[Hold]:
    """The holds of a reference trajectory: where its balls stay still, in the wall frame [Ours]."""
    pdata = model.createData()
    frame_ids = {limb: model.getFrameId(h.link) for limb, h in hooks.HOOKS.items()}
    balls = {limb: np.zeros((len(reference), 3)) for limb in hooks.LIMBS}
    for i, q in enumerate(reference):
        pin.framesForwardKinematics(model, pdata, q)
        for limb, fid in frame_ids.items():
            balls[limb][i] = pdata.oMf[fid] * hooks.ball_centre(hooks.HOOKS[limb])
    holds: list[Hold] = []
    min_frames = max(int(round(HOLD_MIN_TIME / dt)), 1)
    for limb, p in balls.items():
        still = np.r_[np.linalg.norm(np.diff(p, axis=0), axis=1) < HOLD_SPEED * dt, False]
        edges = np.flatnonzero(np.diff(np.r_[False, still].astype(int)))  # starts, ends
        for a, b in zip(edges[::2], edges[1::2], strict=True):
            b += 1  # the frame after the last still step is still in the hold
            if b - a < min_frames:
                continue
            centre = p[a:b].mean(axis=0)
            near = [h for h in holds if np.linalg.norm(h.position - centre) < HOLD_MERGE]
            if near:
                near[0].contacts.append((limb, a, b))
            else:
                holds.append(Hold(centre, [(limb, a, b)]))
    holds.sort(key=lambda h: (h.position[2], h.position[0]))  # bottom to top, left to right
    return holds


def resolve_trajectory(data, override: Path | None) -> Path | None:
    """The log's trajectory npz: --trajectory, else the path the log names (as given, from the
    repo, or by name in data/trajectories)."""
    if override is not None:
        return override
    if "trajectory" not in data:
        return None
    named = Path(str(data["trajectory"]))
    for path in (named, REPO / named, REPO / "data/trajectories" / named.name):
        if path.exists():
            return path
    return None


def describe(path: Path, data) -> str:
    """One line: the log's name, length, how it ended, speed, gains and worst joint RMS error."""
    gains = " ".join(f"{k} {int(data[k])}" for k in ("kp", "ki", "kd") if k in data)
    rms = np.sqrt(np.nanmean(np.rad2deg(data["q"] - data["target"]) ** 2, axis=0))
    worst = ""
    if not np.isnan(rms).all():
        worst = f"worst rms {np.nanmax(rms):.2f}° ({data['joints'][np.nanargmax(rms)]})"
    trajectory = Path(str(data["trajectory"])).name if "trajectory" in data else "?"
    return (
        f"{path.name}: {len(data['t'])} ticks, {data['t'][-1]:.1f} s, {data['how']}, "
        f"×{float(data['speed']):g}, {trajectory}, {gains}, {worst}"
    )


def select_joints(data, joints: list[str] | None) -> dict:
    """The log as a dict, with only the given joints (all if None)."""
    out = {k: data[k] for k in data.files}
    if joints:
        names = [str(j) for j in data["joints"]]
        unknown = set(joints) - set(names)
        if unknown:
            raise SystemExit(f"unknown joints {sorted(unknown)}; the log has {names}")
        cols = [names.index(j) for j in joints]
        out["joints"] = np.array(joints)
        for k in ("target", "q", "q_target", "current"):
            if k in out:
                out[k] = data[k][:, cols]
    return out


def base_rotations(data) -> tuple[np.ndarray, np.ndarray]:
    """(ticks, 3, 3) each: the trunk as the IMU measured it, and the trajectory's base turned to the
    IMU's yaw (as in base_rotation_error). Upright (identity) where unknown."""
    n = len(data["t"])
    measured, target = np.tile(np.eye(3), (n, 1, 1)), np.tile(np.eye(3), (n, 1, 1))
    imu, ref = data["imu_quat"], data["base_quat_ref"]
    yaw = yaw_alignment(imu, ref)
    for i in range(n):
        if not np.isnan(imu[i]).any():
            measured[i] = quat_matrix(imu[i])
        if yaw is not None and not np.isnan(ref[i]).any():
            target[i] = yaw @ quat_matrix(ref[i])
    return measured, target


def report_near_limit(data, near: np.ndarray, warn: float) -> None:
    """Print, per servo, how long it was close to the current limit, and when first."""
    t, joints = data["t"], [str(j) for j in data["joints"]]
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.02
    lines = [
        f"  {joints[k]:<22}{near[:, k].sum() * dt:6.2f} s, first at t {t[near[:, k]][0]:.2f} s, "
        f"max {np.nanmax(np.abs(data['current'][:, k])):.2f} A"
        for k in range(len(joints))
        if near[:, k].any()
    ]
    print(f"close to the current limit (|current| ≥ {warn:g} A):")
    print("\n".join(lines) if lines else "  none")


class Viewer:
    """The robot as measured and its target (shadow) at one tick of a log, with error spheres."""

    def __init__(
        self, server: viser.ViserServer, climber: bool, error_scale: float, current_warn: float
    ):
        self.server, self.error_scale, self.current_warn = server, error_scale, current_warn
        self.reference, self.holds, self.hold_spheres, self.wall = None, [], [], None
        self.model, _, collision, visual = pin.buildModelsFromMJCF(str(mjcf_path(climber=climber)))
        self.pdata = self.model.createData()
        self.measured = ViserVisualizer(self.model, collision, visual)
        self.measured.initViewer(viewer=server, loadModel=True)
        self.shadow = ViserVisualizer(self.model, collision, visual)
        self.shadow.initViewer(viewer=server)
        self.shadow.loadViewerModel(rootNodeName="target", visual_color=SHADOW_COLOR)
        server.scene.add_grid("/ground", width=1.0, height=1.0, cell_size=0.05)
        self.spheres = {}
        self.data = None

    def load(self, data, trajectory: Path | None) -> None:
        self.data = data
        self.load_holds(trajectory, float(data["dt"]) if "dt" in data else 0.02)
        self.joints = [str(j) for j in data["joints"]]
        self.q_idx = np.array([self.model.idx_qs[self.model.getJointId(j)] for j in self.joints])
        self.joint_ids = [self.model.getJointId(j) for j in self.joints]
        self.err = np.rad2deg(np.abs(data["q"] - data["target"]))
        self.current = np.abs(data["current"]) if has_current(data) else None  # [A]
        self.near_limit = np.zeros_like(self.err, dtype=bool)
        if self.current is not None:
            self.near_limit = np.nan_to_num(self.current, nan=0.0) >= self.current_warn
            report_near_limit(data, self.near_limit, self.current_warn)
        self.rot_err = np.rad2deg(base_rotation_error(data["imu_quat"], data["base_quat_ref"]))
        self.r_measured, self.r_target = base_rotations(data)
        for name in self.joints:
            if name not in self.spheres:
                self.spheres[name] = self.server.scene.add_icosphere(
                    f"/errors/{name}", radius=SPHERE_RADIUS, color=tuple(GOOD), cast_shadow=False
                )

    def load_holds(self, trajectory: Path | None, dt: float) -> None:
        """The reference trajectory's holds, drawn as spheres (and the wall face behind them)."""
        for sphere in self.hold_spheres:
            sphere.remove()
        if self.wall is not None:
            self.wall.remove()
        self.reference, self.holds, self.hold_spheres, self.wall = None, [], [], None
        if trajectory is None:
            print("no trajectory npz for this log (--trajectory): no holds shown")
            return
        try:
            self.reference = load_reference(trajectory, self.model)
        except (ValueError, OSError) as e:
            print(f"no holds shown: {e}")
            return
        self.holds = find_holds(self.model, self.reference, dt)
        print(f"{len(self.holds)} holds in {trajectory.name} (wall frame, ball centre [m]):")
        for k, h in enumerate(self.holds):
            limbs = ", ".join(sorted({c[0] for c in h.contacts}))
            x, y, z = h.position
            print(f"  H{k + 1:<3} x {x:.3f}  y {y:.3f}  z {z:.3f}   {limbs}")
        self.hold_spheres = [
            self.server.scene.add_icosphere(
                f"/holds/H{k + 1}", radius=HOLD_RADIUS, color=HOLD_COLOR, cast_shadow=False
            )
            for k in range(len(self.holds))
        ]
        if self.holds:
            p = np.array([h.position for h in self.holds])
            lo, hi = p.min(axis=0) - WALL_MARGIN, p.max(axis=0) + WALL_MARGIN
            self.wall_centre = np.array([(lo[0] + hi[0]) / 2, 0.0025, (lo[2] + hi[2]) / 2])
            self.wall = self.server.scene.add_box(
                "/holds/wall",
                color=WALL_COLOR,
                dimensions=(hi[0] - lo[0], 0.005, hi[2] - lo[2]),
                opacity=0.35,
                cast_shadow=False,
            )

    def show_holds(self, i: int, oriented: bool) -> str:
        """Place the holds and the wall in the reference robot's frame at tick i; returns the
        panel's line on the contacts."""
        if self.reference is None:
            return ""
        f = min(int(self.data["frame"][i]), len(self.reference) - 1)
        ref = self.reference[f]
        r_ref = pin.Quaternion(ref[3:7]).normalized().toRotationMatrix()  # an array: x y z w
        r_shown = self.r_target[i] if oriented else np.eye(3)
        m = r_shown @ r_ref.T  # wall frame -> the shown frame (base at (0, 0, TRUNK_HEIGHT))
        origin = np.array([0.0, 0.0, TRUNK_HEIGHT])
        contacts = []
        for k, (h, sphere) in enumerate(zip(self.holds, self.hold_spheres, strict=True)):
            sphere.position = origin + m @ (h.position - ref[:3])
            limbs = [limb for limb, a, b in h.contacts if a <= f < b]
            sphere.color = HOLD_ACTIVE_COLOR if limbs else HOLD_COLOR
            contacts += [f"{limb} H{k + 1}" for limb in limbs]
        if self.wall is not None:
            self.wall.position = origin + m @ (self.wall_centre - ref[:3])
            x, y, z, w = pin.Quaternion(m).coeffs()
            self.wall.wxyz = (w, x, y, z)
        return "reference contacts: " + (", ".join(sorted(contacts)) or "none") + "\n\n"

    def configuration(self, joints: np.ndarray, rotation: np.ndarray, oriented: bool) -> np.ndarray:
        q = pin.neutral(self.model)
        q[2] = TRUNK_HEIGHT
        if oriented:
            q[3:7] = pin.Quaternion(rotation).coeffs()  # x y z w
        ok = ~np.isnan(joints)
        q[self.q_idx[ok]] = joints[ok]
        return q

    def show(self, i: int, oriented: bool, show_shadow: bool, by_current: bool = False) -> str:
        """Display tick i, the spheres coloured by tracking error or (by_current, if logged) by
        |current|; returns the panel's markdown."""
        by_current = by_current and self.current is not None
        value = self.current[i] if by_current else self.err[i]
        scale = self.current_warn if by_current else self.error_scale
        near = self.near_limit[i]
        d = self.data
        q = self.configuration(d["q"][i], self.r_measured[i], oriented)
        self.measured.display(q)
        if show_shadow:
            self.shadow.display(self.configuration(d["target"][i], self.r_target[i], oriented))
        pin.forwardKinematics(self.model, self.pdata, q)
        for name, jid, e, limit in zip(self.joints, self.joint_ids, value, near, strict=True):
            sphere = self.spheres[name]
            sphere.position = self.pdata.oMi[jid].translation
            sphere.scale = LIMIT_SPHERE_SCALE if limit else 1.0
            if limit:
                sphere.color = LIMIT_COLOR
            elif np.isnan(e):
                sphere.color = NO_DATA_COLOR
            else:
                s = min(e / scale, 1.0)
                sphere.color = tuple(int(c) for c in (1 - s) * GOOD + s * HIGH)
        # the servos close to the limit first, then the largest values
        order = np.lexsort((-np.nan_to_num(value, nan=-1.0), ~near))
        worst = list(order[: max(WORST_SHOWN, int(near.sum()))])
        rot = self.rot_err[i]
        rot_text = (
            "n/a"
            if np.isnan(rot).all()
            else f"roll {rot[0]:+.1f}° · pitch {rot[1]:+.1f}° · yaw {rot[2]:+.1f}° "
            f"(total {np.linalg.norm(rot):.1f}°)"
        )

        def amps(k: int) -> str:
            if self.current is None or np.isnan(self.current[i, k]):
                return "–"
            a = f"{self.current[i, k]:.2f}"
            if near[k]:
                return f"**{a} ⚠**"
            return f"**{a}**" if by_current else a

        def error(k: int) -> str:
            return f"{self.err[i, k]:.1f}" if by_current else f"**{self.err[i, k]:.1f}**"

        rows = "\n".join(
            f"| {self.joints[k]} | {np.rad2deg(d['target'][i, k]):+.1f} | "
            f"{np.rad2deg(d['q'][i, k]):+.1f} | {error(k)} | {amps(k)} |"
            for k in worst
        )
        total = (
            f" · total |current| {np.nansum(self.current[i]):.2f} A"
            if self.current is not None
            else " · no currents in this log"
        )
        limit_text = ""
        if near.any():
            names = ", ".join(
                f"{self.joints[k]} {self.current[i, k]:.2f} A" for k in np.flatnonzero(near)
            )
            limit_text = (
                f"**⚠ close to the current limit (≥ {self.current_warn:g} A): {names}**\n\n"
            )
        return (
            f"**t {d['t'][i]:.2f} s** · tick {i + 1} / {len(d['t'])} · "
            f"frame {int(d['frame'][i])}\n\n"
            f"{limit_text}"
            f"{self.show_holds(i, oriented)}"
            f"base rotation error: {rot_text}{total}\n\n"
            f"| joint | target ° | measured ° | error ° | current A |"
            f"\n|---|---|---|---|---|\n{rows}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "logs", nargs="*", type=Path, help=f"npz logs (default: the newest in {LOG_DIR})"
    )
    parser.add_argument(
        "--last", type=int, default=1, help="without logs: plot the N newest (default: 1)"
    )
    parser.add_argument("--joints", help="comma-separated joints to plot (default: all)")
    parser.add_argument("--list", action="store_true", help="only list the logs in runs/replay")
    parser.add_argument("--port", type=int, default=8080, help="viser port")
    parser.add_argument(
        "--error-scale", type=float, default=10.0, help="error [°] shown amber (default: 10)"
    )
    parser.add_argument(
        "--current-warn",
        type=float,
        default=CURRENT_WARN,
        help=f"|current| [A] from which a servo is close to the limit, drawn red (default: "
        f"{CURRENT_WARN}, about 85 %% of the XL330's {MAX_CURRENT_A} A)",
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        help="the commanded trajectory npz, for the holds (default: the one the log names)",
    )
    parser.add_argument(
        "--variant",
        choices=("climber", "classic"),
        help="the robot model shown (default: the log's, else climber)",
    )
    args = parser.parse_args()

    available = sorted(LOG_DIR.glob("*.npz"))  # replay-<date>-<time>: oldest first
    if args.list:
        for path in available:
            print(describe(path, np.load(path)))
        if not available:
            print(f"no logs in {LOG_DIR}")
        return
    logs = args.logs or available[-args.last :]
    if not logs:
        raise SystemExit(f"no logs in {LOG_DIR}: run robot-replay first, or give a path")
    joints = args.joints.split(",") if args.joints else None
    selected = [(path, np.load(path)) for path in logs]
    for _, data in selected:
        select_joints(data, joints)  # fail on an unknown joint before starting the page

    server = viser.ViserServer(port=args.port)
    server.gui.configure_theme(control_width="large")
    newest_first = list(reversed(selected))
    by_name = {path.name: data for path, data in newest_first}
    variant = args.variant or str(newest_first[0][1].get("variant", "climber"))
    viewer = Viewer(
        server,
        climber=variant == "climber",
        error_scale=args.error_scale,
        current_warn=args.current_warn,
    )

    with server.gui.add_folder("Replay"):
        log_choice = server.gui.add_dropdown("Log", list(by_name), visible=len(by_name) > 1)
        slider = server.gui.add_slider("Time [s]", 0.0, 1.0, 0.02, 0.0)
        play = server.gui.add_button("Play")
        speed = server.gui.add_dropdown("Speed", SPEEDS, initial_value="1")
        oriented = server.gui.add_checkbox("Orient the base (IMU / trajectory)", True)
        show_shadow = server.gui.add_checkbox("Show target (shadow)", True)
        color_by = server.gui.add_dropdown(
            "Spheres show", ("tracking error", "current"), initial_value="tracking error"
        )
        panel = server.gui.add_markdown("")
    for path, data in newest_first:  # the newest first on the page
        print(f"{describe(path, data)}\n{summarize(data)}\n")
        title = f"{path.stem} ({data['how']}, ×{float(data['speed']):g})"
        plot_log(server, select_joints(data, joints), title)

    playing = [False]

    @play.on_click
    def _(_):
        playing[0] = not playing[0]
        if playing[0] and slider.value >= slider.max:
            slider.value = 0.0
        play.label = "Pause" if playing[0] else "Play"

    @show_shadow.on_update
    def _(_):
        viewer.shadow.displayVisuals(show_shadow.value)

    def load(name: str) -> None:
        data = by_name[name]
        viewer.load(data, resolve_trajectory(data, args.trajectory))
        t = data["t"]
        slider.max = float(t[-1])
        slider.step = float(np.median(np.diff(t))) if len(t) > 1 else 0.02
        slider.value = 0.0

    print(f"viewer and plots at http://localhost:{args.port} (Ctrl+C to quit)")
    loaded, shown = None, None
    last = time.perf_counter()
    try:
        while True:
            if log_choice.value != loaded:
                loaded = log_choice.value
                load(loaded)
                shown = None
            now = time.perf_counter()
            if playing[0]:
                value = slider.value + (now - last) * float(speed.value)
                if value >= slider.max:
                    value, playing[0], play.label = slider.max, False, "Play"
                slider.value = value
            last = now
            t = viewer.data["t"]
            i = int(np.clip(np.searchsorted(t, slider.value), 0, len(t) - 1))
            by_current = color_by.value == "current"
            state = (i, oriented.value, show_shadow.value, by_current, loaded)
            if state != shown:
                shown = state
                panel.content = viewer.show(i, oriented.value, show_shadow.value, by_current)
            time.sleep(1 / 30)
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
