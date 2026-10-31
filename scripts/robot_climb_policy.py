"""Run an mjlab policy bundle on the real robot, as sim_check_climb_policy.py does in sim [Ours].

    uv run --no-sync scripts/robot_climb_policy.py ../mjlab_microban/logs/deploy/<bundle> \\
        --host <robot> --view --seconds 60

1. Copies our Pi code (scripts/link_pi.py deploy: ~/rcr/src) and the bundle (~/climb_policy) to
   the robot. rsync sends only what changed: the weights are cached on the robot after the first
   copy, and nothing here needs the internet (--no-copy: skip this step).
2. Starts Microban's program with our move (make run MOVES=src/rcr/robot/climb_policy_move.py;
   --program-running: it already runs, started by hand).
3. Asks to confirm the robot is on the wall (balls in the start holds, rope on; --yes: no
   question), then starts the move climb_policy over the dashboard link. The robot runs the policy
   by itself at 50 Hz; this script only starts / stops it and watches.
4. --view: the wall and the robot in MuJoCo's viewer, live: the joints the robot measures and its
   IMU orientation (yaw aligned on the start holds); the trunk position is an estimate (the balls
   in a socket stay there), the robot does not measure it.
5. After --seconds of climbing (after the move's ramp and start delay), Ctrl+C or the window
   closed: stops the move (it holds, then the torque goes off: the robot hangs on its rope), stops
   the program, saves every frame to runs/robot_climb/<time>.npz and fetches the move's log
   (~/climb_logs on the robot).

Needs the sim group for --view only (uv sync --group sim --group rl).
"""

import argparse
import importlib.util
import math
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rcr.controller.robot_link import MOVE_ACTIVE, RobotLink, wait_for_schema  # noqa: E402

MOVE_FILE = ROOT / "src" / "rcr" / "robot" / "climb_policy_move.py"
MOVE = "climb_policy"  # Microban's name of ClimbPolicyMove (moves/discover.py)
HOOK_TOL = 0.025  # [m] a ball this close to a socket is in it, for the view's trunk estimate


def move_options() -> dict:
    """The move's default options (ramp_s, start_delay_s, stop_hold_s, ...), from its file."""
    spec = importlib.util.spec_from_file_location("climb_policy_move", MOVE_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(module.ClimbPolicyMove.options)


def bundle_dir(path: Path) -> Path:
    path = path.resolve()
    if path.is_file():
        path = path.parent
    if not (path / "climb_bundle.npz").is_file() or not (path / "policy.onnx").is_file():
        sys.exit(f"{path}: not a bundle folder (climb_bundle.npz + policy.onnx)")
    return path


def copy_to_robot(host: str, bundle: Path) -> None:
    print(f"[ROBOT] our Pi code to {host}:~/rcr/src", flush=True)
    if subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "link_pi.py"), "deploy", host]
    ).returncode:
        sys.exit("copy of our code failed: is the robot on this network (ssh works)?")
    print(f"[ROBOT] the bundle to {host}:~/climb_policy (only what changed)", flush=True)
    command = [
        "rsync",
        "-a",
        "--itemize-changes",
        "--delete",
        "--exclude",
        "parity.npz",
        "--exclude",
        "sim_check.npz",
        "--exclude",
        "__pycache__",
        f"{bundle}/",
        f"{host}:climb_policy/",
    ]
    out = subprocess.run(command, capture_output=True, text=True)
    if out.returncode:
        sys.exit(f"copy of the bundle failed: {out.stderr.strip()}")
    sent = [line for line in out.stdout.splitlines() if line.startswith("<f")]
    print(f"[ROBOT]   {len(sent)} file(s) sent" + ("" if sent else ": already on the robot"))


def start_program(host: str, microban: Path) -> subprocess.Popen:
    """Microban's program on the robot with our move, in its own process (its output here)."""
    if not (microban / "Makefile").is_file():
        sys.exit(f"Microban not found at {microban} (--microban)")
    print(f"[ROBOT] starting the robot's program (make run, {microban})", flush=True)
    return subprocess.Popen(
        ["make", "-C", str(microban), "run", f"HOST={host}", f"MOVES={MOVE_FILE}"],
        stdin=subprocess.PIPE,
        start_new_session=True,  # Ctrl+C here reaches this script only: it stops the move first
    )


def stop_program(program: subprocess.Popen | None) -> None:
    """Ctrl+C to the robot's program (ssh -tt: its pty turns it into SIGINT), so that its cleanup
    runs (torque off), then the local make."""
    if program is None or program.poll() is not None:
        return
    try:
        program.stdin.write(b"\x03")
        program.stdin.flush()
        program.wait(timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        program.terminate()
        program.wait(timeout=5)


class View:
    """The wall and the robot as measured, in MuJoCo's viewer (the organisers' sim's scene)."""

    def __init__(self, bundle: Path, window: bool = True) -> None:
        import mujoco
        from climbing_sim import JOINTS, ClimbingSim, Wall

        self.mujoco, self.joints = mujoco, list(JOINTS)
        self.sim = ClimbingSim(Wall.from_json(bundle / "wall.json"), cameras=[])
        self.sim.reset_hanging(settle_s=0.0)  # the start: the four balls in their sockets
        d = self.sim.data
        self.pos, self.R_start = d.qpos[0:3].copy(), quat_mat(d.qpos[3:7])
        self.sockets = self.sim.wall.sockets()
        self.adr = [self.sim.model.joint(n).qposadr[0] for n in self.joints]
        self.yaw_align = None
        self.order = None  # the link's joint order -> JOINTS
        self.viewer = None
        if window:
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(self.sim.model, d)

    def running(self) -> bool:
        return self.viewer is None or self.viewer.is_running()

    def show(self, q: np.ndarray, imu_quat: np.ndarray) -> np.ndarray:
        """The robot at joints q (the link's order) and IMU orientation; its trunk estimate."""
        q = np.asarray(q, float)
        if self.order is not None:
            q = q[self.order]
        if not (np.isfinite(q).all() and np.isfinite(imu_quat).all()):
            return self.pos
        R_imu = quat_mat(imu_quat)
        if self.yaw_align is None:  # the IMU's yaw reference is arbitrary: the start's
            self.yaw_align = yaw_of(self.R_start) - yaw_of(R_imu)
        R = rot_z(self.yaw_align) @ R_imu
        quat = mat_quat(R)
        rel = np.array(list(self.sim.ball_positions_at(q, np.zeros(3), quat).values()))
        balls = self.pos + rel
        dist = np.linalg.norm(balls[:, None] - self.sockets[None], axis=-1)
        hooked = dist.min(1) < HOOK_TOL
        if hooked.any():  # the balls in a socket stay there: the trunk follows from them
            self.pos = np.mean(self.sockets[dist.argmin(1)][hooked] - rel[hooked], axis=0)
        d = self.sim.data
        d.qpos[0:3], d.qpos[3:7] = self.pos, quat
        d.qpos[self.adr] = q
        self.mujoco.mj_forward(self.sim.model, d)
        if self.viewer is not None:
            self.viewer.sync()
        return self.pos


def quat_mat(q) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def mat_quat(R) -> np.ndarray:
    import mujoco

    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R).flatten())
    return q


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def yaw_of(R) -> float:
    return math.atan2(R[1, 0], R[0, 0])


def fetch_log(host: str, out: Path) -> None:
    """The move's log of this run (the newest of ~/climb_logs on the robot)."""
    name = subprocess.run(
        ["ssh", host, "ls -t climb_logs/*.npz 2>/dev/null | head -1"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    if name and not subprocess.run(["scp", "-q", f"{host}:{name}", str(out)]).returncode:
        print(f"[ROBOT] the move's log: {out / Path(name).name}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("bundle", type=Path, help="the bundle folder (or its climb_bundle.npz)")
    p.add_argument("--host", required=True, help="the robot (ssh host, e.g. microban)")
    p.add_argument("--seconds", type=float, default=60.0, help="of climbing, then the move stops")
    p.add_argument("--view", action="store_true", help="the robot live in MuJoCo's viewer")
    p.add_argument("--no-copy", action="store_true", help="the code and bundle are on the robot")
    p.add_argument("--program-running", action="store_true", help="make run is already running")
    p.add_argument("--yes", action="store_true", help="no confirmation before the climb")
    p.add_argument("--microban", type=Path, default=ROOT / "third_party" / "microban")
    p.add_argument(
        "--connect-timeout",
        type=float,
        default=90.0,
        help="[s] for the program to start and answer (make run syncs Microban first)",
    )
    args = p.parse_args()

    bundle = bundle_dir(args.bundle)
    options = move_options()
    lead = float(options["ramp_s"]) + float(options["start_delay_s"])
    out = ROOT / "runs" / "robot_climb"
    out.mkdir(parents=True, exist_ok=True)
    view = View(bundle) if args.view else None  # before the robot: a missing sim group fails here

    if not args.no_copy:
        copy_to_robot(args.host, bundle)
    program = None if args.program_running else start_program(args.host, args.microban)
    link = None
    rec: dict[str, list] = {
        k: []
        for k in (
            "time",
            "robot_time",
            "q",
            "q_target",
            "dq",
            "imu_quat",
            "gyro",
            "current",
            "torque",
            "move",
            "trunk_view",
        )
    }
    stop = {"asked": False}
    joints: list[str] = []
    try:
        link = RobotLink(args.host)
        print(f"[ROBOT] waiting for the robot's program (up to {args.connect_timeout:.0f} s) ...")
        wait_for_schema(link, timeout=args.connect_timeout)
        link.check_move(
            MOVE, f"run with MOVES={MOVE_FILE} (this script does, without --program-running)"
        )
        joints = link.labels("q")
        if view is not None:
            view.order = [joints.index(n) for n in view.joints]
        if not args.yes:
            input(
                f"[ROBOT] robot on the wall, its balls in the start holds, rope on? Enter starts "
                f"{MOVE} (ramp {options['ramp_s']} s, hold {options['start_delay_s']} s, then "
                f"{args.seconds:.0f} s of climbing); Ctrl+C aborts "
            )
        # From here, Ctrl+C asks for a stop (the move's own smooth stop), it does not kill.
        signal.signal(signal.SIGINT, lambda *_: stop.update(asked=True))
        link.command("move_start", move=MOVE)
        t0 = time.monotonic()
        while link.move_state(MOVE) != MOVE_ACTIVE:
            if time.monotonic() - t0 > 3.0:
                sys.exit(f"{MOVE} did not start (see the robot's answer above)")
            link.poll()
            time.sleep(0.02)
        print(
            f"[ROBOT] {MOVE} running: climbing from t = {lead:.0f} s, for {args.seconds:.0f} s "
            "(Ctrl+C stops it)"
        )
        frames, last_print = link.frames_received, 0.0
        while True:
            link.poll()
            now = time.monotonic() - t0
            if link.frames_received != frames:  # a new 50 Hz frame
                frames = link.frames_received
                f = link.frame
                trunk = view.show(f["q"], f["imu_quat"]) if view is not None else np.full(3, np.nan)
                rec["time"].append(now)
                rec["robot_time"].append(link.frame_time)
                for k in ("q", "q_target", "dq", "imu_quat", "gyro", "current", "torque"):
                    rec[k].append(np.array(f.get(k, [np.nan]), float))
                rec["move"].append(link.move_state(MOVE))
                rec["trunk_view"].append(trunk)
                if now - last_print >= 5.0:
                    last_print = now
                    total = np.nansum(np.abs(f.get("current", [np.nan])))
                    print(
                        f"[ROBOT] t = {now:5.1f} s, total current {total:.1f} A, "
                        f"tick {float(f.get('tick_ms', [np.nan])[0]):.1f} ms"
                    )
            if link.move_state(MOVE) != MOVE_ACTIVE:
                print(f"[ROBOT] {MOVE} stopped on the robot (see its events above)")
                break
            if now > lead + args.seconds:
                print(f"[ROBOT] {args.seconds:.0f} s of climbing done")
                break
            if stop["asked"] or (view is not None and not view.running()):
                print("[ROBOT] stopping")
                break
            time.sleep(0.002)
    finally:
        if link is not None and link.schema is not None:
            link.command("move_stop", move=MOVE)  # it holds stop_hold_s, then the torque goes off
            end = time.monotonic() + float(options["stop_hold_s"]) + 3.0
            while time.monotonic() < end and link.move_state(MOVE) != 0:
                link.poll()
                if view is not None and view.running() and link.frame.get("q") is not None:
                    view.show(link.frame["q"], link.frame["imu_quat"])
                time.sleep(0.02)
            link.close()
        stop_program(program)
        if rec["time"]:
            path = out / time.strftime("climb_%Y%m%d_%H%M%S.npz")
            np.savez(
                path,
                joint_names=np.array(joints),
                bundle=str(bundle),
                **{k: np.array(v) for k, v in rec.items()},
            )
            print(f"[ROBOT] {len(rec['time'])} frames: {path}")
            fetch_log(args.host, out)


if __name__ == "__main__":
    main()
