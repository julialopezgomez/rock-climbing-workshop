"""Run the tracking policy (or playback) on the robot, closed loop, from the laptop.

Like the replay (robot_replay.py on feature/control), but each target is computed live by
rcr.controller.Controller from the state the robot measures, not read from a file:

    robot frame (q, dq, IMU, gyro) --Wi-Fi--> RobotState -> Controller.step -> 19 targets
    --Wi-Fi--> the robot's "replay" move, which sets them as the servos' position targets

    uv run --no-sync scripts/robot_policy.py --dry-run                 # the sim plays the robot
    uv run --no-sync scripts/robot_policy.py --host <robot> --playback # the reference only (a = 0)
    uv run --no-sync scripts/robot_policy.py --host <robot>            # the policy

The robot runs Microban with the replay move and its gain API (feature/control's Microban), the
servos at the policy's kp_fw with Ki = Kd = 0 (P only, as in the sim the policy trained in). On the
robot: the move starts and holds where the robot is; Enter ramps to the reference's first pose
(--ramp s) and holds it; Enter again starts the climb (t = 0); it holds the last target after the
reference's end, until Ctrl+C. On exit the move stops and the torque goes off (--hold keeps it on).
Every tick is logged to runs/policy/<time>.npz.
"""

import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np

from rcr import io
from rcr.controller import Controller, tracking
from rcr.types import RobotState

DT = 0.02  # [s] the robot's control tick (50 Hz)
GAINS = ("kp", "ki", "kd", "feedforward_1st_gain", "feedforward_2nd_gain")  # the move's sliders


class LinkRobot:
    """The real robot, through its dashboard link and the replay move's sliders."""

    def __init__(self, host: str, move: str, kp: int):
        from rcr.controller.robot_link import MOVE_ACTIVE, RobotLink, wait_for_schema

        self.link, self.move, self.active = RobotLink(host), move, MOVE_ACTIVE
        print(f"connecting to {host} ...")
        wait_for_schema(self.link, timeout=5.0)
        entry = self.link.check_move(move, "deploy feature/control's Microban (make deploy).")
        self.joints = self.link.labels("q")
        sliders = entry.get("sliders", {})
        missing = [s for s in (*GAINS, *self.joints) if s not in sliders]
        if missing:
            raise SystemExit(f"the {move} move has no sliders {missing}: deploy its gain API")
        self.low, self.high = np.array([sliders[j][:2] for j in self.joints]).T
        self.gains = {g: 0 for g in GAINS} | {"kp": kp}  # P only, as in the sim
        print(f"frame fields: {[name for name, _ in self.link.schema['fields']]}")

    def start(self) -> None:
        self.link.command("move_start", move=self.move)
        end = time.monotonic() + 2.0
        while self.link.move_state(self.move) != self.active:
            if time.monotonic() > end:
                raise SystemExit(f"{self.move} did not start (see the robot's answer above)")
            self.link.poll()
            time.sleep(DT)

    def read(self) -> dict:
        self.link.poll()
        if self.link.move_state(self.move) != self.active:
            raise SystemExit(f"{self.move} stopped on the robot")
        return dict(self.link.frame)

    def send(self, targets: dict[str, float]) -> None:
        q = np.clip([targets[j] for j in self.joints], self.low, self.high)
        for name, value in (*self.gains.items(), *zip(self.joints, q, strict=True)):
            self.link.command("slider", move=self.move, slider=name, value=float(value))

    def stop(self) -> None:
        self.link.command("move_stop", move=self.move)
        self.link.close()


class SimRobot:
    """The organisers' sim in the robot's place, at the reference's first pose, settled."""

    def __init__(self, data: Path, trajectory, kp: float):
        from climbing_sim import JOINTS, ClimbingSim, Wall

        self.sim = ClimbingSim(Wall.load(str(data / "sim_wall.json")), cameras=[], kp_fw=kp)
        self.joints = list(JOINTS)
        pose = trajectory.trunk_poses[0]
        start = dict(zip(trajectory.joint_names, trajectory.planned_positions[0], strict=True))
        self.sim.reset(pose[:3], pose[3:], start)
        for _ in range(25):  # 0.5 s holding the first pose
            self.sim.step([start[j] for j in self.joints])
        self.ticks, self.fell = 0, None  # when fewer than two balls are hooked

    def start(self) -> None:
        pass

    def read(self) -> dict:
        obs = self.sim.observation()
        return {"q": obs["q"], "dq": obs["dq"], "imu_quat": obs["quat"], "gyro": obs["gyro"]}

    def send(self, targets: dict[str, float]) -> None:
        self.sim.step([targets[j] for j in self.joints])  # one call = one 20 ms tick
        self.ticks += 1
        hooked = sum(h is not None for h in self.sim.hooked().values())
        if self.fell is None and hooked < 2:
            self.fell = self.ticks * DT
            print(f"sim: fell at t = {self.fell:.2f} s (fewer than two balls hooked)")

    def stop(self) -> None:
        print(f"sim: balls hooked at the end: {self.sim.hooked()}")


def state(frame: dict, joints: list[str], t: float, last_q: np.ndarray | None) -> RobotState:
    """The robot's frame as the controller's input. Joint velocities from the frame if it has
    them, else from the last two q; the gyro is 0 if the frame has none (warned at start)."""
    q = np.asarray(frame["q"], dtype=float)
    dq = frame.get("dq")
    if dq is None:
        dq = np.zeros_like(q) if last_q is None else (q - last_q) / DT
    quat = frame.get("imu_quat", frame.get("quat"))
    gyro = frame.get("gyro", np.zeros(3))
    return RobotState(t, dict(zip(joints, q, strict=True)), dict(zip(joints, dq, strict=True)),
                      tuple(float(v) for v in quat), tuple(float(v) for v in gyro))  # fmt: skip


def enter_pressed() -> threading.Event:
    """Set when Enter is pressed, without stopping the 50 Hz loop."""
    event = threading.Event()
    threading.Thread(target=lambda: (input(), event.set()), daemon=True).start()
    return event


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--host", help="the robot's host name or IP")
    parser.add_argument("--dry-run", action="store_true", help="the sim plays the robot")
    parser.add_argument("--data", type=Path, default=Path("build/rl/n1"), help="trajectory.json")
    parser.add_argument(
        "--policy", type=Path, default=Path("data/policies/clone_kp1000_ls-1_s2.onnx")
    )
    parser.add_argument("--playback", action="store_true", help="no policy: the reference only")
    parser.add_argument("--kp", type=float, help="servo P gain (default: the policy's kp_fw)")
    parser.add_argument("--move", default="replay", help="the robot-side move")
    parser.add_argument("--ramp", type=float, default=3.0, help="[s] to the first pose")
    parser.add_argument("--seconds", type=float, help="stop after this long (default: the end)")
    parser.add_argument("--hold", action="store_true", help="on exit, leave the torque on")
    args = parser.parse_args()
    if not args.dry_run and args.host is None:
        parser.error("--host (or --dry-run)")

    trajectory = io.load_trajectory(args.data / "trajectory.json")
    kp = args.kp or json.loads(args.policy.with_suffix(".json").read_text())["kp_fw"]
    controller = Controller(trajectory) if args.playback else Controller(trajectory, args.policy)
    end = trajectory.times[-1] + 1.0 if args.seconds is None else args.seconds
    first = dict(zip(tracking.JOINTS, tracking.at(controller.reference, 0.0)[0], strict=True))
    robot = (
        SimRobot(args.data, trajectory, kp)
        if args.dry_run
        else LinkRobot(args.host, args.move, int(kp))
    )
    frame = robot.read()
    for name in ("q", "imu_quat", "dq", "gyro"):
        if name not in frame and not (name == "imu_quat" and "quat" in frame):
            print(
                f"warning: the robot sends no {name!r}"
                + (": computed from q" if name == "dq" else "")
            )
    print(f"{'playback' if args.playback else args.policy} at kp_fw {kp:g}, {end:.1f} s")

    log = {"t": [], "q": [], "target": [], "robot_time": []}
    robot.start()
    try:
        if not args.dry_run:  # hold, ramp to the first pose, hold, then climb on Enter
            hold = dict(zip(robot.joints, np.asarray(robot.read()["q"], dtype=float), strict=True))
            print("holding where the robot is: Enter ramps to the first pose")
            go = enter_pressed()
            while not go.is_set():
                robot.read()
                robot.send(hold)
                time.sleep(DT)
            go, ramp_t0 = enter_pressed(), time.monotonic()
            print(f"ramping to the first pose in {args.ramp:g} s, then holding: Enter starts")
            while not go.is_set():
                s = min(1.0, (time.monotonic() - ramp_t0) / args.ramp)
                robot.read()
                robot.send({j: hold[j] + s * (first[j] - hold[j]) for j in robot.joints})
                time.sleep(DT)
        print("climbing (Ctrl+C stops)")
        t, last_q, next_tick, t0 = 0.0, None, time.perf_counter(), time.monotonic()
        while t <= end:
            frame = robot.read()
            targets = controller.step(state(frame, robot.joints, t, last_q)).positions
            robot.send(targets)
            last_q = np.asarray(frame["q"], dtype=float)
            log["t"].append(t)
            log["q"].append(last_q)
            log["target"].append([targets[j] for j in robot.joints])
            log["robot_time"].append(frame.get("time", np.nan) if not args.dry_run else t)
            if args.dry_run:
                t += DT  # the sim advances one tick per send
            else:  # the robot runs in real time: a late tick catches up, it doesn't slow the climb
                next_tick += DT
                time.sleep(max(0.0, next_tick - time.perf_counter()))
                t = time.monotonic() - t0
    except KeyboardInterrupt:
        print("stopped")
    finally:
        if not args.hold:
            robot.stop()
        if log["t"]:
            path = Path("runs/policy") / time.strftime("%Y%m%d-%H%M%S.npz")
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, joints=robot.joints, **{k: np.array(v) for k, v in log.items()})
            print(f"log: {path} ({len(log['t'])} ticks)")


if __name__ == "__main__":
    main()
