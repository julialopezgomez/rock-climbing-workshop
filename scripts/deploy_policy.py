"""Run a W&B climbing policy on the laptop, streaming joint targets to the robot over Wi-Fi [Ours].

    # once, with internet: the checkpoint -> data/policies/<run_id>/ (policy.onnx + bundle)
    uv run --no-sync scripts/deploy_policy.py fetch kotochleb-cnrs-laas/mjlab_microban_climbing/9tqq5l4d

    # offline from here
    uv run --no-sync scripts/deploy_policy.py run 9tqq5l4d --sim            # the sim plays the robot
    uv run --no-sync scripts/deploy_policy.py run 9tqq5l4d --host <robot>   # the real robot

The robot runs Microban's `replay` move (third_party/microban, branch feature/expose-pid-params): one
slider per joint + kp/ki/kd, set over the dashboard link. Each 50 Hz tick here: robot frame ->
rcr.controller.mjlab_policy.ClimbController (onnxruntime) -> 19 targets -> sliders.
Sequence on the robot: ramp to the climb's pose at --start-s (--ramp s), hold; Enter starts the climb;
Ctrl+C stops (the move stops, torque off).

fetch: model_<N>.pt (newest, or --checkpoint) -> policy.onnx (obs normaliser + ELU MLP written with
onnx.helper, no torch at run time), plus a copy of data/climb_bundle_l1s0/ (reference, observation
layout, wall: Mjlab-Climbing-Microban-Adaptive-Phase on motion l1s0, kp 120).
"""

import argparse
import json
import re
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from rcr.controller.mjlab_policy import G_DOWN, ClimbController, quat_mat  # noqa: E402

TEMPLATE = ROOT / "data" / "climb_bundle_l1s0"
POLICIES = ROOT / "data" / "policies"
NORM_EPS = 1e-2  # rsl_rl's EmpiricalNormalization: (x - mean) / (std + eps)
DT = 0.02


# -- fetch -----------------------------------------------------------------------------------------


def actor_to_onnx(state: dict, path: Path) -> None:
    """policy.onnx (input "obs", output "actions") from rsl_rl's actor_state_dict."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    def arr(name, x):
        return numpy_helper.from_array(np.asarray(x, np.float32), name)

    mean = state["obs_normalizer._mean"].numpy().reshape(-1)
    std = state["obs_normalizer._std"].numpy().reshape(-1)
    layers = sorted({int(m.group(1)) for k in state if (m := re.match(r"mlp\.(\d+)\.weight", k))})
    inits = [arr("mean", mean), arr("den", std + NORM_EPS)]
    nodes = [
        helper.make_node("Sub", ["obs", "mean"], ["x0"]),
        helper.make_node("Div", ["x0", "den"], ["h0"]),
    ]
    h = "h0"
    for i, idx in enumerate(layers):
        inits += [arr(f"w{i}", state[f"mlp.{idx}.weight"]), arr(f"b{i}", state[f"mlp.{idx}.bias"])]
        last = i == len(layers) - 1
        out = "actions" if last else f"z{i}"
        nodes.append(helper.make_node("Gemm", [h, f"w{i}", f"b{i}"], [out], transB=1))
        if not last:
            h = f"h{i + 1}"
            nodes.append(helper.make_node("Elu", [out], [h], alpha=1.0))
    n_out = state[f"mlp.{layers[-1]}.bias"].numel()
    graph = helper.make_graph(
        nodes,
        "actor",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, len(mean)])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, n_out])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def fetch(args) -> None:
    import torch
    import wandb

    run = wandb.Api(timeout=60).run(args.run_path)
    files = {f.name: f for f in run.files() if re.fullmatch(r"model_\d+\.pt", f.name)}
    name = args.checkpoint or max(files, key=lambda n: int(n[6:-3]), default=None)
    if name not in files:
        sys.exit(f"{name} not in W&B run {args.run_path}: {sorted(files)}")
    out = POLICIES / run.id
    out.mkdir(parents=True, exist_ok=True)
    files[name].download(root=str(out), replace=True)
    ckpt = torch.load(out / name, map_location="cpu", weights_only=False)
    actor_to_onnx(ckpt["actor_state_dict"], out / "policy.onnx")
    b = dict(np.load(TEMPLATE / "climb_bundle.npz", allow_pickle=False))
    b["checkpoint"] = np.array(f"{args.run_path}/{name}")
    # the adaptive phase as this run trained it (the template is another run's)
    ph = run.config["env_cfg"]["actions"]["phase_rate"]
    nan_if_none = lambda x: float("nan") if x is None else float(x)  # noqa: E731
    b["phase_sigma"], b["phase_k"] = float(ph["sigma"]), float(ph["k"])
    b["phase_delta_clip"] = nan_if_none(ph.get("delta_clip"))
    b["phase_min_rate"] = nan_if_none(ph.get("min_rate"))
    b["phase_rate_terms"] = np.array(ph["rate_terms"])
    b["phase_tolerances"] = np.array([ph["tolerances"][t] for t in ph["rate_terms"]])
    np.savez(out / "climb_bundle.npz", **b)
    shutil.copy(TEMPLATE / "wall.json", out / "wall.json")
    info = {"run_path": args.run_path, "checkpoint": name, "template": TEMPLATE.name}
    (out / "info.json").write_text(json.dumps(info, indent=2))
    print(f"[FETCH] {name} of {args.run_path}: bundle ready (offline from here): {out}")


# -- run -------------------------------------------------------------------------------------------


class Sim:
    """The organisers' sim in the robot's place, hanging on the bundle's start holds."""

    def __init__(self, bundle: Path, kp: int, start: int = 0):
        from climbing_sim import JOINTS, ClimbingSim, Wall
        from rcr.controller.mjlab_policy import mat_quat

        self.sim = ClimbingSim(Wall.from_json(bundle / "wall.json"), cameras=[], kp_fw=kp)
        self.sim.reset_hanging()
        self.joints = list(JOINTS)
        if start > 0:  # hanging on the holds the reference has the balls in at that frame
            from climbing_sim import LIMBS

            b = np.load(bundle / "climb_bundle.npz")
            d = self.sim.data  # motion frame -> sim frame, from frame 0 (both on the start holds)
            R = quat_mat(d.qpos[3:7]) @ quat_mat(b["ref_trunk_quat"][0]).T
            p0 = d.qpos[0:3] - R @ b["ref_trunk_pos"][0]
            balls = p0 + b["ref_ball_pos"][start] @ R.T  # LIMBS order (ball_site_names)
            holds = self.sim.wall.holds
            sockets = np.array([h.socket for h in holds])
            near = np.linalg.norm(balls[:, None] - sockets[None], axis=-1).argmin(1)
            self.sim.reset_hanging({limb: holds[i] for limb, i in zip(LIMBS, near)})
        self.viewer = None

    def start(self) -> None:
        pass

    def read(self) -> dict:
        o = self.sim.observation()
        return {"q": o["q"], "dq": o["dq"], "imu_quat": o["quat"], "gyro": o["gyro"]}

    def send(self, targets: dict) -> None:
        self.sim.step([targets[j] for j in self.joints])  # one 20 ms tick
        if self.viewer is not None:
            self.viewer.sync()

    def stop(self) -> None:
        print(f"[SIM] balls hooked at the end: {self.sim.hooked()}")


def bundle_dir(arg: str) -> Path:
    p = Path(arg)
    if not (p / "policy.onnx").is_file():
        p = POLICIES / arg.rstrip("/").split("/")[-1]
    if not (p / "policy.onnx").is_file():
        sys.exit(f"no bundle {arg}: run `fetch` first (looked in {p})")
    return p


def run(args) -> None:
    bundle = bundle_dir(args.bundle)
    b = np.load(bundle / "climb_bundle.npz")
    kp = int(round(float(b["kp_fw"])))
    trunk_to_imu = quat_mat(b["imu_quat"]).T
    ctrl = ClimbController(bundle)  # adaptive phase on, min_rate from the bundle
    ctrl.max_rate = args.max_speed
    start = min(int(round(args.start_s / DT)), ctrl.T - 2)  # nominal 50 Hz frames
    four = np.flatnonzero(ctrl.ref_contact.all(1)[start:])  # all four balls on holds there
    if start > 0 and len(four):
        start += int(four[0])
    print(f"[RUN] starting at frame {start} ({start * DT:.2f} s), the robot on its holds there")
    names = ctrl.joint_names
    if args.sim:
        robot = Sim(bundle, kp, start)
        if args.view:
            import mujoco.viewer

            robot.viewer = mujoco.viewer.launch_passive(robot.sim.model, robot.sim.data)
    else:
        from robot_policy import LinkRobot  # the replay move over the dashboard link

        robot = LinkRobot(args.host, args.move, kp)
    idx = [robot.joints.index(n) for n in names]  # robot order -> policy order

    def meas(frame) -> dict:
        quat = np.asarray(frame["imu_quat"], float)
        return {
            "q": np.asarray(frame["q"], float)[idx],
            "dq": np.asarray(frame["dq"], float)[idx],
            "gyro": trunk_to_imu @ np.asarray(frame["gyro"], float),  # Microban: trunk frame
            "gravity": quat_mat(quat).T @ G_DOWN,
            "quat": quat,
        }

    def send(q_policy) -> None:
        robot.send(dict(zip(names, map(float, q_policy), strict=True)))

    print(f"[RUN] {bundle.name}: {str(b['checkpoint'])}, kp {kp}, {'sim' if args.sim else args.host}")
    robot.start()
    try:
        q0 = meas(robot.read())["q"]
        go = threading.Event()
        if args.sim:
            go.set()
        else:
            threading.Thread(target=lambda: (input(), go.set()), daemon=True).start()
            print(f"[RUN] ramping to the pose at {args.start_s:g} s in {args.ramp:g} s, then holding: Enter climbs")
        t0, ramp_ticks = time.monotonic(), int(args.ramp / DT)
        for k in range(10**9):  # ramp, then hold until Enter
            a = min(1.0, k / max(ramp_ticks, 1))
            robot.read()
            send((1 - a) * q0 + a * ctrl.ref_q[start])
            if a >= 1.0 and go.is_set():
                break
            time.sleep(0.0 if args.sim and not args.view else DT)
        m = meas(robot.read())
        ctrl.reset(m, start_frame=start)
        print(f"[RUN] climbing ({ctrl.T} frames), Ctrl+C stops")
        next_tick, t0 = time.perf_counter(), time.monotonic()
        for k in range(int(args.seconds / DT)):
            m = meas(robot.read())
            if not (np.isfinite(m["q"]).all() and np.isfinite(m["dq"]).all()):
                print("[RUN] non-finite measurement: holding")
                break
            targets = ctrl.tick(m)
            if not np.isfinite(targets).all():
                print("[RUN] non-finite targets: holding")
                break
            send(targets)
            if k % 250 == 0:
                print(f"[RUN] t {k * DT:5.1f} s, frame {ctrl.t}/{ctrl.T - 1}, v_ref {ctrl.v_ref:.2f}, speed {ctrl.speed:.2f}")
            if not args.sim or args.view:
                next_tick += DT
                time.sleep(max(0.0, next_tick - time.perf_counter()))
        if not args.sim:
            print("[RUN] holding the last targets: Ctrl+C ends")
            while True:
                robot.read()
                time.sleep(DT)
    except KeyboardInterrupt:
        print("[RUN] stopped")
    finally:
        robot.stop()


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download a W&B run's checkpoint and build its bundle")
    f.add_argument("run_path", help="entity/project/run_id")
    f.add_argument("--checkpoint", help="model_<N>.pt (default: the newest)")
    r = sub.add_parser("run", help="the policy on this laptop, the robot (or sim) over the link")
    r.add_argument("bundle", help="run id, run path or bundle folder")
    where = r.add_mutually_exclusive_group(required=True)
    where.add_argument("--host", help="the robot's host name or IP")
    where.add_argument("--sim", action="store_true", help="the organisers' sim plays the robot")
    r.add_argument("--view", action="store_true", help="with --sim: MuJoCo's viewer, real time")
    r.add_argument("--move", default="replay", help="the robot-side move")
    r.add_argument("--start-s", type=float, default=0.0, help="[s] into the reference to start at")
    r.add_argument("--ramp", type=float, default=3.0, help="[s] to the first pose")
    r.add_argument("--seconds", type=float, default=60.0, help="of climbing")
    r.add_argument(
        "--max-speed", type=float, default=0.8, help="cap on the reference speed (1 = nominal)"
    )
    args = p.parse_args()
    fetch(args) if args.cmd == "fetch" else run(args)


if __name__ == "__main__":
    main()
