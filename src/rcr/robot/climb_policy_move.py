"""The Microban move of rcr.controller.mjlab_policy [Ours].

Run it with Microban's ``make run HOST=<robot> MOVES=<repo>/src/rcr/robot/climb_policy_move.py``.

Microban's make run sends this one file (as ~/microban/user_moves.py): our code comes from ~/rcr/src
on the Pi
(scripts/link_pi.py deploy), or from this repository's src/ on the laptop (make sim,
climbing_sim.run_move).
The bundle (mjlab_microban's train_climbing.sh --deploy) is found as
rcr.controller.mjlab_policy.find_bundle says.

Sequence: on start, the joints ramp from where they are to the climb's first frame (``ramp_s``),
hold it
(``start_delay_s``), then the policy climbs. At the end of the climb the policy keeps tracking the
last frame until the
move is stopped. Stopped, it holds its last targets ``stop_hold_s`` then ends (Microban turns the
torque off: the
robot hangs on its rope). If something goes wrong (non-finite measurement or action, the trunk
tilted past
``max_tilt_deg``), the climb is aborted: the robot holds the joint positions it measured then, until
stopped.
Every tick of the climb is logged to ``~/climb_logs/<time>.npz`` when the move stops (``save_log``).
"""

import math
import sys
import time
from pathlib import Path

import numpy as np

for _src in (Path(__file__).resolve().parents[2], Path.home() / "rcr" / "src"):
    if (_src / "rcr" / "controller" / "mjlab_policy.py").is_file() and str(_src) not in sys.path:
        sys.path.insert(0, str(_src))
        break

from rcr.controller.mjlab_policy import G_DOWN, ClimbController, find_bundle, quat_mat  # noqa: E402


def _smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


class ClimbPolicyMove:
    """Climb with the trained RL policy (Mjlab-Climbing-Microban, adaptive phase), from the climb's
    start holds."""

    description = "Climb (learned policy): ramp to the climb's first pose, then the policy climbs"
    variants = ("climber",)
    start_pose = "current"  # torque on where the robot is: it is on the wall
    # the BAM firmware P gain the policy was trained with (overwritten by the bundle's kp_fw)
    kp = 120
    ki = 0
    kd = 0
    options = {
        "ramp_s": 2.0,
        "start_delay_s": 2.0,
        "stop_hold_s": 5.0,
        "adaptive_phase": True,
        "min_phase_rate": 0.1,
        "max_tilt_deg": 60.0,
        "save_log": True,
    }
    option_docs = {
        "ramp_s": "s, from the current pose to the climb's first frame",
        "start_delay_s": "s, holding the first frame before the policy starts",
        "stop_hold_s": "s, holding the last targets once stopped (then the torque goes off)",
        "adaptive_phase": "the reference waits for the robot (v_ref estimated); off: nominal speed",
        "min_phase_rate": "the reference advances at least at this fraction of its speed",
        "max_tilt_deg": "trunk tilt from the vertical that aborts the climb (holds where it is)",
        "save_log": "every tick to ~/climb_logs/<time>.npz when it stops",
    }

    def __init__(self) -> None:
        self.bundle_dir = find_bundle()
        if self.bundle_dir is None:
            print(
                "[climb_policy] no bundle (climb_bundle.npz + policy.onnx): set "
                "MICROBAN_CLIMB_BUNDLE, or copy it to "
                "~/climb_policy (train_climbing.sh --deploy --host ...). The move is not listed."
            )
            self.variants = ()
            return
        b = np.load(self.bundle_dir / "climb_bundle.npz", allow_pickle=False)
        self.kp = int(round(float(b["kp_fw"])))
        self.options = dict(
            self.options,
            min_phase_rate=float(b["phase_min_rate"])
            if not math.isnan(float(b["phase_min_rate"]))
            else 0.0,
        )
        self._joints = [str(n) for n in b["joint_names"]]
        self.description += (
            f" [{str(b['task'])}, {Path(str(b['checkpoint'])).name}, wall {str(b['wall'])}]"
        )
        self._ctrl = ClimbController(
            self.bundle_dir
        )  # loads the network once, at the program's start
        R_imu = quat_mat(b["imu_quat"])
        self._trunk_to_imu = R_imu.T
        print(f"[climb_policy] bundle {self.bundle_dir}: {self.description}")

    # -- Microban's observation -> the controller's measurements (env joint order)
    # ---------------------------------

    def _meas(self, obs) -> dict:
        from_joints = self._index
        gravity = np.asarray(obs["gravity"], float)
        quat = np.asarray(obs["quat"], float)
        if not np.isfinite(quat).all():
            quat = self._last_quat
        if not np.isfinite(gravity).all():
            gravity = quat_mat(quat).T @ G_DOWN
        self._last_quat = quat
        return {
            "q": np.asarray(obs["q"], float)[from_joints],
            "dq": np.asarray(obs["dq"], float)[from_joints],
            "gyro": self._trunk_to_imu
            @ np.asarray(obs["gyro"], float),  # Microban gives it in the trunk frame
            "gravity": gravity,
            "quat": quat,
        }

    def _targets(self, targets_env) -> dict:
        return dict(zip(self._joints, np.asarray(targets_env, float).tolist(), strict=True))

    # -- the move ----------------------------------------------------------------------------------
    # ----------------

    def start(self, obs) -> None:
        if self.bundle_dir is None:
            raise RuntimeError(
                "no climb bundle on this machine (~/climb_policy): train_climbing.sh --deploy "
                "--host"
            )
        self._robot_joints = list(getattr(self, "_robot_joints", None) or _microban_joints(obs))
        self._index = [self._robot_joints.index(n) for n in self._joints]
        self._last_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self._t0 = obs["time"]
        self._q0 = np.asarray(obs["q"], float)[self._index]
        self._first = self._ctrl.ref_q[0]
        self._stage, self._stage_t0 = "ramp", obs["time"]
        self._last = self._q0.copy()
        self._log: list[dict] = []
        self._reason = ""
        self._stop_t0 = None

    def step(self, obs):
        now = obs["time"]
        meas = self._meas(obs)
        opt = self.options
        if self._stage in ("ramp", "wait"):
            if self._stage == "ramp":
                a = _smoothstep((now - self._stage_t0) / max(float(opt["ramp_s"]), 1e-3))
                self._last = (1.0 - a) * self._q0 + a * self._first
                if a >= 1.0:
                    self._stage, self._stage_t0 = "wait", now
            elif now - self._stage_t0 >= float(opt["start_delay_s"]):
                self._ctrl.adaptive = bool(opt["adaptive_phase"])
                self._ctrl.min_rate = float(opt["min_phase_rate"])
                self._ctrl.reset(meas)
                self._stage, self._stage_t0 = "climb", now
                print(
                    f"[climb_policy] climbing ({self._ctrl.T} frames, {self._ctrl.T * 0.02:.1f} s "
                    "at nominal speed)"
                )
            if self._stage != "climb":
                return self._targets(self._last)

        if self._stage == "climb":
            reason = self._check(meas)
            if not reason:
                t_start = time.perf_counter()
                try:
                    targets = self._ctrl.tick(meas)
                except Exception as e:  # noqa: BLE001
                    targets, reason = None, f"policy failed: {e!r}"
                dt = time.perf_counter() - t_start
                if targets is not None and not np.isfinite(targets).all():
                    reason = "non-finite targets"
                if not reason:
                    self._last = targets
                    self._record(now, meas, targets, dt)
                    if dt > 0.015:
                        print(f"[climb_policy] slow tick: {dt * 1e3:.1f} ms")
            if reason:
                self._abort(reason, meas)
        return self._targets(self._last)

    def _check(self, meas) -> str:
        if not (np.isfinite(meas["q"]).all() and np.isfinite(meas["dq"]).all()):
            return "non-finite joint measurement"
        g = meas["gravity"]
        tilt = math.degrees(
            math.acos(min(1.0, max(-1.0, -float(g[2]) / max(np.linalg.norm(g), 1e-9))))
        )
        if tilt > float(self.options["max_tilt_deg"]):
            return f"trunk tilted {tilt:.0f} deg"
        return ""

    def _abort(self, reason: str, meas) -> None:
        self._reason = reason
        self._stage = "aborted"
        if np.isfinite(meas["q"]).all():
            self._last = meas["q"].copy()
        print(
            f"[climb_policy] climb aborted at frame {self._ctrl.t}: {reason}. Holding; stop the "
            "move to end."
        )

    def _record(self, now, meas, targets, dt) -> None:
        c = self._ctrl
        if c.finished and not getattr(self, "_said_end", False):
            self._said_end = True
            print("[climb_policy] end of the climb: tracking its last frame until stopped")
        if not self.options["save_log"]:
            return
        est = c.est or {}
        self._log.append(
            {
                "time": now,
                "q": meas["q"],
                "dq": meas["dq"],
                "gyro": meas["gyro"],
                "gravity": meas["gravity"],
                "quat": meas["quat"],
                "targets": targets,
                "action": c.last_action,
                "frame": c.t,
                "phase": c.phase,
                "v_ref": c.v_ref,
                "speed": c.speed,
                "tick_s": dt,
                "trunk_pos_est": est.get("trunk_pos", np.full(3, np.nan)),
            }
        )

    def stopping(self, obs):
        if self._stop_t0 is None:
            self._stop_t0 = obs["time"]
            self._save_log()
        if obs["time"] - self._stop_t0 < float(self.options["stop_hold_s"]):
            return self._targets(self._last)
        return None

    def _save_log(self) -> None:
        if not self._log:
            return
        out = Path.home() / "climb_logs"
        try:
            out.mkdir(exist_ok=True)
            path = out / time.strftime("climb_%Y%m%d_%H%M%S.npz")
            np.savez(
                path,
                joint_names=np.array(self._joints),
                reason=self._reason,
                **{k: np.array([r[k] for r in self._log]) for k in self._log[0]},
            )
            print(f"[climb_policy] log: {path} ({len(self._log)} ticks)")
        except OSError as e:
            print(f"[climb_policy] log not saved: {e!r}")
        self._log = []


def _microban_joints(obs) -> list[str]:
    """The joint order of Microban's observation (microban_model.JOINTS, on the robot and in the
    simulators)."""
    import microban_model

    joints = list(microban_model.JOINTS)
    assert len(joints) == len(obs["q"]), (
        f"{len(joints)} joints in microban_model, {len(obs['q'])} in obs['q']"
    )
    return joints
