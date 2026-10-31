"""A policy trained in mjlab_microban (Mjlab-Climbing-Microban[-Adaptive-Phase]), run at 50 Hz
[Ours].

Pi-safe: numpy and onnxruntime only, nothing of mjlab. Its Microban move is
rcr/robot/climb_policy_move.py.

It reads a deployment bundle made on the laptop by mjlab_microban's
scripts/export_climbing_deploy.py
(``train_climbing.sh --deploy ...``), a folder with:
  - ``policy.onnx``: the actor (obs normaliser + MLP), obs -> actions;
  - ``climb_bundle.npz``: the planned climb (reference joint positions, trunk poses, ball positions,
    expected ball
    contacts), the actor's observation layout, the action scale, the adaptive phase parameters and
    the robot's
    kinematic tree (for the forward kinematics of the balls);
found at the path given, else $MICROBAN_CLIMB_BUNDLE, else ``~/climb_policy``. The export checks
this file against
the env tick by tick: change it together with the export (``BUNDLE_FORMAT``).

What the env computed from the simulator, this file computes from what the robot measures (encoders,
IMU):
  - the actor's observation, term by term, with the same proprioceptive history (oldest first,
    backfilled at the
    start), in the env's joint order;
  - the residual targets: reference joint positions of the frame the step ends on + action scale *
    action;
  - the adaptive phase (Sleiman et al., CoRL 2024): phase += v, v = max(v_ref + sigma * dv,
    min_rate), where v_ref
    measures how well the robot tracks the reference (balls, trunk position, trunk orientation). The
    env had the
    true trunk pose; here it is **estimated** [Ours]: the trunk orientation is the IMU's, its yaw
    aligned at the start
    so that the four balls sit on their start holds, and the trunk position is the one that puts the
    balls the plan
    has on holds at this frame (``ref_contact``) where the plan has them (forward kinematics from
    the encoders,
    least squares over those balls). A ball that slips off its hold is not seen by this estimate.
"""

import math
import os
from pathlib import Path

import numpy as np

BUNDLE_FORMAT = 1
PHASE_EPS = 1e-4  # as mjlab_microban.tasks.climbing.mdp._PHASE_EPS
F32 = np.float32
G_DOWN = np.array([0.0, 0.0, -1.0])


# --------------------------------------------------------------------------------------------------
# ------------------
# Rotations (quaternions w x y z)
# --------------------------------------------------------------------------------------------------
# ------------------


def quat_mat(q) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            # fmt: skip
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def mat_quat(R) -> np.ndarray:
    """Rotation matrix -> unit quaternion (w >= 0)."""
    t = np.trace(R)
    if t > 0:
        s = 2.0 * math.sqrt(t + 1.0)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * math.sqrt(max(1.0 + R[i, i] - R[j, j] - R[k, k], 1e-12))
        q = [0.0] * 4
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.array(q)
    return q * (1.0 if q[0] >= 0 else -1.0) / np.linalg.norm(q)


def axis_angle_mat(axis, angle) -> np.ndarray:
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    C = 1.0 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])  # fmt: skip


def rot_z(angle) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def yaw_of(R) -> float:
    return math.atan2(R[1, 0], R[0, 0])


def rotation_angle(Ra, Rb) -> float:
    """Angle of the rotation between two orientations (mjlab's quat_error_magnitude)."""
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return math.acos(min(1.0, max(-1.0, c)))


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


# --------------------------------------------------------------------------------------------------
# ------------------
# The bundle and the controller (no Microban, no simulator: also run against the env by the export's
# parity check)
# --------------------------------------------------------------------------------------------------
# ------------------


def find_bundle(explicit=None) -> Path | None:
    candidates = [explicit, os.environ.get("MICROBAN_CLIMB_BUNDLE"), Path.home() / "climb_policy"]
    for c in candidates:
        if c and Path(c).is_file():  # the bundle's climb_bundle.npz given: its folder
            c = Path(c).parent
        if c and (Path(c) / "climb_bundle.npz").is_file() and (Path(c) / "policy.onnx").is_file():
            return Path(c)
    return None


class Kinematics:
    """Forward kinematics of the robot's bodies in the trunk frame (MuJoCo's convention, from the
    env's model)."""

    def __init__(self, b) -> None:
        self.parent = b["kin_parent"].astype(int)
        self.pos, self.quat = b["kin_body_pos"], b["kin_body_quat"]
        self.joint = b["kin_joint"].astype(
            int
        )  # index in the env's joint order, -1: fixed to its parent
        self.axis, self.jpos, self.qpos0 = (
            b["kin_joint_axis"],
            b["kin_joint_pos"],
            b["kin_joint_qpos0"],
        )
        self.R_local = [quat_mat(q) for q in self.quat]
        self.site_body = b["ball_site_body"].astype(int)
        self.site_pos = b["ball_site_pos"]

    def balls(self, q) -> np.ndarray:
        """Ball centres in the trunk frame, [N, 3]."""
        n = len(self.parent)
        Rs, ps = [None] * n, [None] * n
        for i in range(n):  # parents first
            if self.parent[i] < 0:
                Rs[i], ps[i] = np.eye(3), np.zeros(3)
                continue
            Rp, pp = Rs[self.parent[i]], ps[self.parent[i]]
            R, p = Rp @ self.R_local[i], pp + Rp @ self.pos[i]
            j = self.joint[i]
            if j >= 0:
                anchor = p + R @ self.jpos[i]
                R = R @ axis_angle_mat(self.axis[i], q[j] - self.qpos0[i])
                p = anchor - R @ self.jpos[i]
            Rs[i], ps[i] = R, p
        return np.array(
            [ps[b] + Rs[b] @ s for b, s in zip(self.site_body, self.site_pos, strict=True)]
        )


class ClimbController:
    """The policy, its observation and the reference phase, from the robot's measurements; one
    ``tick`` every 20 ms.

    ``meas``: q, dq (env joint order, rad, rad/s), gyro (the IMU's angular velocity, IMU frame),
    gravity (unit, trunk
    frame), quat (trunk orientation w x y z, any yaw reference). ``tick`` returns the 19 joint
    targets (env order).
    """

    def __init__(self, bundle_dir, min_phase_rate=None, adaptive=True, session=None) -> None:
        """session: an onnxruntime-like session instead of the bundle's policy.onnx (tests)."""
        bundle_dir = Path(bundle_dir)
        b = dict(np.load(bundle_dir / "climb_bundle.npz", allow_pickle=False))
        if int(b["format"]) != BUNDLE_FORMAT:
            raise ValueError(
                f"bundle format {int(b['format'])}, this move reads {BUNDLE_FORMAT}: export it "
                "again"
            )
        self.b = b
        if session is None:  # onnxruntime: on the Pi, and in this repo's rl group
            import onnxruntime as ort

            if hasattr(ort, "disable_telemetry_events"):  # no network from the control loop
                ort.disable_telemetry_events()
        self.session = session or ort.InferenceSession(
            str(bundle_dir / "policy.onnx"), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.joint_names = [str(n) for n in b["joint_names"]]
        self.default = b["default_joint_pos"]
        self.scale = b["action_scale"]
        self.clip = float(b["clip_actions"])  # NaN: none
        self.num_actions = int(b["num_actions"])
        self.terms = [str(t) for t in b["obs_terms"]]
        self.term_dims = b["obs_dims"].astype(int)
        self.term_hist = b["obs_history"].astype(int)
        self.future_offsets = b["future_offsets"].astype(int)
        self.ref_q = b["ref_joint_pos"]
        self.ref_trunk_pos, self.ref_trunk_R = (
            b["ref_trunk_pos"],
            np.array([quat_mat(q) for q in b["ref_trunk_quat"]]),
        )
        self.ref_balls, self.ref_contact = b["ref_ball_pos"], b["ref_contact"].astype(bool)
        self.T = len(self.ref_q)
        self.R_imu = quat_mat(b["imu_quat"])  # IMU site in the trunk
        self.kin = Kinematics(b)
        self.has_phase = bool(b["phase_enabled"])
        self.sigma, self.k = float(b["phase_sigma"]), float(b["phase_k"])
        self.delta_clip = float(b["phase_delta_clip"])  # NaN: none
        self.rate_terms = [str(t) for t in b["phase_rate_terms"]]
        self.tolerances = dict(
            zip(self.rate_terms, b["phase_tolerances"].astype(float), strict=True)
        )
        bundle_min = float(b["phase_min_rate"])
        self.min_rate = bundle_min if min_phase_rate is None else min_phase_rate
        self.min_rate = (
            None if self.min_rate is None or math.isnan(self.min_rate) else self.min_rate
        )
        self.adaptive = adaptive
        self.max_rate = None  # [Ours] cap on the reference speed (frames / tick), None: none
        self.obs_size = int(sum(self.term_dims * np.maximum(self.term_hist, 1)))
        expected = self.session.get_inputs()[0].shape[-1]
        if isinstance(expected, int) and expected != self.obs_size:
            raise ValueError(
                f"policy.onnx takes {expected} inputs, the bundle's layout makes {self.obs_size}"
            )

    # -- the reference phase
    # ---------------------------------------------------------------------------------------

    def reset(self, meas, v_ref=None, start_frame: int = 0) -> None:
        """The start of the climb (the env's reset): frame 0, history backfilled, last action 0. As
        the env's reset,
        the reference then advances once (dv = 0): the first observation is about one frame in."""
        # start_frame [Ours]: the climb from that reference frame (the robot in its pose)
        start_frame = int(min(max(start_frame, 0), self.T - 2))
        self.t, self.phase, self.speed, self.v_ref = start_frame, float(start_frame), 1.0, 1.0
        self.last_action = np.zeros(self.num_actions)
        self.history: dict[str, list] = {}
        self.ticks = 0
        self.finished = False
        R = quat_mat(meas["quat"])
        self.yaw_align = self._yaw_alignment(meas["q"], R, self.t)
        self.yaw_windup, self.prev_yaw = 0.0, yaw_of(R)
        self.est = None
        self._advance(0.0, self.rate_ref(meas, self.t) if v_ref is None else v_ref)

    # The phase arithmetic is float32, as the env's (its floor() lands on the same frames).

    def frames_ahead(self, offsets) -> np.ndarray:
        frac = F32(self.phase) - F32(self.t)
        if not (-PHASE_EPS <= frac < 1.0):
            frac = F32(0.0)
        x = frac + F32(self.speed) * np.asarray(offsets, np.float32) + F32(PHASE_EPS)
        idx = self.t + np.floor(x.astype(np.float32)).astype(int)
        return np.clip(idx, 0, self.T - 1)

    def _advance(self, dv: float, v_ref: float) -> None:
        """The motion command's update after a step: v from v_ref and the policy's dv, then the
        phase."""
        self.v_ref = v_ref
        if self.has_phase:
            if not math.isnan(self.delta_clip):
                dv = min(max(dv, -self.delta_clip), self.delta_clip)
            v = F32(v_ref) + F32(self.sigma) * F32(dv)
            if self.min_rate is not None:
                v = max(v, F32(self.min_rate))
            if self.max_rate is not None:
                v = min(v, F32(self.max_rate))
            self.speed = float(max(v, -F32(self.phase)))  # not before the first frame
        frac = F32(self.phase) - F32(self.t)
        if not (-PHASE_EPS <= frac < 1.0):
            self.phase = float(self.t)
        self.phase = float(F32(self.phase) + F32(self.speed))
        self.t = int(math.floor(F32(self.phase) + F32(PHASE_EPS)))
        if (
            self.t >= self.T - 1
        ):  # the end of the climb: stay on the last frame (the env restarts it)
            self.t, self.phase, self.finished = self.T - 1, float(self.T - 1), True

    # -- the state estimate and v_ref
    # ------------------------------------------------------------------------------

    def _yaw_alignment(self, q, R_imu_w, frame: int = 0) -> float:
        """Yaw to add to the IMU's orientation so that the four balls best match the climb's first
        frame (Procrustes
        about z), where the robot starts: hanging on the start holds."""
        a = (R_imu_w @ self.kin.balls(q).T).T
        b = self.ref_balls[frame]
        a, b = a - a.mean(0), b - b.mean(0)
        return math.atan2(
            float(np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])),
            float(np.sum(a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1])),
        )

    def estimate(self, meas, frame: int) -> dict:
        """Trunk pose and ball positions in the motion frame: IMU orientation (yaw aligned),
        position from the balls
        the plan has on holds at ``frame``."""
        R = rot_z(self.yaw_align) @ quat_mat(meas["quat"])
        balls_b = self.kin.balls(meas["q"])
        stance = self.ref_contact[frame]
        if stance.any():
            pos = np.mean(self.ref_balls[frame][stance] - (R @ balls_b[stance].T).T, axis=0)
        else:
            pos = self.ref_trunk_pos[frame].copy()
        return {"trunk_pos": pos, "trunk_R": R, "balls": pos + (R @ balls_b.T).T, "stance": stance}

    def rate_ref(self, meas, frame: int, state=None) -> float:
        """v_ref = prod_i min(k ** (1 - e_i / tol_i^2), 1) against the reference ``frame``.
        ``state``: the trunk pose
        and balls (estimate() by default; the parity check gives the simulator's)."""
        if not self.has_phase:
            return 1.0
        if not self.adaptive:
            return 1.0
        s = self.estimate(meas, frame) if state is None else state
        self.est = s
        errors = {}
        if "balls" in self.rate_terms:
            errors["balls"] = float(
                np.mean(np.sum((self.ref_balls[frame] - s["balls"]) ** 2, axis=-1))
            )
        if "trunk_pos" in self.rate_terms:
            errors["trunk_pos"] = float(np.sum((self.ref_trunk_pos[frame] - s["trunk_pos"]) ** 2))
        if "trunk_ori" in self.rate_terms:
            errors["trunk_ori"] = rotation_angle(self.ref_trunk_R[frame], s["trunk_R"]) ** 2
        if "joint_pos" in self.rate_terms:
            errors["joint_pos"] = float(np.mean((self.ref_q[frame] - meas["q"]) ** 2))
        v, log_k = 1.0, math.log(self.k)
        for name, e in errors.items():
            v *= min(math.exp(log_k * (1.0 - e / self.tolerances[name] ** 2)), 1.0)
        return 0.0 if math.isnan(v) else v

    # -- the observation and the policy
    # ----------------------------------------------------------------------------

    def _term(self, name: str, meas) -> np.ndarray:
        if name == "base_ang_vel":
            return meas["gyro"]
        if name == "projected_gravity":
            return meas["gravity"]
        if name == "base_yaw":
            return np.array([math.cos(self.yaw_windup), math.sin(self.yaw_windup)])
        if name == "joint_pos":
            return meas["q"] - self.default
        if name == "joint_vel":
            return meas["dq"]
        if name == "actions":
            return self.last_action
        if name == "command":
            return (self.ref_q[self.frames_ahead(self.future_offsets)] - self.default).reshape(-1)
        if name == "phase":
            return np.array([self.phase / max(self.T - 1.0, 1.0), self.v_ref])
        raise ValueError(f"unknown observation term {name!r}")

    def observation(self, meas) -> np.ndarray:
        parts = []
        for name, dim, hist in zip(self.terms, self.term_dims, self.term_hist, strict=True):
            x = np.asarray(self._term(name, meas), float).reshape(-1)
            assert x.shape == (dim,), f"{name}: {x.shape} != ({dim},)"
            if hist > 0:
                buf = self.history.setdefault(name, [x] * hist)  # backfilled on the first append
                if self.ticks > 0:
                    buf.append(x)
                    del buf[0]
                parts.extend(buf)
            else:
                parts.append(x)
        return np.concatenate(parts)

    def tick(self, meas, v_ref=None, state=None) -> np.ndarray:
        """One 20 ms step: the reference advances (from the second tick on, with the last action's
        dv), the
        observation is built, the policy runs, and the joint targets are returned. ``v_ref`` /
        ``state``: given by
        the parity check instead of the estimate."""
        if self.ticks > 0:
            if v_ref is None:
                v_ref = self.rate_ref(meas, self.t, state)
            dv = float(self.last_action[-1]) if self.has_phase else 0.0
            self._advance(dv, v_ref)
        yaw = yaw_of(quat_mat(meas["quat"]))
        if math.isfinite(yaw):
            self.yaw_windup += wrap(yaw - self.prev_yaw)
            self.prev_yaw = yaw
        obs = self.observation(meas).astype(np.float32)
        action = self.session.run(None, {self.input_name: obs[None]})[0][0].astype(float)
        if not math.isnan(self.clip):
            action = np.clip(action, -self.clip, self.clip)
        self.last_action = action
        self.obs = obs
        self.ticks += 1
        ref = self.ref_q[self.frames_ahead([1])[0]]
        return ref + self.scale * action[: len(self.scale)]
