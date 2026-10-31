"""Feedforward joint torques of a climbing trajectory: inverse dynamics with the limbs on their
holds. Laptop only (Pinocchio); robot_replay.py --feedforward sends them to the replay_ff move.

At each frame, with the frame's acceleration (central differences of the trajectory, so the one
that takes the robot to the next frame):

    M(q) a + h(q, v) = S^T tau + sum_c J_c(q)^T f_c

The 6 base rows fix the contact forces up to the null space of the contacts; we pick the forces
that need the least joint torque, min |tau|^2 + REG |f|^2 subject to the base rows. [Ours] Point
contacts at the balls, 3D forces, no friction cone or unilaterality: a hold can pull a ball, as
the sockets do within their play.

Contacts: the npz's `contact` / `contact_names` when given, else the balls that stay still
(STILL_SPEED). [Ours] The wall frame (z up, gravity -z); the base is the trajectory's free-flyer
(x y z, quaternion w x y z). Online (Feedforward.at with the IMU's trunk quaternion), the trunk
keeps its planned heading but takes the measured tilt, so the forces are split for the gravity the
robot really feels; the joints, velocities and accelerations stay the planned ones. [Ours]
"""

import numpy as np
import pinocchio as pin

from rcr.robot.hooks import LIMBS
from rcr.robot.microban import mjcf_path

REG = 1e-4  # [ (N.m)^2 / N^2 ] weight of the contact forces against the joint torques [Ours]
STILL_SPEED = (
    0.01  # [m/s] a ball slower than this is on its hold, when the npz has no contacts [Ours]
)
# [frames] moving average of the velocities before the accelerations: some trajectories chatter
# frame to frame (level3-2steps-highergains.npz: the left knee by +-20 deg, ~1000 rad/s^2) [Ours]
SMOOTH = 5
# [N.m] the XL330's torque at its 1.75 A current limit [Microban]; the torques are clipped to it
TAU_LIMIT = 0.605


class ClimbDynamics:
    """The climbing model (MJCF, with its armature) and the joint order of the robot's link."""

    def __init__(self, joints: list[str]):
        # its trunk_freejoint: q[0:7]
        self.model = pin.buildModelFromMJCF(str(mjcf_path(climber=True)))
        self.data = self.model.createData()
        self.joints = joints
        self.q_idx = np.array([self.model.idx_qs[self.model.getJointId(j)] for j in joints])
        self.v_idx = np.array([self.model.idx_vs[self.model.getJointId(j)] for j in joints])
        self.balls = [self.model.getFrameId(f"{limb}_ball") for limb in LIMBS]

    def configurations(self, base: np.ndarray, q: np.ndarray) -> np.ndarray:
        """(frames, nq) Pinocchio configurations of the base (frames, 7: x y z, w x y z) and the
        joints (frames, joints)."""
        out = np.tile(pin.neutral(self.model), (len(q), 1))
        out[:, 0:3] = base[:, 0:3]
        out[:, 3:7] = base[:, [4, 5, 6, 3]]  # Pinocchio's quaternion is x y z w
        out[:, self.q_idx] = q
        out[:, 3:7] /= np.linalg.norm(out[:, 3:7], axis=1, keepdims=True)
        return out

    def velocities(
        self, qs: np.ndarray, dt: float, smooth: int = SMOOTH
    ) -> tuple[np.ndarray, np.ndarray]:
        """(frames, nv) velocities and accelerations: central differences of the mid-frame
        velocities, those averaged over `smooth` frames first; one-sided at the ends."""
        n = len(qs)
        dq = np.array(
            [pin.difference(self.model, qs[i], qs[i + 1]) / dt for i in range(n - 1)]
        )  # mid-frame
        if smooth > 1:
            pad = np.pad(dq, ((smooth // 2, smooth - 1 - smooth // 2), (0, 0)), mode="edge")
            dq = np.array([np.convolve(c, np.ones(smooth) / smooth, mode="valid") for c in pad.T]).T
        v = np.zeros((n, self.model.nv))
        v[1:-1] = (dq[:-1] + dq[1:]) / 2
        v[0], v[-1] = dq[0], dq[-1]
        a = np.zeros_like(v)
        a[1:-1] = (dq[1:] - dq[:-1]) / dt
        return v, a

    def ball_speeds(self, qs: np.ndarray, dt: float) -> np.ndarray:
        """(frames, limbs) speed of each ball in the wall frame [m/s]."""
        pos = np.zeros((len(qs), len(self.balls), 3))
        for i, q in enumerate(qs):
            pin.framesForwardKinematics(self.model, self.data, q)
            pos[i] = [self.data.oMf[f].translation for f in self.balls]
        return np.linalg.norm(np.gradient(pos, dt, axis=0), axis=2)

    def torques(
        self, q: np.ndarray, v: np.ndarray, a: np.ndarray, contact: np.ndarray
    ) -> np.ndarray:
        """Joint torques [N.m] (the link's joint order) of one frame, contact (limbs,) bool."""
        m, d = self.model, self.data
        rhs = pin.rnea(m, d, q, v, a)  # M a + h, with gravity and the armature
        pin.computeJointJacobians(m, d, q)
        rows = [
            pin.getFrameJacobian(m, d, f, pin.LOCAL_WORLD_ALIGNED)[:3]
            for f, on in zip(self.balls, contact, strict=True)
            if on
        ]
        if not rows:  # free fall: the base rows cannot be met, give the joints' part alone
            return rhs[self.v_idx]
        J = np.vstack(rows)
        Jb, Jj = J[:, :6].T, J[:, self.v_idx].T  # (6, 3c), (joints, 3c)
        tj, tb = rhs[self.v_idx], rhs[:6]
        # min |tj - Jj f|^2 + REG |f|^2  s.t.  Jb f = tb  (KKT)
        nf = J.shape[0]
        H = Jj.T @ Jj + REG * np.eye(nf)
        K = np.block([[H, Jb.T], [Jb, np.zeros((6, 6))]])
        sol = np.linalg.lstsq(K, np.concatenate([Jj.T @ tj, tb]), rcond=None)[0]
        return tj - Jj @ sol[:nf]


def tilted(planned: np.ndarray, measured: np.ndarray) -> np.ndarray:
    """The trunk quaternion (x y z w, Pinocchio's) with the planned heading and the measured tilt:
    the planned orientation turned, in the trunk frame, by the least rotation that brings the
    planned gravity onto the measured one. The IMU's yaw drifts and is not the wall's; its gravity
    is right. measured: the IMU's trunk quaternion (w x y z); not finite: the planned one."""
    if not np.all(np.isfinite(measured)) or np.linalg.norm(measured) < 0.5:
        return planned
    down = np.array([0.0, 0.0, -1.0])
    rotation = pin.Quaternion(planned[3], *planned[:3]).normalized().matrix()
    g_plan = rotation.T @ down
    g_meas = pin.Quaternion(*measured).normalized().matrix().T @ down  # w x y z constructor
    turn = pin.Quaternion.FromTwoVectors(g_meas, g_plan).matrix()  # turn @ g_meas = g_plan
    return pin.Quaternion(rotation @ turn).normalized().coeffs()  # x y z w


class Feedforward:
    """A trajectory's feedforward torques, frame by frame: the configurations, velocities,
    accelerations and contacts computed once (for one replay speed), the torques per frame, with
    the trunk tilted as measured when given (the contact forces split for the real gravity)."""

    def __init__(
        self,
        dyn: ClimbDynamics,
        base: np.ndarray,
        q: np.ndarray,
        dt: float,
        contact: np.ndarray | None = None,
    ):
        """base (frames, 7: x y z, w x y z), q (frames, joints), sampled every dt [s] (as replayed:
        the replay speed already in dt). contact: (frames, limbs in rcr.robot.hooks.LIMBS order)
        bool, or None to find the still balls."""
        self.dyn = dyn
        self.qs = dyn.configurations(base, q)
        self.v, self.a = dyn.velocities(self.qs, dt)
        self.contact = dyn.ball_speeds(self.qs, dt) < STILL_SPEED if contact is None else contact

    def __len__(self) -> int:
        return len(self.qs)

    def at(self, i: int, trunk_quat: np.ndarray | None = None, scale: float = 1.0) -> np.ndarray:
        """Joint torques [N.m] of frame i, times scale, clipped to TAU_LIMIT. trunk_quat: the
        measured trunk orientation (w x y z, the robot's imu_quat), or None for the planned one."""
        q = self.qs[i]
        if trunk_quat is not None:
            q = q.copy()
            q[3:7] = tilted(q[3:7], np.asarray(trunk_quat, dtype=float))
        tau = self.dyn.torques(q, self.v[i], self.a[i], self.contact[i])
        return np.clip(scale * tau, -TAU_LIMIT, TAU_LIMIT)


def feedforward_torques(
    dyn: ClimbDynamics,
    base: np.ndarray,
    q: np.ndarray,
    dt: float,
    contact: np.ndarray | None = None,
) -> np.ndarray:
    """(frames, joints) feedforward torques [N.m] of the planned trajectory (see Feedforward)."""
    ff = Feedforward(dyn, base, q, dt, contact)
    return np.array([ff.at(i) for i in range(len(ff))])
