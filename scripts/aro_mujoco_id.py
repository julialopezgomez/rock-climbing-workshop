#!/usr/bin/env python3
"""ARO-style piecewise cubic joint Bezier + inverse-dynamics control.

Uses the actual kinematics/new ik.py to generate a 50 Hz IK configuration path.
The joint path is converted to ARO-style cubic Bezier segments with four
consecutive control points, overlapping by one endpoint (stride of three).

The organisers' ClimbingSim.step() accepts position targets, NOT torques.
This experiment uses its MuJoCo model and steps physics directly, bypassing
the BAM firmware/servo loop. It is simulation-only, not a real-robot command.

Run from the repository root:
    uv run python scripts/aro_mujoco_id.py --no-viewer
    uv run python scripts/aro_mujoco_id.py
On macOS, use: uv run mjpython scripts/aro_mujoco_id.py
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np

from climbing_sim import ClimbingSim, Wall
from climbing_sim.scene import BALL_SITES, JOINTS
from rcr import io
from rcr.kinematics import ik
from rcr.kinematics.bezier import Bezier
from rcr.types import Stance

# ARO control structure: ddq_star = ddq_d + Kp*(q_d-q) + Kv*(dq_d-dq).
KP = 300.0
KV = 2.0 * np.sqrt(KP)

# Small socket preload. +Y points from the robot towards the wall.
CONTACT_KP = 15.0       # N/m: correct displacement of a held ball
CONTACT_KD = 1.0        # Ns/m: damp motion of a held ball
NORMAL_PRELOAD = 0.10   # N: small additional push towards the socket
MAX_CONTACT_FORCE = 0.50  # N: cap each added Cartesian force

# EXPERIMENTAL simulation safety cap, NOT a verified Microban actuator rating.
MAX_MOTOR_TORQUE = 0.15  # Nm
STOP_IF_HELD_ERROR = 0.025  # m; stop if a supporting ball has drifted too far


def simulator_to_pin_q(sim: ClimbingSim) -> np.ndarray:
    """Return Pinocchio q (xyzw floating quaternion) from current MuJoCo qpos."""
    q = np.zeros(ik.model.nq)
    q[:3] = sim.data.qpos[:3]
    q[3:7] = sim.data.qpos[[4, 5, 6, 3]]  # MuJoCo wxyz -> Pin xyzw
    for name in JOINTS:
        q[ik.model.idx_qs[ik.model.getJointId(name)]] = (
            sim.data.qpos[sim.model.joint(name).qposadr]
        )
    return q


def aro_joint_bezier(qs: np.ndarray, duration: float):
    """The *actual ARO maketraj construction*, adapted for arbitrary path length.

    Each four consecutive joint samples are cubic Bezier CONTROL POINTS:
        [q0, q1, q2, q3], [q3, q4, q5, q6], ...
    For a final remainder shorter than three sample intervals, interpolate
    two interior control points so the last segment ends at the last sample.

    Time is allocated proportionally to the covered path sample intervals.
    Returns (q_d(t), dq_d(t), ddq_d(t)), as in ARO control.py.

    As in the original ARO algorithm, position is C0 at joins; velocity and
    acceleration are generally discontinuous across those joins.
    """
    qs = np.asarray(qs, dtype=float)
    if qs.ndim != 2 or qs.shape[0] < 2 or qs.shape[1] != ik.model.nq:
        raise ValueError("Expected at least two full Pinocchio configurations")
    if duration <= 0:
        raise ValueError("Bezier duration must be positive")

    # Match the servo actuator order, not an assumed Pinocchio joint order.
    pin_indices = [ik.model.idx_qs[ik.model.getJointId(name)] for name in JOINTS]
    path = qs[:, pin_indices]
    n_intervals = len(path) - 1

    curves = []
    ends = []
    starts = []
    for start in range(0, n_intervals, 3):
        end = min(start + 3, n_intervals)
        if end - start == 3:
            control = [path[k].copy() for k in range(start, end + 1)]
        else:
            # Preserve the final endpoint for a short remaining interval.
            delta = path[end] - path[start]
            control = [
                path[start].copy(),
                path[start] + delta / 3.0,
                path[start] + 2.0 * delta / 3.0,
                path[end].copy(),
            ]
        t0 = duration * start / n_intervals
        t1 = duration * end / n_intervals
        curves.append(Bezier(control, t_min=t0, t_max=t1))
        starts.append(t0)
        ends.append(t1)

    v_curves = [curve.derivative(1) for curve in curves]
    a_curves = [curve.derivative(2) for curve in curves]

    def evaluate(t, family):
        t = float(np.clip(t, 0.0, duration))
        index = min(int(np.searchsorted(ends, t, side="left")), len(curves) - 1)
        return family[index](t)

    return (
        lambda t: evaluate(t, curves),
        lambda t: evaluate(t, v_curves),
        lambda t: evaluate(t, a_curves),
    )


class AROController:
    """Torque-level ARO control structure on the organisers' MuJoCo model."""

    def __init__(self, sim: ClimbingSim, trajs, held_sockets, preload: float):
        self.sim = sim
        self.trajs = trajs
        self.held_sockets = held_sockets
        self.preload = float(preload)
        m = sim.model

        self.dof_indices = np.array(
            [m.jnt_dofadr[m.joint(name).id] for name in JOINTS], dtype=int
        )
        self.qpos_indices = np.array(
            [m.jnt_qposadr[m.joint(name).id] for name in JOINTS], dtype=int
        )
        actuator_ids = []
        for name in JOINTS:
            joint_id = m.joint(name).id
            candidates = np.where(m.actuator_trnid[:, 0] == joint_id)[0]
            if len(candidates) != 1:
                raise ValueError(f"Expected one motor actuator for {name}; found {len(candidates)}")
            actuator_ids.append(int(candidates[0]))
        self.actuator_ids = np.array(actuator_ids, dtype=int)
        self.gear = m.actuator_gear[self.actuator_ids, 0].copy()
        if np.any(np.abs(self.gear) < 1e-12):
            raise ValueError("A motor actuator has zero gear")

        self.M = np.empty((m.nv, m.nv))
        self.jac = np.zeros((3, m.nv))

    def step(self, t: float):
        """Compute torques, add held-ball J^T f, then step one MuJoCo physics tick."""
        sim = self.sim
        m, d = sim.model, sim.data
        mujoco.mj_forward(m, d)

        qd, dqd, ddqd = (f(t) for f in self.trajs)
        q = d.qpos[self.qpos_indices]
        dq = d.qvel[self.dof_indices]

        ddq_star = ddqd + KP * (qd - q) + KV * (dqd - dq)

        # MuJoCo equivalents of pin.crba and pin.nle on the SAME simulation model:
        # mass M(q), bias h(q,dq). The 6-DoF floating base remains unactuated.
        mujoco.mj_fullM(m, self.M, d.qM)
        ddq_full = np.zeros(m.nv)
        ddq_full[self.dof_indices] = ddq_star
        tau_full = self.M @ ddq_full + d.qfrc_bias
        tau = tau_full[self.dof_indices].copy()

        max_error = 0.0
        for limb, target in self.held_sockets.items():
            site_id = m.site(BALL_SITES[limb]).id
            p = d.site_xpos[site_id]
            mujoco.mj_jacSite(m, d, self.jac, None, site_id)
            v_ball = self.jac @ d.qvel

            error = target - p
            max_error = max(max_error, float(np.linalg.norm(error)))
            # Cartesian *force*: no moment at the ball, just as requested.
            f = CONTACT_KP * error - CONTACT_KD * v_ball
            f += np.array([0.0, self.preload, 0.0])
            f_norm = float(np.linalg.norm(f))
            if f_norm > MAX_CONTACT_FORCE:
                f *= MAX_CONTACT_FORCE / f_norm

            # J^T f contributes commanded actuator torques only.
            tau += self.jac[:, self.dof_indices].T @ f

        tau = np.clip(tau, -MAX_MOTOR_TORQUE, MAX_MOTOR_TORQUE)
        d.ctrl[:] = 0.0
        d.ctrl[self.actuator_ids] = tau / self.gear
        mujoco.mj_step(m, d)
        return max_error, float(np.max(np.abs(tau)))


def run(wall_name="N1", moving="right_hand", destination="c29_r27",
        method="qp", viewer=True, preload=NORMAL_PRELOAD):
    """Plan one move using the branch's IK and track it with ARO ID in MuJoCo."""
    if moving not in BALL_SITES:
        raise ValueError(f"Unknown moving limb: {moving}")
    sim = ClimbingSim(Wall.load(wall_name), cameras=[])
    try:
        sim.reset_hanging()
        wall = io.load_wall(Path("data/examples") / f"{wall_name}.json")
        a = wall.start
        contacts = dict(a.contacts)
        contacts[moving] = destination
        b = Stance(contacts=contacts)

        # Warm-start the *existing branch* IK from the settled MuJoCo state.
        q0 = simulator_to_pin_q(sim)
        qs = ik.ik_traj(wall, a, b, q0, method=method)
        duration = (len(qs) - 1) * ik.DT
        trajs = aro_joint_bezier(qs, duration)

        # A held limb is defined by the actual simulator socket, not an
        # arbitrary initial ball position or an ARO cube grasp.
        sockets = {
            limb: np.asarray(sim.wall.start[limb].socket, dtype=float)
            for limb in BALL_SITES
            if limb != moving
        }
        controller = AROController(sim, trajs, sockets, preload)
        initial_hooked = sim.hooked()
        print("Initial hooked:", initial_hooked)
        print(f"Following {len(qs)} IK samples over {duration:.3f} s")
        print("Servo loop: bypassed (direct torque experiment, simulation only)")

        v = sim.view() if viewer else None
        dt_phys = float(sim.model.opt.timestep)
        steps = int(round(duration / dt_phys))
        every_control = max(1, int(round(sim.control_dt / dt_phys)))
        worst_error = 0.0
        saturated = 0
        for step in range(steps):
            if v is not None and not v.is_running():
                break
            t = min(step * dt_phys, duration)
            error, peak = controller.step(t)
            worst_error = max(worst_error, error)
            saturated += int(peak >= MAX_MOTOR_TORQUE - 1e-10)
            if error > STOP_IF_HELD_ERROR:
                print(f"STOP: held-ball error {1000*error:.1f} mm")
                break
            if (step + 1) % every_control == 0:
                if v is not None:
                    sim._sync_viewer()
                if (step + 1) % (10 * every_control) == 0:
                    print(
                        f"t={sim.data.time:.2f}s  held error={1000*error:.1f} mm  "
                        f"hooked={sim.hooked()}"
                    )

        print(f"Maximum held-ball error: {1000*worst_error:.1f} mm")
        print(f"Motor saturation: {saturated} physics steps")
        print("Final hooked:", sim.hooked())
    finally:
        sim.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wall", default="N1")
    parser.add_argument("--moving", default="right_hand")
    parser.add_argument("--destination", default="c29_r27")
    parser.add_argument("--method", choices=("qp", "pinv"), default="qp")
    parser.add_argument("--preload", type=float, default=NORMAL_PRELOAD)
    parser.add_argument("--no-viewer", action="store_true")
    args = parser.parse_args()
    run(
        wall_name=args.wall,
        moving=args.moving,
        destination=args.destination,
        method=args.method,
        viewer=not args.no_viewer,
        preload=args.preload,
    )
