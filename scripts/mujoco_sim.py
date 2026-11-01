
"""ARO-style Bezier trajectory + inverse-dynamics control in MuJoCo.

Uses the organisers' ClimbingSim physics directly, bypassing the
built-in position-controlled servos.

Run:
    uv run python scripts/mujoco_sim.py

On macOS with the MuJoCo viewer, use mjpython.
"""

from pathlib import Path

import mujoco
import numpy as np

from climbing_sim import ClimbingSim, Wall
from climbing_sim.scene import BALL_SITES, JOINTS

from rcr import io
from rcr.kinematics import ik
from rcr.kinematics.bezier import Bezier, connecting
from rcr.types import LIMBS, Stance


# ------------------------------------------------------------
# Experiment
# ------------------------------------------------------------

WALL_NAME = "N1"
MOVING = "right_hand"
DESTINATION = "c29_r27"
METHOD = "qp"  # or "pinv"

# Same PD gain structure as ARO
KP = 3000.0
KV = 2.0 * np.sqrt(KP)

# Additional linear contact forces
CONTACT_KP = 15.0
CONTACT_KD = 1.0
NORMAL_PRELOAD = 0.15  # N, into the wall (+y)
MAX_CONTACT_FORCE = 0.5  # N

# The moving limb tracks its ball, not its joints: the joints track well, but the trunk (not
# actuated, no feedback on it) drifts from its plan and takes the ball with it
BALL_DAMPING = 0.03  # of the limb's damped inverse, as DAMPING in the kinematics

# The XL330's torque at its 1.75 A current limit [Microban]
MAX_MOTOR_TORQUE = 0.605  # Nm

SETTLE_TIME = 1.0  # s holding the last target, for the ball to seat in its socket

KNOT_STRIDE = 3  # One Bezier knot every 3 IK configurations
RENDER = True
REALTIME = True


# ------------------------------------------------------------
# Initial state conversion
# ------------------------------------------------------------

def pin_q_from_mujoco(sim):
    """Convert actual MuJoCo configuration to Pinocchio layout."""
    q = np.zeros(ik.model.nq)
    d = sim.data

    q[:3] = d.qpos[:3]

    # MuJoCo: wxyz; Pinocchio: xyzw
    q[3:7] = d.qpos[[4, 5, 6, 3]]

    for name in JOINTS:
        joint_id = ik.model.getJointId(name)
        q_index = ik.model.idx_qs[joint_id]
        q[q_index] = d.qpos[sim.model.joint(name).qposadr[0]]

    return q


# ------------------------------------------------------------
# Joint-space Bezier trajectory
# ------------------------------------------------------------

def make_joint_bezier(qs):
    """Create a C2 joint trajectory from the IK configurations.

    Uses quintic Bezier segments through selected IK configurations.
    Returns a function desired(t) -> (q, dq, ddq).

    Only the 19 actuated joints are interpolated.
    """
    # ik's q is in Pinocchio's joint order; sim._qadr is in the sim's (JOINTS)
    joints = [ik.model.idx_qs[ik.model.getJointId(name)] for name in JOINTS]
    q = np.asarray(qs)[:, joints]
    times = np.arange(len(q)) * ik.DT

    indices = np.r_[
        np.arange(0, len(q) - 1, KNOT_STRIDE),
        len(q) - 1,
    ]

    knots = q[indices]
    knot_times = times[indices]

    if len(knots) >= 3:
        velocities = np.gradient(
            knots, knot_times, axis=0
        )

        accelerations = np.gradient(
            velocities, knot_times, axis=0
        )
    else:
        velocities = np.zeros_like(knots)
        accelerations = np.zeros_like(knots)

    # Start and end at rest
    velocities[0] = 0.0
    velocities[-1] = 0.0
    accelerations[0] = 0.0
    accelerations[-1] = 0.0

    curves = []

    for i in range(len(knots) - 1):
        T = knot_times[i + 1] - knot_times[i]

        start = (
            knots[i],
            velocities[i],
            accelerations[i],
        )

        end = (
            knots[i + 1],
            velocities[i + 1],
            accelerations[i + 1],
        )

        control_points = connecting(start, end, T)

        curves.append(
            Bezier(
                control_points,
                t_min=float(knot_times[i]),
                t_max=float(knot_times[i + 1]),
            )
        )

    v_curves = [c.derivative(1) for c in curves]
    a_curves = [c.derivative(2) for c in curves]

    def desired(t):
        t = float(np.clip(
            t, knot_times[0], knot_times[-1]
        ))

        i = min(
            np.searchsorted(
                knot_times[1:], t, side="left"
            ),
            len(curves) - 1,
        )

        return (
            curves[i](t),
            v_curves[i](t),
            a_curves[i](t),
        )

    return desired


# ------------------------------------------------------------
# ARO-style inverse-dynamics controller
# ------------------------------------------------------------

def control_law(sim, desired, t, held_targets, ball_target):
    """Compute ID/PD torques plus linear contact-force torques."""
    m = sim.model
    d = sim.data

    mujoco.mj_forward(m, d)

    q_des, v_des, a_des = desired(t)

    # Actual actuated joint positions and velocities
    q = d.qpos[sim._qadr]
    v = d.qvel[sim._vadr]

    # The moving limb's joint error: its ball's error through a damped inverse of its Jacobian.
    # Only that limb's joints move its ball.
    jacobian = np.zeros((3, m.nv))
    site_id = m.site(BALL_SITES[MOVING]).id

    mujoco.mj_jacSite(
        m, d, jacobian, None, site_id
    )

    J_actuated = jacobian[:, sim._vadr]
    limb = np.flatnonzero(np.abs(J_actuated).sum(axis=0) > 0.0)
    J_limb = J_actuated[:, limb]

    q_error = q_des - q
    q_error[limb] = J_limb.T @ np.linalg.solve(
        J_limb @ J_limb.T + BALL_DAMPING**2 * np.eye(3),
        ball_target - d.site_xpos[site_id],
    )

    # ARO PD tracking with feedforward acceleration
    a_cmd = (
        a_des
        + KP * q_error
        + KV * (v_des - v)
    )

    # MuJoCo generalized mass matrix
    M = np.empty((m.nv, m.nv))
    mujoco.mj_fullM(m, d, M)

    # Desired base acceleration set to zero for this baseline
    full_acc = np.zeros(m.nv)
    full_acc[sim._vadr] = a_cmd

    # tau = M(q) * ddq_cmd + h(q, dq)
    tau_full = M @ full_acc + d.qfrc_bias

    # Only the actuated joints receive motor torques
    tau = tau_full[sim._vadr].copy()

    # Add contact-force torques, as in ARO
    max_contact_error = 0.0

    for limb, target in held_targets.items():
        site_id = m.site(BALL_SITES[limb]).id

        p = d.site_xpos[site_id]

        mujoco.mj_jacSite(
            m, d, jacobian, None, site_id
        )

        v_ball = jacobian @ d.qvel

        error = target - p
        max_contact_error = max(
            max_contact_error,
            np.linalg.norm(error),
        )

        # Small linear spring-damper force
        force = (
            CONTACT_KP * error
            - CONTACT_KD * v_ball
        )

        # Push slightly into the wall
        force += np.array([
            0.0, NORMAL_PRELOAD, 0.0
        ])

        magnitude = np.linalg.norm(force)

        if magnitude > MAX_CONTACT_FORCE:
            force *= MAX_CONTACT_FORCE / magnitude

        # Cartesian force -> joint torques
        J_actuated = jacobian[:, sim._vadr]

        tau += J_actuated.T @ force

    # Motor torque saturation
    tau = np.clip(
        tau,
        -MAX_MOTOR_TORQUE,
        MAX_MOTOR_TORQUE,
    )

    # Apply torque to the corresponding MuJoCo motor actuator
    for name, torque in zip(JOINTS, tau):
        actuator_id = mujoco.mj_name2id(
            m,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            name,
        )

        if actuator_id == -1:
            candidates = np.where(
                m.actuator_trnid[:, 0]
                == m.joint(name).id
            )[0]

            if len(candidates) != 1:
                raise ValueError(
                    f"Cannot find actuator for {name}"
                )

            actuator_id = int(candidates[0])

        d.ctrl[actuator_id] = torque

    return max_contact_error, np.max(np.abs(tau))


# ------------------------------------------------------------
# Run in MuJoCo
# ------------------------------------------------------------

def main():
    # Organisers' wall and simulator
    sim_wall = Wall.load(WALL_NAME)
    sim = ClimbingSim(sim_wall, cameras=[])

    sim.reset_hanging()

    # Our corresponding wall model and stances
    wall = io.load_wall(
        Path("data/examples") / f"{WALL_NAME}.json"
    )

    a = wall.start

    contacts = dict(a.contacts)
    contacts[MOVING] = DESTINATION

    b = Stance(contacts=contacts)

    # Use the actual settled simulator configuration
    # as the starting point for IK
    q0 = pin_q_from_mujoco(sim)

    # Generate the configurations using existing IK
    qs = ik.ik_traj(
        wall, a, b, q0, method=METHOD
    )

    # Fit the ARO-style joint-space Bezier
    desired = make_joint_bezier(qs)

    # Where the plan puts the moving ball, at each IK configuration
    ball_path = np.array([
        ik.ball_positions(q)[LIMBS.index(MOVING)]
        for q in qs
    ])

    # Three limbs stay attached
    initial_balls = sim.ball_positions()

    held_targets = {
        limb: p.copy()
        for limb, p in initial_balls.items()
        if limb != MOVING
    }

    print("Initial hooked:", sim.hooked())

    viewer = (
        sim.view(realtime=REALTIME)
        if RENDER else None
    )

    # MuJoCo physics timestep, typically 2 ms
    physics_dt = sim.model.opt.timestep

    duration = (len(qs) - 1) * ik.DT
    n_steps = round((duration + SETTLE_TIME) / physics_dt)

    max_held_error = 0.0
    saturated = 0

    try:
        for k in range(n_steps):
            t = k * physics_dt

            sample = min(round(t / ik.DT), len(qs) - 1)

            contact_error, peak_tau = control_law(
                sim, desired, t, held_targets, ball_path[sample]
            )

            max_held_error = max(
                max_held_error, contact_error
            )

            if peak_tau >= MAX_MOTOR_TORQUE - 1e-8:
                saturated += 1

            # Direct physics stepping, not sim.step()
            mujoco.mj_step(sim.model, sim.data)

            # Update the viewer every 20 ms
            if (k + 1) % sim.n_sub == 0:
                if viewer is not None:
                    if not viewer.is_running():
                        break
                    sim._sync_viewer()

                if (k + 1) % (10 * sim.n_sub) == 0:
                    print(
                        f"t={sim.data.time:.2f}s | "
                        f"held error={1000*contact_error:.1f} mm | "
                        f"hooked={sim.hooked()}"
                    )

    finally:
        sim.close()

    print(
        "Maximum held-ball error:",
        max_held_error * 1000,
        "mm",
    )
    print(
        "Motor saturation:",
        saturated,
        "/",
        n_steps,
        "physics steps",
    )
    print("Final hooked:", sim.hooked())


if __name__ == "__main__":
    main()
