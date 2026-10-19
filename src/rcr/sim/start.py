"""A start pose for a stance: where the robot is when the run begins.

Sim.reset() needs the robot already hanging on wall.start, so this solves for a base pose and
joint angles that put each limb's ball on its cup [Ours]. It is setup, not control: a damped
least-squares IK on the sim's own model, run once. The optimiser owns the real IK.
"""

from __future__ import annotations

import mujoco
import numpy as np

# Microban's neutral pose [Microban constants.py NEUTRAL_POSE], in degrees there.
NEUTRAL_POSE = {
    name: float(np.deg2rad(deg))
    for name, deg in {
        "left_hip_yaw": 0.0,
        "left_hip_roll": 5.0,
        "left_hip_pitch": -10.0,
        "left_knee": 0.0,
        "left_ankle_pitch": 0.0,
        "left_ankle_roll": -5.0,
        "right_hip_yaw": 0.0,
        "right_hip_roll": -5.0,
        "right_hip_pitch": -10.0,
        "right_knee": 0.0,
        "right_ankle_pitch": 0.0,
        "right_ankle_roll": 5.0,
        "left_shoulder_pitch": 10.0,
        "left_shoulder_roll": 10.0,
        "left_elbow": -20.0,
        "right_shoulder_pitch": 10.0,
        "right_shoulder_roll": -10.0,
        "right_elbow": -20.0,
        "head": 0.0,
    }.items()
}

TOLERANCE = 0.003  # [m] worst ball-to-cup error accepted for a start pose [Ours]


def solve(
    model: mujoco.MjModel,
    targets: dict[str, np.ndarray],
    site_of: dict[str, int],
    joint_qpos: np.ndarray,
    joint_dof: np.ndarray,
    neutral: np.ndarray,
    iterations: int = 300,
) -> np.ndarray:
    """qpos with each limb's ball site on its target, the trunk upright and clear of the wall.

    targets: limb -> wall-frame point; site_of: limb -> site id. Raises if unreachable.
    """
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qpos[joint_qpos] = neutral
    centre = np.mean(list(targets.values()), axis=0)
    data.qpos[0:3] = (centre[0], model.qpos0[1], centre[2] + 0.05)  # start near the holds
    upright = model.qpos0[3:7].copy()  # scene.py attaches the robot facing the wall
    standoff = model.qpos0[1]  # and this far from it, so the arms clear the board

    lo, hi = model.jnt_range[model.dof_jntid[joint_dof]].T
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    posture = np.zeros((len(joint_dof), model.nv))
    posture[np.arange(len(joint_dof)), joint_dof] = 1.0
    w_tip, w_upright, w_standoff, w_posture, damping = 1.0, 0.2, 0.3, 0.02, 1e-4

    for k in range(iterations):
        # Regularise early to pick a natural pose, then let the tips converge exactly.
        fade = max(0.0, 1.0 - 2.0 * k / iterations)
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        rows, errs = [], []
        for limb, target in targets.items():
            mujoco.mj_jacSite(model, data, jacp, None, site_of[limb])
            rows.append(w_tip * jacp.copy())
            errs.append(w_tip * (target - data.site_xpos[site_of[limb]]))
        mujoco.mj_jacBody(model, data, None, jacr, model.jnt_bodyid[0])
        rot = np.zeros(3)
        mujoco.mju_subQuat(rot, upright, data.qpos[3:7])  # body-frame error
        rows.append(fade * w_upright * data.xmat[model.jnt_bodyid[0]].reshape(3, 3).T @ jacr)
        errs.append(fade * w_upright * rot)
        rows.append(fade * w_standoff * np.eye(model.nv)[1:2])  # base y
        errs.append(fade * w_standoff * np.array([standoff - data.qpos[1]]))
        rows.append(fade * w_posture * posture)
        errs.append(fade * w_posture * (neutral - data.qpos[joint_qpos]))
        jac, err = np.vstack(rows), np.concatenate(errs)
        dq = np.linalg.solve(jac.T @ jac + damping * np.eye(model.nv), jac.T @ err)
        mujoco.mj_integratePos(model, data.qpos, dq, 1.0)
        data.qpos[joint_qpos] = np.clip(data.qpos[joint_qpos], lo, hi)

    mujoco.mj_kinematics(model, data)
    worst = max(np.linalg.norm(t - data.site_xpos[site_of[k]]) for k, t in targets.items())
    if worst > TOLERANCE:
        raise ValueError(f"stance not reachable: a ball is {worst * 1e3:.0f} mm off its cup")
    return data.qpos.copy()
