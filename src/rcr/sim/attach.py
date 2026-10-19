"""Attaching and releasing limbs at runtime.

Backend A (default): one weld equality per limb, between the static wall body
and the limb, with torquescale 0 — a hook transmits force but no moment.
The weld is written from the current relative pose, so a limb can be grabbed
wherever it happens to be, and released by clearing the active flag.

Backend B (contact/friction) can implement the same four methods later; the
controller only talks to this interface.
"""

from __future__ import annotations

import mujoco
import numpy as np

from rcr.types import LIMBS


class WeldAttachment:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, limbs=LIMBS):
        self.model, self.data = model, data
        self.limbs = tuple(limbs)
        self._eq = {limb: model.equality(f"grip_{limb}").id for limb in self.limbs}
        self._body = {limb: model.body(f"rob_{limb}").id for limb in self.limbs}
        self._wall = model.body("wall").id
        self._held: dict[str, str | None] = {limb: None for limb in self.limbs}

    # -- queries ---------------------------------------------------------
    def is_attached(self, limb: str) -> bool:
        return bool(self.data.eq_active[self._eq[limb]])

    def held_hold(self, limb: str) -> str | None:
        return self._held[limb]

    def attached_limbs(self) -> list[str]:
        return [limb for limb in self.limbs if self.is_attached(limb)]

    # -- actions ---------------------------------------------------------
    def attach(self, limb: str, hold_id: str | None = None) -> None:
        """Freeze the limb against the wall in its current relative pose."""
        eq = self._eq[limb]
        b1, b2 = self._wall, self._body[limb]

        pos1, quat1 = self.data.xpos[b1], self.data.xquat[b1]
        pos2, quat2 = self.data.xpos[b2], self.data.xquat[b2]

        # Pose of body2 expressed in body1's frame.
        quat1_inv = np.zeros(4)
        mujoco.mju_negQuat(quat1_inv, quat1)
        rel_pos = np.zeros(3)
        mujoco.mju_rotVecQuat(rel_pos, pos2 - pos1, quat1_inv)
        rel_quat = np.zeros(4)
        mujoco.mju_mulQuat(rel_quat, quat1_inv, quat2)

        self.model.eq_data[eq, 0:3] = 0.0  # anchor at body2's origin
        self.model.eq_data[eq, 3:6] = rel_pos
        self.model.eq_data[eq, 6:10] = rel_quat
        self.model.eq_data[eq, 10] = 0.0  # torquescale: force only
        self.data.eq_active[eq] = 1
        self._held[limb] = hold_id

    def detach(self, limb: str) -> None:
        self.data.eq_active[self._eq[limb]] = 0
        self._held[limb] = None

    def detach_all(self) -> None:
        for limb in self.limbs:
            self.detach(limb)

    # -- forces ----------------------------------------------------------
    def limb_force(self, limb: str) -> np.ndarray:
        """Constraint force (world frame, N) carried by this limb's weld."""
        if not self.is_attached(limb):
            return np.zeros(3)
        eq = self._eq[limb]
        force = np.zeros(3)
        try:
            first = self._first_row(eq)
        except RuntimeError:
            return force
        force[:] = self.data.efc_force[first : first + 3]
        return force

    def _first_row(self, eq_id: int) -> int:
        for i in range(self.data.nefc):
            if (
                self.data.efc_type[i] == mujoco.mjtConstraint.mjCNSTR_EQUALITY
                and self.data.efc_id[i] == eq_id
            ):
                return i
        raise RuntimeError(f"equality {eq_id} has no constraint rows; is it active?")

    def total_force(self) -> np.ndarray:
        return sum((self.limb_force(limb) for limb in self.attached_limbs()), np.zeros(3))
