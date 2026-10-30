"""A reference motion to track: a whole-body trajectory from elsewhere (e.g. an IK solve), made
feasible by the optimiser (TransitionConfig.w_track).

The reference is configurations q (T x NQ, robot_model's layout) sampled at REFERENCE_RATE. Its
plan is read from it: a stance wherever all four balls are on holds of the wall (within
ON_HOLD), a move wherever one of them changes. Each move is split where the moving ball leaves
its hold, and both parts are resampled onto the transition's samples: the held part onto phase
A (k = 0..K), the part in the air onto phase B (k = K..N).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from rcr.optimiser import robot_model as rm
from rcr.optimiser import transition as tr
from rcr.types import LIMBS, ContactPlan, Stance, WallModel

REFERENCE_RATE = 50.0  # [Hz] the reference's samples [as given]
ON_HOLD = 0.003  # [m] a ball this close to a contact point is on that hold [Ours]
# Tracking mode's weight on the distance to the reference (the trunk's position counted 1 cm as
# 0.1 rad: TransitionConfig.track_position_scale). At 10, a 3 deg miss on every joint costs ~0.5
# per pose, against ~0.075 of torque: the reference wins unless the physics needs otherwise [Ours].
TRACK_WEIGHT = 10.0


@dataclass
class ReferenceMove:
    start: int  # sample of the reference where the stance before the move is reached
    release: int  # the last sample with the moving ball still on its old hold
    land: int  # the first sample with it on its new hold (the next stance)


def read(path: Path | str) -> np.ndarray:
    """The reference configurations (T x NQ) of an .npz file with an array `q`."""
    q = np.load(path)["q"]
    if q.ndim != 2 or q.shape[1] != rm.NQ:
        raise ValueError(f"{path}: q must be T x {rm.NQ}, got {q.shape}")
    return q


def holds_held(wall: WallModel, q: np.ndarray, cfg: tr.TransitionConfig) -> np.ndarray:
    """T x 4: the index of the hold each ball is on (wall.holds order), -1 when on none."""
    model = rm.load_robot_model()
    points = np.array([tr.contact_point(wall, h.id, cfg) for h in wall.holds])  # H x 3
    balls = np.array([np.array(model.limb_positions(x)).T for x in q])  # T x 4 x 3
    distance = np.linalg.norm(balls[:, :, None, :] - points[None, None], axis=3)  # T x 4 x H
    return np.where(distance.min(axis=2) < ON_HOLD, distance.argmin(axis=2), -1)


def find_wall(walls: list[WallModel], q: np.ndarray, cfg: tr.TransitionConfig) -> WallModel:
    """The wall whose holds the reference's balls are on most of the time."""
    return max(walls, key=lambda w: (holds_held(w, q[::10], cfg) >= 0).mean())


def plan_of(wall: WallModel, q: np.ndarray, cfg: tr.TransitionConfig):
    """The reference's contact plan and its moves (ContactPlan, [ReferenceMove]). ValueError if
    two limbs change hold between two stances."""
    held = holds_held(wall, q, cfg)
    stances, times = [], []
    for t, row in enumerate(held):
        if (row >= 0).all() and (not stances or tuple(row) != stances[-1]):
            stances.append(tuple(row))
            times.append(t)
    moves = []
    for i in range(len(stances) - 1):
        changed = [m for m in range(len(LIMBS)) if stances[i][m] != stances[i + 1][m]]
        if len(changed) != 1:
            raise ValueError(
                f"reference: {len(changed)} limbs change hold at sample {times[i + 1]}"
            )
        m = changed[0]
        on_old = np.flatnonzero(held[times[i] : times[i + 1], m] == stances[i][m])
        moves.append(ReferenceMove(times[i], times[i] + int(on_old[-1]), times[i + 1]))
    plan = ContactPlan(
        wall.id,
        [Stance({limb: wall.holds[s[j]].id for j, limb in enumerate(LIMBS)}) for s in stances],
    )
    return plan, moves


def resample(q: np.ndarray, at: np.ndarray) -> np.ndarray:
    """q at fractional sample indices `at` (len(at) x NQ): linear, quaternion normalised."""
    i = np.clip(np.floor(at).astype(int), 0, len(q) - 2)
    s = (at - i)[:, None]
    out = (1 - s) * q[i] + s * q[i + 1]
    out[:, 3:7] /= np.linalg.norm(out[:, 3:7], axis=1, keepdims=True)
    return out


def move_samples(q: np.ndarray, move: ReferenceMove, cfg: tr.TransitionConfig) -> np.ndarray:
    """The reference on one transition's samples ((N + 1) x NQ): its held part on phase A, its
    part in the air on phase B."""
    n, k = cfg.n_samples, cfg.release_sample
    a = np.linspace(move.start, move.release, k + 1)
    b = np.linspace(move.release, move.land, n - k + 1)[1:]
    return resample(q, np.concatenate([a, b]))


def guesses(plan: ContactPlan, wall: WallModel, q: np.ndarray, cfg: tr.TransitionConfig):
    """Per move of `plan`, the reference's samples ((N + 1) x NQ) of the same move: from the same
    stance to the same stance (so the same limb, from the same hold to the same hold); None
    where the reference has no such move. For the solver's starting guess (and w_track)."""
    ref_plan, moves = plan_of(wall, q, cfg)

    def key(stance: Stance) -> tuple:
        return tuple(stance.contacts[limb] for limb in LIMBS)

    found = {}
    for move, a, b in zip(moves, ref_plan.stances[:-1], ref_plan.stances[1:], strict=True):
        found.setdefault((key(a), key(b)), move)
    return [
        move_samples(q, found[key(a), key(b)], cfg) if (key(a), key(b)) in found else None
        for a, b in zip(plan.stances, plan.stances[1:], strict=False)
    ]


def tracking_config(cfg: tr.TransitionConfig, moves: list[ReferenceMove]) -> tr.TransitionConfig:
    """cfg for tracking: w_track on; the costs that shape the motion themselves off (swing,
    keep-out, hooks, posture), the reference decides it; phase durations the reference's
    (medians over its moves), so the result keeps its timing."""
    held = np.median([m.release - m.start for m in moves]) / REFERENCE_RATE
    air = np.median([m.land - m.release for m in moves]) / REFERENCE_RATE
    return replace(
        cfg,
        w_track=TRACK_WEIGHT,
        w_swing=0.0,
        w_keep_out=0.0,
        w_hook=0.0,
        w_nominal=0.0,
        w_spine=0.0,
        phase_a_duration=float(held),
        phase_b_duration=float(air),
    )
