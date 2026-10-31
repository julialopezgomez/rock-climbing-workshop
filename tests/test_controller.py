"""The controller's tracking core: the numbers training and the robot must agree on."""

import numpy as np
import pytest

from rcr.controller import Controller, tracking
from rcr.types import RobotState, Trajectory

NAMES = list(reversed(tracking.JOINTS))  # a file order unlike ours: everything goes by name


def trajectory(positions, planned=None, times=(0.0, 0.25)):
    """Two samples; each row's value for a joint is given in JOINTS order, stored in NAMES order."""
    order = [tracking.JOINTS.index(n) for n in NAMES]

    def rows(values):
        return [np.asarray(v, dtype=float)[order].tolist() for v in values]

    return Trajectory(NAMES, list(times), rows(positions), rows(planned) if planned else None)


def test_joints_are_reordered_by_name():
    index = np.arange(len(tracking.JOINTS), dtype=float)
    ref = tracking.reference(trajectory([index, index]))
    assert np.array_equal(ref.feedforward[0], index)


def test_preload_is_rescaled_to_the_softer_gain():
    zero, preloaded = np.zeros(19), np.full(19, 0.1)  # pre-load tau / kp at kp_fw 400
    ref = tracking.reference(trajectory([preloaded, preloaded], [zero, zero]))
    assert np.allclose(ref.feedforward, 0.1 * 400 / 125)
    assert np.allclose(ref.planned, 0.0)


def test_without_planned_positions_plays_positions_back():
    q = np.full(19, 0.3)
    ref = tracking.reference(trajectory([q, q]))
    assert np.allclose(ref.feedforward, q) and np.allclose(ref.planned, q)


def test_interpolation_between_samples_and_held_outside():
    a, b = np.zeros(19), np.ones(19)
    ref = tracking.reference(trajectory([a, b], [a, b]))
    assert np.allclose(tracking.at(ref, 0.05)[1], 0.2)  # 20 % of the way at 0.05 s of 0.25 s
    assert np.allclose(tracking.at(ref, -1.0)[1], 0.0)
    assert np.allclose(tracking.at(ref, 9.0)[1], 1.0)


def test_gravity_from_the_trunk_orientation():
    assert np.allclose(tracking.gravity((1.0, 0.0, 0.0, 0.0)), (0.0, 0.0, -1.0))
    c = np.cos(np.pi / 4)  # 90 degrees about x: the world's down is the trunk's -y
    assert np.allclose(tracking.gravity((c, c, 0.0, 0.0)), (0.0, -1.0, 0.0))


def test_observation_layout():
    planned, q = np.full(19, 0.5), np.full(19, 0.2)
    obs = tracking.observation(planned, q, np.zeros(19), (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.1))
    assert obs.shape == (tracking.OBS_SIZE,) and obs.dtype == np.float32
    assert np.allclose(obs[:19], 0.3) and np.allclose(obs[19:38], 0.2)
    assert np.allclose(obs[-6:], (0.0, 0.0, -1.0, 0.0, 0.0, 0.1))


def test_targets_blend_the_feedforward_and_the_policy():
    ff, low, high = np.full(19, 0.5), np.full(19, -1.0), np.full(19, 1.0)
    assert np.allclose(tracking.targets(ff, np.ones(19), 0.0, low, high), 0.5)  # playback
    assert np.allclose(tracking.targets(ff, np.full(19, -1.0), 1.0, low, high), -1.0)  # own
    assert np.allclose(tracking.targets(ff, np.zeros(19), 0.2, low, high), 0.8 * 0.5)  # mid: 0
    assert np.allclose(tracking.targets(ff, np.full(19, 3.0), 1.0, low, high), 1.0)  # action <= 1
    assert np.allclose(tracking.targets(np.full(19, 2.0), np.ones(19), 0.0, low, high), 1.0)


def test_controller_without_policy_plays_the_feedforward_back():
    zero, preloaded = np.zeros(19), np.full(19, 0.1)
    ctrl = Controller(trajectory([preloaded, preloaded], [zero, zero]))
    state = RobotState(0.1, {}, {}, (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    targets = ctrl.step(state).positions
    assert set(targets) == set(tracking.JOINTS)
    assert np.allclose(list(targets.values()), 0.1 * 400 / 125)


def test_a_policy_needs_planned_positions():
    q = np.zeros(19)
    with pytest.raises(ValueError, match="planned_positions"):
        Controller(trajectory([q, q]), policy="unused.onnx")
