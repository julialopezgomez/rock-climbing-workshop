"""rcr.controller.mjlab_policy on a tiny synthetic bundle, with a fake policy (no ONNX file).

The real bundles are checked against mjlab's env tick by tick by mjlab_microban's export; these
tests pin the reference phase, the history, the targets and the state estimate.
"""

import math

import numpy as np
import pytest

from rcr.controller import mjlab_policy as mp

T = 50  # frames of the synthetic climb
FUTURE = [1, 2, 3]
TERMS = [  # (term, per-step dim, history), as the env's actor
    ("base_ang_vel", 3, 2),
    ("projected_gravity", 3, 2),
    ("base_yaw", 2, 2),
    ("joint_pos", 2, 2),
    ("joint_vel", 2, 2),
    ("actions", 3, 2),
    ("command", 2 * len(FUTURE), 0),
    ("phase", 2, 0),
]


class FakeInput:
    name = "obs"
    shape = [1, sum(d * max(h, 1) for _, d, h in TERMS)]


class FakeSession:
    """onnxruntime's InferenceSession, returning a fixed action."""

    def __init__(self, action=(0.0, 0.0, 0.0)) -> None:
        self.action = np.asarray(action, np.float32)
        self.inputs = []

    def get_inputs(self):
        return [FakeInput()]

    def run(self, _outputs, feed):
        self.inputs.append(feed["obs"][0].copy())
        return [self.action[None]]


def kinematics() -> dict:
    """Trunk; body 1 on a hinge about z (joint 0) at x = 0.1; body 2 fixed at x = -0.1."""
    return {
        "kin_parent": np.array([-1, 0, 0]),
        "kin_body_pos": np.array([[0, 0, 0], [0.1, 0, 0], [-0.1, 0, 0]], float),
        "kin_body_quat": np.array([[1, 0, 0, 0]] * 3, float),
        "kin_joint": np.array([-1, 0, -1]),
        "kin_joint_axis": np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]], float),
        "kin_joint_pos": np.zeros((3, 3)),
        "kin_joint_qpos0": np.zeros(3),
        "ball_site_body": np.array([1, 1, 2, 2]),
        "ball_site_pos": np.array([[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05], [0, 0, -0.05]], float),
    }


@pytest.fixture
def bundle(tmp_path):
    kin = kinematics()
    ref_q = np.stack([0.01 * np.arange(T), np.zeros(T)], axis=1)
    trunk = np.stack([np.zeros(T), np.zeros(T), 0.002 * np.arange(T)], axis=1)
    fk = mp.Kinematics(kin)
    balls = np.array([trunk[t] + fk.balls(ref_q[t]) for t in range(T)])
    np.savez(
        tmp_path / "climb_bundle.npz",
        format=mp.BUNDLE_FORMAT,
        joint_names=np.array(["a", "b"]),
        default_joint_pos=np.zeros(2),
        action_scale=np.full(2, 0.5),
        clip_actions=np.nan,
        num_actions=3,
        obs_terms=np.array([t for t, _, _ in TERMS]),
        obs_dims=np.array([d for _, d, _ in TERMS]),
        obs_history=np.array([h for _, _, h in TERMS]),
        future_offsets=np.array(FUTURE),
        ref_joint_pos=ref_q,
        ref_trunk_pos=trunk,
        ref_trunk_quat=np.tile([1.0, 0, 0, 0], (T, 1)),
        ref_ball_pos=balls,
        ref_contact=np.ones((T, 4), bool),
        imu_quat=np.array([1.0, 0, 0, 0]),
        phase_enabled=True,
        phase_sigma=0.1,
        phase_k=10.0,
        phase_delta_clip=2.0,
        phase_rate_terms=np.array(["balls", "trunk_pos", "trunk_ori"]),
        phase_tolerances=np.array([0.025, 0.04, 0.3]),
        phase_min_rate=0.1,
        **kin,
    )
    return tmp_path


def meas_at(q, yaw=0.0) -> dict:
    """The robot's measurements: still, upright, heading yaw (any IMU reference)."""
    return {
        "q": np.asarray(q, float),
        "dq": np.zeros(2),
        "gyro": np.zeros(3),
        "gravity": np.array([0.0, 0.0, -1.0]),
        "quat": np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]),
    }


def test_playback_follows_the_reference(bundle):
    ctrl = mp.ClimbController(bundle, session=FakeSession())
    ctrl.reset(meas_at(ctrl.ref_q[0]))
    assert ctrl.t == 1  # as the env's reset, the reference advances once
    for _ in range(10):
        t = ctrl.t
        targets = ctrl.tick(meas_at(ctrl.ref_q[t]))  # the robot exactly on the reference
        assert ctrl.v_ref == pytest.approx(1.0)
        if ctrl.ticks > 1:
            t += 1  # v = v_ref = 1: one frame per tick
        assert ctrl.t == t
        np.testing.assert_allclose(targets, ctrl.ref_q[t + 1])  # the frame the step ends on


def test_history_is_backfilled_then_rolls(bundle):
    session = FakeSession()
    ctrl = mp.ClimbController(bundle, session=session)
    ctrl.reset(meas_at(ctrl.ref_q[0]))
    ctrl.tick(meas_at(ctrl.ref_q[1]))
    ctrl.tick(meas_at(ctrl.ref_q[2] + [0.1, 0.0]))
    first, second = session.inputs
    assert first.shape == (ctrl.obs_size,)
    joint_pos = slice(16, 20)  # after base_ang_vel, projected_gravity, base_yaw: 2 x (3 + 3 + 2)
    np.testing.assert_allclose(first[joint_pos], np.tile(ctrl.ref_q[1], 2))  # oldest == newest
    np.testing.assert_allclose(second[joint_pos], np.r_[ctrl.ref_q[1], ctrl.ref_q[2] + [0.1, 0]])


def test_the_reference_waits_for_a_robot_behind(bundle):
    ctrl = mp.ClimbController(bundle, session=FakeSession())
    ctrl.reset(meas_at(ctrl.ref_q[0]))
    ctrl.tick(meas_at(ctrl.ref_q[0]))
    ctrl.tick(meas_at(ctrl.ref_q[0] + [1.0, 0.0]))  # one arm 1 rad off: its balls ~5 cm off
    assert ctrl.v_ref < 0.5
    assert 0.1 - 1e-6 <= ctrl.speed < 0.5  # slower, but at least min_phase_rate


def test_the_policy_rate_residual_is_clipped(bundle):
    ctrl = mp.ClimbController(bundle, session=FakeSession(action=(0.0, 0.0, 100.0)))
    ctrl.reset(meas_at(ctrl.ref_q[0]))
    ctrl.tick(meas_at(ctrl.ref_q[1]))
    ctrl.tick(meas_at(ctrl.ref_q[1]))
    assert ctrl.speed == pytest.approx(1.0 + 0.1 * 2.0, rel=1e-5)  # v_ref + sigma * clip(dv)


def test_estimate_aligns_the_imu_yaw_and_finds_the_trunk(bundle):
    ctrl = mp.ClimbController(bundle, session=FakeSession())
    ctrl.reset(meas_at(ctrl.ref_q[0], yaw=0.7))  # the IMU's yaw reference is arbitrary
    est = ctrl.estimate(meas_at(ctrl.ref_q[20], yaw=0.7), 20)
    np.testing.assert_allclose(est["trunk_pos"], ctrl.ref_trunk_pos[20], atol=1e-9)
    assert mp.rotation_angle(est["trunk_R"], np.eye(3)) == pytest.approx(0.0, abs=1e-9)


def test_end_of_the_climb_holds_the_last_frame(bundle):
    ctrl = mp.ClimbController(bundle, session=FakeSession())
    ctrl.reset(meas_at(ctrl.ref_q[0]))
    for _ in range(T + 5):
        targets = ctrl.tick(meas_at(ctrl.ref_q[ctrl.t]))
    assert ctrl.finished and ctrl.t == T - 1
    np.testing.assert_allclose(targets, ctrl.ref_q[-1])


def test_forward_kinematics_of_the_hinge():
    fk = mp.Kinematics(kinematics())
    balls = fk.balls(np.array([math.pi / 2, 0.0]))
    np.testing.assert_allclose(balls[0], [0.1, 0.05, 0.0], atol=1e-12)  # (0.05, 0, 0) turned 90 deg
    np.testing.assert_allclose(balls[2], [-0.1, 0.0, 0.05], atol=1e-12)
