"""Portable perception resource, calibration and measured-marker checks."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

cv2 = pytest.importorskip("cv2")
pytest.importorskip("mujoco")
pytest.importorskip("trimesh")
from rcr.perception.challenge.camera_calibration import calibrate_dataset  # noqa: E402 - optional runtime dependencies checked above
from rcr.perception.challenge.capture_calibration import apply_intrinsics  # noqa: E402 - optional runtime dependencies checked above
from rcr.perception.challenge.markers import MarkerEstimator  # noqa: E402 - optional runtime dependencies checked above
from rcr.perception.challenge.robot_pose.frame_conventions import transform  # noqa: E402 - optional runtime dependencies checked above
from rcr.perception.challenge.team_frame import team_pose_record  # noqa: E402 - optional runtime dependencies checked above

REPO = Path(__file__).resolve().parents[1]


def test_team_basis_is_a_rotation():
    source = transform(Rotation.from_euler("xyz", [0.1, 0.2, 0.3]).as_matrix(), [1, 2, 3])
    result = team_pose_record({"valid": True, "T_wall_trunk": source, "timestamp_s": 10.0})
    assert result["position"] == [1.0, -3.0, 2.0]
    q = result["quat"]
    r = Rotation.from_quat([*q[1:], q[0]]).as_matrix()
    np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-12)
    assert team_pose_record({"valid": False}) is None


def test_marker_example_refuses_unmeasured_mount():
    config = json.loads((REPO / "config/perception/marker.example.json").read_text())
    with pytest.raises(ValueError, match="measured=true"):
        MarkerEstimator(config)


def test_measured_intrinsics_bind_camera_order_and_resolution():
    k = [[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]]
    packet = {
        "cameras": [{"camera_id": eye, "width": 640, "height": 480, "K": k} for eye in ["a", "b"]]
    }
    spec = {
        "pixel_convention": "opencv_centres",
        "camera_ids": ["a", "b"],
        "image_size": [640, 480],
        "stereo_rms_px": 0.2,
        "input_intrinsics": [k, k],
        "cameras": [{"K": k, "distortion": [0] * 5}] * 2,
    }
    assert apply_intrinsics(packet, spec)["cameras"][0]["intrinsics_source"]
    spec["camera_ids"] = ["b", "a"]
    with pytest.raises(ValueError, match="serial numbers/order"):
        apply_intrinsics(packet, spec)


def test_chessboard_calibration_from_generated_rgb(tmp_path):
    from PIL import Image

    rng = np.random.default_rng(42)
    k = np.array([[520.0, 0, 320.0], [0, 520.0, 240.0], [0, 0, 1.0]])
    square = 0.025
    pattern = np.zeros((7 * 60, 10 * 60), np.uint8)
    for row in range(7):
        for col in range(10):
            pattern[row * 60 : (row + 1) * 60, col * 60 : (col + 1) * 60] = (
                255 if (row + col) % 2 else 0
            )
    outer = np.array(
        [
            [-square, -square, 0],
            [9 * square, -square, 0],
            [9 * square, 6 * square, 0],
            [-square, 6 * square, 0],
        ]
    )
    image_corners = np.array([[0, 0], [599, 0], [599, 419], [0, 419]], np.float32)
    cases = []
    for index in range(18):
        scene = tmp_path / "scenes" / f"{index:06d}"
        scene.mkdir(parents=True)
        rotation = Rotation.from_euler(
            "xyz", rng.uniform([-0.3, -0.35, -0.15], [0.3, 0.35, 0.15])
        ).as_matrix()
        translation = np.array(
            [rng.uniform(-0.13, -0.03), rng.uniform(-0.1, -0.02), rng.uniform(0.55, 0.85)]
        )
        cameras = []
        for eye, offset in enumerate([0, 0.064]):
            points = outer @ rotation.T + translation - [offset, 0, 0]
            uv = points @ k.T
            uv = uv[:, :2] / uv[:, 2:]
            homography = cv2.getPerspectiveTransform(image_corners, uv.astype(np.float32))
            image = cv2.warpPerspective(
                pattern, homography, (640, 480), flags=cv2.INTER_LINEAR, borderValue=150
            )
            Image.fromarray(image).convert("RGB").save(scene / f"{eye}.png")
            cameras.append({"camera_id": str(eye), "rgb": f"{eye}.png", "K": k.tolist()})
        (scene / "input.json").write_text(json.dumps({"cameras": cameras}))
        cases.append({"id": scene.name})
    (tmp_path / "manifest.json").write_text(json.dumps({"cases": cases}))
    calibrate_dataset(
        SimpleNamespace(dataset=tmp_path, out=tmp_path / "fit", square=square, columns=9, rows=6)
    )
    fitted = json.loads((tmp_path / "fit/calibration.json").read_text())
    assert len(fitted["accepted_pairs"]) >= 10
    assert fitted["stereo_rms_px"] < 1
    assert abs(np.linalg.norm(fitted["t_right_left_m"]) - 0.064) < 0.005


def test_measured_marker_on_augmented_rgb_pair():
    from PIL import Image

    scene = REPO / "data/perception/robot/scenes/000000"
    packet = json.loads((scene / "input.json").read_text())
    reference = json.loads((scene / "evaluation.json").read_text())
    truth = np.asarray(reference["T_wall_trunk"])
    camera = np.asarray(reference["cameras"][0]["T_camera_wall"])
    marker_wall = np.eye(4)
    marker_wall[:3, :3] = camera[:3, :3].T @ np.diag([1.0, -1.0, -1.0])
    marker_wall[:3, 3] = truth[:3, 3] + [0.0, 0.0, 0.04]
    mount = np.linalg.inv(truth) @ marker_wall
    size = 0.08
    half = size / 2
    points = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]])
    code = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 5, 256
    )
    code = cv2.copyMakeBorder(code, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    # Homography uses the outer white margin; model coordinates specify black-code corners.
    expanded = points * (296 / 256)
    images = []
    for eye, c in enumerate(reference["cameras"]):
        image = np.asarray(Image.open(scene / packet["cameras"][eye]["rgb"]).convert("RGB")).copy()
        tc = np.asarray(c["T_camera_wall"]) @ marker_wall
        p = expanded @ tc[:3, :3].T + tc[:3, 3]
        uv = p @ np.asarray(c["K"]).T
        uv = uv[:, :2] / uv[:, 2:]
        homography = cv2.getPerspectiveTransform(
            np.array([[0, 0], [295, 0], [295, 295], [0, 295]], np.float32), uv.astype(np.float32)
        )
        warped = cv2.warpPerspective(
            code, homography, (image.shape[1], image.shape[0]), flags=cv2.INTER_NEAREST
        )
        area = (
            cv2.warpPerspective(
                np.full_like(code, 255),
                homography,
                (image.shape[1], image.shape[0]),
                flags=cv2.INTER_NEAREST,
            )
            > 0
        )
        image[area] = warped[area, None]
        images.append(image)
    estimator = MarkerEstimator(
        {"measured": True, "code_size_m": size, "marker_id": 5, "T_trunk_marker": mount.tolist()}
    )
    output = estimator.infer(packet, images)
    estimated = np.asarray(output["T_wall_trunk"])
    assert np.linalg.norm(estimated[:3, 3] - truth[:3, 3]) < 0.005
    assert Rotation.from_matrix(estimated[:3, :3] @ truth[:3, :3].T).magnitude() < np.deg2rad(5)


def test_capture_chooses_fresh_pair_after_slow_inference(monkeypatch):
    """A tracker/map refresh must not leave capture consuming an old FIFO."""
    import queue
    from collections import deque

    from rcr.perception.challenge import realsense_capture as capture_module

    capture = capture_module.DualColorCapture.__new__(capture_module.DualColorCapture)
    capture.errors = queue.Queue()
    capture.queues = [queue.Queue(), queue.Queue()]
    capture.buffers = [deque(), deque()]
    capture.max_skew_s = 0.015
    yy, xx = np.mgrid[:2, :2]
    capture.maps = [(xx.astype(np.float32), yy.astype(np.float32))] * 2
    for eye in range(2):
        for timestamp in [98.0, 99.9, 99.97]:
            capture.queues[eye].put(
                {"timestamp_s": timestamp + eye * 0.004, "rgb": np.zeros((2, 2, 3), np.uint8)}
            )
    monkeypatch.setattr(capture_module.time, "monotonic", lambda: 100.0)
    frames = capture.pair()
    assert frames[0]["timestamp_s"] == 99.97
    assert frames[1]["timestamp_s"] == 99.974
    assert all(not buffer for buffer in capture.buffers)


def test_live_keeps_failures_and_suppresses_stale_poses(tmp_path, monkeypatch):
    from rcr.perception.challenge import cli, online

    class Capture:
        serials = ["left", "right"]
        intrinsics = [SimpleNamespace(width=2, height=2, fx=1, fy=1, ppx=1, ppy=1)] * 2
        closed = False

        def __init__(self, **kwargs):
            self.stamps = iter([99.98, 99.0, 99.99])

        def start(self):
            pass

        def pair(self):
            stamp = next(self.stamps)
            return [{"timestamp_s": stamp, "rgb": np.zeros((2, 2, 3), np.uint8)}] * 2

        def close(self):
            Capture.closed = True

    class Estimator:
        closed = False

        def infer(self, packet, images):
            return {
                "valid": True,
                "T_wall_trunk": np.eye(4).tolist(),
                "timestamp_s": packet["timestamp_s"],
            }

        def close(self):
            Estimator.closed = True

    calls = 0

    def telemetry(path, stamp):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("Encoder sample is stale")
        return {}

    monkeypatch.setattr(online, "DualColorCapture", Capture)
    monkeypatch.setattr(online, "matching_telemetry", telemetry)
    monkeypatch.setattr(online.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(cli, "make_estimator", lambda args: Estimator())
    args = SimpleNamespace(
        serial=None,
        width=2,
        height=2,
        fps=30,
        out=tmp_path,
        camera_calibration=None,
        telemetry_log=tmp_path / "encoders.jsonl",
        count=3,
        task="robot",
        estimator="learned",
        max_output_age=0.25,
    )
    assert online.live(args) == 0
    records = [
        json.loads(line) for line in (tmp_path / "predictions.jsonl").read_text().splitlines()
    ]
    assert records[0]["valid"] is False and records[0]["reason"] == "Encoder sample is stale"
    assert records[1]["valid"] is False and records[1]["team_pose"] is None
    assert records[2]["valid"] is True and records[2]["fresh"] is True
    assert json.loads((tmp_path / "latest.json").read_text()) == records[2]
    assert Capture.closed and Estimator.closed
