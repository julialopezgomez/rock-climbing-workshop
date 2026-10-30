"""Geometric controls for physical replay, independent of the real recordings."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("mujoco")
pytest.importorskip("trimesh")
from rcr.perception.challenge.physical_calibration import fit_grid_pose  # noqa: E402
from rcr.perception.challenge.physical_mapping import cup_instances  # noqa: E402
from rcr.perception.challenge.physical_robot import fit_square  # noqa: E402
from rcr.perception.challenge.physical_run import merge_static_map  # noqa: E402


def geometry():
    return dict(
        wall_width_m=0.8,
        wall_height_m=1.4,
        hole_pitch_m=0.015,
        first_hole_left_m=0.01,
        first_hole_top_m=0.01,
    )


def camera(position):
    position = np.asarray(position, float)
    forward = np.array([0.4, 0.7, 0]) - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 1, 0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    transform = np.eye(4)
    transform[:3, :3] = np.stack([right, down, forward])
    transform[:3, 3] = -transform[:3, :3] @ position
    return dict(K=[[913, 0, 640], [0, 912, 360], [0, 0, 1]], T_camera_wall=transform)


def project(points, record):
    transform, matrix = np.asarray(record["T_camera_wall"]), np.asarray(record["K"])
    p = points @ transform[:3, :3].T + transform[:3, 3]
    pixels = p @ matrix.T
    return pixels[:, :2] / pixels[:, 2:]


def synthetic_holes():
    indices = np.array([(column, row) for row in range(60) for column in range(52)], float)
    points = np.column_stack(
        [0.01 + indices[:, 0] * 0.015, 1.39 - indices[:, 1] * 0.015, np.zeros(len(indices))]
    )
    source = camera([-0.4, 0.8, 1.6])
    pixels = project(points, source)
    rng = np.random.default_rng(11)
    selected = rng.random(len(points)) > 0.3
    pixels, indices = pixels[selected], indices[selected]
    pixels += rng.normal(0, 0.12, pixels.shape)
    pixels[:40] = rng.uniform([0, 0], [1280, 720], (40, 2))
    return pixels, indices, np.asarray(source["K"])


def test_camera_fit_tolerates_missing_holes_and_false_correspondences():
    pixels, indices, matrix = synthetic_holes()
    fitted, _, good = fit_grid_pose(pixels, indices, matrix, geometry())
    assert len(good) > len(pixels) * 0.9
    np.testing.assert_allclose(fitted["camera_position_wall_m"], [-0.4, 0.8, 1.6], atol=0.001)
    assert fitted["reprojection_rms_px"] < 0.3
    assert fitted["grid_origin_verified"] is False


def test_low_pixel_residual_cannot_verify_the_metric_scale():
    pixels, indices, matrix = synthetic_holes()
    nominal, _, _ = fit_grid_pose(pixels, indices, matrix, geometry())
    scaled_geometry = {key: value * 1.1 for key, value in geometry().items()}
    scaled, _, _ = fit_grid_pose(pixels, indices, matrix, scaled_geometry)
    np.testing.assert_allclose(
        scaled["camera_position_wall_m"],
        np.array(nominal["camera_position_wall_m"]) * 1.1,
        atol=1e-5,
    )
    assert scaled["reprojection_rms_px"] == pytest.approx(nominal["reprojection_rms_px"], abs=1e-5)
    assert scaled["grid_origin_verified"] is False


def test_square_geometry_recovers_metric_pose_from_two_known_cameras():
    from scipy.spatial.transform import Rotation

    cameras = [camera([-0.4, 0.8, 1.6]), camera([1.05, 0.9, 1.65])]
    centre = np.array([0.44, 0.65, 0.12])
    rotation = Rotation.from_euler("xyz", [0.1, 0.05, 0.3]).as_matrix()
    square = np.array(
        [[-0.025, 0.025, 0], [0.025, 0.025, 0], [0.025, -0.025, 0], [-0.025, -0.025, 0]]
    )
    points = square @ rotation.T + centre
    fitted = fit_square([project(points, c) for c in cameras], cameras)
    np.testing.assert_allclose(np.array(fitted["T_wall_marker"])[:3, 3], centre, atol=1e-5)
    assert fitted["estimated_black_square_size_m"] == pytest.approx(0.05, abs=1e-5)
    assert fitted["reprojection_rms_px"] < 1e-5


def test_touching_cups_are_split_before_cad_fitting():
    image = np.zeros((1400, 800, 3), np.uint8)
    image[200:231, 100:140] = [240, 80, 30]
    image[200:231, 145:185] = [240, 80, 30]
    image[214:217, 139:146] = [240, 80, 30]
    transform = np.diag([1.0, -1, -1, 1])
    transform[:3, 3] = [0, 1.4, 1.026]
    record = dict(K=[[1000, 0, 0], [0, 1000, 0], [0, 0, 1]], T_camera_wall=transform)
    entries, _ = cup_instances(image, record, geometry())
    assert len(entries) == 2
    assert all(entry["split_from"] == 2 for entry in entries)
    assert np.linalg.norm(entries[0]["point"] - entries[1]["point"]) > 0.035
    mask = np.ones(image.shape[:2], bool)
    excluded, _ = cup_instances(image, record, geometry(), exclude_mask=mask)
    assert not excluded


def test_static_map_rejects_robot_present_cups_contradicted_by_empty_rgb():
    def hold(point):
        return dict(
            socket_hypothesis_wall_m=point,
            color="orange",
            image_supported=True,
            observed_by=[0, 1],
            boundary_loss_px2=1.0,
        )

    prior = [hold([0.3, 0.6, 0.026])]
    occupied = [hold([0.301, 0.6, 0.026]), hold([0.5, 0.6, 0.026])]
    images = [np.full((720, 1280, 3), 150, np.uint8)] * 2
    cameras = [camera([-0.4, 0.8, 1.6]), camera([1.05, 0.9, 1.65])]
    result = merge_static_map(prior, occupied, images, cameras)
    assert len(result["holds"]) == result["reobserved"] == 1
    assert len(result["new_candidates_rejected"]) == 1
    assert result["valid_for_control"] is False
    assert "reobserved_with_robot" not in prior[0]


def test_label_initialized_trunk_hypotheses_are_proper_rigid_transforms():
    from types import SimpleNamespace

    from rcr.perception.challenge.physical_trunk import seeds_from_label
    from rcr.perception.challenge.robot_pose.frame_conventions import validate_transform

    centre = np.array([0.44, 0.65, 0.12])
    corners = np.array([[-0.03, 0.05, 0], [0.03, 0.05, 0], [0.03, -0.05, 0], [-0.03, -0.05, 0]])
    label = dict(
        outer_corners_wall_m=(corners + centre).tolist(), square=dict(T_wall_marker=np.eye(4))
    )
    mesh = SimpleNamespace(bounds=np.array([[-0.044, -0.054, -0.0015], [0.033, 0.054, 0.126]]))
    hypotheses = seeds_from_label(label, mesh)
    assert len(hypotheses) == 3
    for pose in hypotheses:
        validate_transform(pose)
        assert pose[2, 0] < 0  # The trunk's forward axis points toward the wall.
