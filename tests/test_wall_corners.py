"""Geometry checks for manual markerless and printed-square calibration."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from rcr.perception.challenge.src.calibration import (  # noqa: E402
    CalibrationError,
    calibrate_wall_corners,
    calibrate_wall_corner_squares,
    detect_wall_corner_squares,
    detect_wall_texture_core,
)


def test_recovers_front_camera_pose_from_exact_board_corners():
    K = np.array([[800.0, 0, 320], [0, 800, 240], [0, 0, 1]])
    wall = np.array([[0, 1.4, 0], [0.8, 1.4, 0], [0.8, 0, 0], [0, 0, 0]], dtype=float)
    rvec = np.array([[np.pi], [0], [0]])
    tvec = np.array([[-0.4], [0.7], [3.0]])
    pixels = cv2.projectPoints(wall, rvec, tvec, K, None)[0].reshape(4, 2)

    result = calibrate_wall_corners(pixels, K, (640, 480))

    np.testing.assert_allclose(result.camera_position_wall_m, [0.4, 0.7, 3.0], atol=1e-4)
    assert result.reprojection_rms_px < 1e-3
    assert result.marker_ids == []


def test_rejects_invalid_board_quadrilateral():
    K = np.array([[800.0, 0, 320], [0, 800, 240], [0, 0, 1]])
    corners = [[100, 100], [300, 300], [300, 100], [100, 300]]

    with pytest.raises(CalibrationError, match="convex quadrilateral"):
        calibrate_wall_corners(corners, K, (640, 480))


def test_texture_core_prefers_large_front_board_over_small_rear_board():
    image = np.full((480, 640, 3), 65, dtype=np.uint8)
    image[20:470, 245:515] = [180, 190, 205]
    image[130:420, 90:170] = [180, 190, 205]
    for x0, x1, y0, y1 in ((245, 515, 20, 470), (90, 170, 130, 420)):
        for y in range(y0 + 4, y1 - 4, 7):
            for x in range(x0 + 4, x1 - 4, 7):
                cv2.circle(image, (x, y), 1, (40, 45, 50), -1)

    mask, (x, y, width, height) = detect_wall_texture_core(image)

    assert 240 < x < 280
    assert 10 < y < 50
    assert width > 220
    assert height > 400
    assert mask[240, 370] == 255
    assert mask[240, 130] == 0


def test_detects_four_corner_squares_without_decoding_ids():
    image = np.full((480, 640, 3), 65, dtype=np.uint8)
    image[30:450, 250:490] = [180, 190, 205]
    image[130:420, 90:170] = [180, 190, 205]
    for x0, x1, y0, y1 in ((250, 490, 30, 450), (90, 170, 130, 420)):
        for y in range(y0 + 4, y1 - 4, 7):
            for x in range(x0 + 4, x1 - 4, 7):
                cv2.circle(image, (x, y), 1, (40, 45, 50), -1)
    for x, y in ((255, 35), (455, 35), (255, 415), (455, 415)):
        cv2.rectangle(image, (x, y), (x + 30, y + 30), (10, 10, 10), -1)
        cv2.rectangle(image, (x + 10, y + 10), (x + 20, y + 20), (255, 255, 255), -1)

    squares = detect_wall_corner_squares(image)

    assert set(squares) == {"TL", "TR", "BR", "BL"}
    np.testing.assert_allclose(squares["TL"].mean(axis=0), [270, 50], atol=3)
    np.testing.assert_allclose(squares["BR"].mean(axis=0), [470, 430], atol=3)

    K = np.array([[600.0, 0, 320], [0, 600, 240], [0, 0, 1]])
    calibration, _ = calibrate_wall_corner_squares(image, K)
    np.testing.assert_allclose(
        calibration.camera_position_wall_m, [0.233, 0.7, 2.0], atol=0.03
    )
    assert calibration.reprojection_rms_px < 2.0
    assert calibration.marker_ids == []
