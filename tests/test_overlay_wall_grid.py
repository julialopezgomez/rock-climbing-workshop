"""Wall-grid overlay geometry checks."""

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("overlay_wall_grid", SCRIPTS / "overlay_wall_grid.py")
grid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grid)
pose = importlib.import_module("robot_pose_aruco")


class WallGridTest(unittest.TestCase):
    def test_hole_margins_and_pitch(self):
        columns, rows = grid.hole_coordinates()
        self.assertEqual(len(columns), 52)
        self.assertEqual(len(rows), 93)
        np.testing.assert_allclose([columns[0], columns[-1]], [0.0175, 0.7825])
        np.testing.assert_allclose([rows[0], rows[-1]], [0.010, 1.390])
        np.testing.assert_allclose(np.diff(columns), 0.015)
        np.testing.assert_allclose(np.diff(rows), 0.015)

    def test_homography_recovers_wall_origin_and_marker_center(self):
        known = np.array([[500.0, 35.0, 70.0], [-20.0, -350.0, 680.0],
                          [0.03, 0.08, 1.0]])
        detections = {
            mid: grid.project(known, grid._wall_corners(mid)[:, [0, 2]])
            for mid in (1, 2, 3, 4)
        }
        fitted, ids, rms = grid.wall_homography(detections)
        self.assertEqual(ids, [1, 2, 3, 4])
        self.assertLess(rms, 0.001)
        np.testing.assert_allclose(grid.project(fitted, [[0, 0]]), [[70, 680]], atol=0.001)
        np.testing.assert_allclose(
            grid.project(fitted, [[0.068, 0.068]]),
            grid.project(known, [[0.068, 0.068]]),
            atol=0.001,
        )

    def test_selects_intrinsics_by_serial_and_image_size(self):
        first, distortion = grid.calibration_for("243222071545", (1280, 720))
        second, _ = grid.calibration_for("233722072758", (640, 480))
        self.assertEqual(first[0, 0], 909.7978515625)
        self.assertEqual(second[0, 0], 606.871154785156)
        np.testing.assert_array_equal(distortion, np.zeros(5))
        with self.assertRaisesRegex(ValueError, "No Color calibration"):
            grid.calibration_for("243222071545", (800, 600))

    def test_pose_report_records_image_order_and_calibration(self):
        paths = [Path("AruCo4_Color.png"), Path("AruCo3_Color.png")]
        images = [np.zeros((480, 640, 3), dtype=np.uint8),
                  np.zeros((720, 1280, 3), dtype=np.uint8)]
        serials = ["233722072758", "243222071545"]
        calibrations = [grid.calibration_for(serial, (image.shape[1], image.shape[0]))
                        for serial, image in zip(serials, images, strict=True)]
        result = {"position_m": [0.4, -0.1, 0.6],
                  "rotation_matrix": np.eye(3).tolist(), "quaternion_wxyz": [1, 0, 0, 0]}
        fits = [(np.eye(3), [1, 2, 3], 0.5), (np.eye(3), [1, 2, 4], 0.6)]
        report = grid.pose_report(paths, images, serials, calibrations, result, fits)
        self.assertEqual([camera["assumed_serial"] for camera in report["cameras"]], serials)
        self.assertEqual(report["cameras"][0]["resolution"], [640, 480])
        self.assertEqual(report["cameras"][1]["resolution"], [1280, 720])
        self.assertEqual(report["cameras"][0]["camera_matrix"][0][0], 606.871154785156)
        self.assertEqual(report["cameras"][1]["camera_matrix"][0][0], 909.7978515625)
        np.testing.assert_allclose(report["marker_center_wall_m"], [0.367, -0.1, 0.65475])

    def test_single_image_pose_keeps_ambiguity_visible(self):
        matrix = np.array([[925.0, 0, 640.0], [0, 920.0, 360.0], [0, 0, 1.0]])
        camera_rotation = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
        translation = -camera_rotation @ np.array([0.0, -2.4, 0.9])
        trunk_rotation = cv2.Rodrigues(np.array([0.03, -0.04, 1.5]))[0]
        trunk_position = np.array([0.4, -0.12, 0.3])
        marker_rotation = trunk_rotation @ pose.R_TRUNK_MARKER
        marker_center = trunk_position + trunk_rotation @ pose.MARKER_CENTRE
        half = pose.MARKER_SIZE / 2
        local = np.array([[-half, half, 0], [half, half, 0],
                          [half, -half, 0], [-half, -half, 0]])

        def project(points):
            return cv2.projectPoints(points, cv2.Rodrigues(camera_rotation)[0],
                                     translation, matrix, None)[0][:, 0]

        detections = {mid: project(pose._wall_corners(mid)) for mid in (1, 2, 3, 4)}
        detections[5] = project(local @ marker_rotation.T + marker_center)
        with patch.object(pose, "detect_markers", return_value=detections):
            result = pose.estimate_robot_pose([np.zeros((720, 1280, 3), np.uint8)], [matrix])
        self.assertEqual(result["method"], "single_camera")
        self.assertEqual(result["cameras_used"], [0])
        self.assertTrue(result["warnings"])
        self.assertTrue(result["alternate_pose_candidates"])
        np.testing.assert_allclose(result["position_m"], trunk_position, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
