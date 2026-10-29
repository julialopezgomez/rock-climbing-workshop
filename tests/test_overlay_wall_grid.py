"""Wall-grid overlay geometry checks."""

import importlib.util
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("overlay_wall_grid", SCRIPTS / "overlay_wall_grid.py")
grid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grid)


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


if __name__ == "__main__":
    unittest.main()
