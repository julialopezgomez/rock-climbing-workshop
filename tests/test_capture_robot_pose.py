"""Camera acquisition checks without requiring USB devices."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "capture_robot_pose", SCRIPTS / "capture_robot_pose.py"
)
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


class FakeCamera:
    def __init__(self, value):
        self.frame = np.full((480, 640, 3), value, np.uint8)
        self.grabs = 0
        self.released = False

    def isOpened(self):
        return True

    def set(self, property_id, value):
        return True

    def grab(self):
        self.grabs += 1
        return True

    def retrieve(self):
        return True, self.frame

    def release(self):
        self.released = True


class CapturePoseTest(unittest.TestCase):
    def test_grabs_both_cameras_and_releases_them(self):
        cameras = [FakeCamera(10), FakeCamera(20)]
        with patch.object(capture.cv2, "VideoCapture", side_effect=cameras):
            frames = capture.capture_pair([0, 1], [640, 480], warmup=3)
        self.assertEqual([int(frame[0, 0, 0]) for frame in frames], [10, 20])
        self.assertEqual([camera.grabs for camera in cameras], [4, 4])
        self.assertTrue(all(camera.released for camera in cameras))

    def test_resolution_mismatch_still_releases_cameras(self):
        cameras = [FakeCamera(10), FakeCamera(20)]
        with patch.object(capture.cv2, "VideoCapture", side_effect=cameras):
            with self.assertRaisesRegex(ValueError, "calibration expects"):
                capture.capture_pair([0, 1], [1280, 720], warmup=0)
        self.assertTrue(all(camera.released for camera in cameras))

    def test_loads_two_camera_calibration(self):
        matrix = [[800, 0, 320], [0, 800, 240], [0, 0, 1]]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "calibration.json"
            path.write_text(
                json.dumps({"camera_matrices": [matrix, matrix], "image_size": [640, 480]})
            )
            matrices, distortion, extrinsics, size, serials = capture.load_calibration(path)
        self.assertEqual(len(matrices), 2)
        self.assertEqual(distortion, [None, None])
        self.assertIsNone(extrinsics)
        self.assertEqual(size, [640, 480])
        self.assertIsNone(serials)

    def test_loads_default_realsense_calibration(self):
        matrices, distortion, extrinsics, size, serials = capture.load_calibration(
            capture.DEFAULT_CALIBRATION
        )
        self.assertEqual(serials, ["243222071545", "233722072758"])
        self.assertEqual(size, [640, 480])
        self.assertEqual(matrices[0][0, 0], 606.531921386719)
        self.assertEqual(matrices[1][0, 2], 323.282012939453)
        for coefficients in distortion:
            np.testing.assert_array_equal(coefficients, np.zeros(5))
            self.assertIsInstance(coefficients, np.ndarray)
        self.assertIsNone(extrinsics)

    def test_loads_1280_realsense_calibration(self):
        path = capture.DEFAULT_CALIBRATION.with_name("realsense-1280x720.json")
        matrices, distortion, extrinsics, size, serials = capture.load_calibration(path)
        self.assertEqual(size, [1280, 720])
        self.assertEqual(serials, ["243222071545", "233722072758"])
        self.assertEqual(matrices[0][0, 0], 909.7978515625)
        self.assertEqual(matrices[1][1, 2], 373.988861083984)
        self.assertTrue(all(isinstance(value, np.ndarray) for value in distortion))
        self.assertIsNone(extrinsics)

    def test_rejects_duplicate_camera_serials(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "calibration.json"
            path.write_text(json.dumps({"cameras": [
                {"serial": "same", "camera_matrix": np.eye(3).tolist()},
                {"serial": "same", "camera_matrix": np.eye(3).tolist()},
            ]}))
            with self.assertRaisesRegex(ValueError, "must be different"):
                capture.load_calibration(path)


if __name__ == "__main__":
    unittest.main()
