"""Image-only pose geometry and corner refinement; OpenCV is optional in base env."""

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

if cv2 is not None:
    spec = importlib.util.spec_from_file_location(
        "robot_pose_aruco", Path(__file__).parents[1] / "scripts" / "robot_pose_aruco.py"
    )
    pose = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pose)


# @unittest.skipIf(cv2 is None, "OpenCV is required for image perception tests")
class ArucoPoseTest(unittest.TestCase):
    def setUp(self):
        self.local = np.array(
            [[-0.021, 0.021, 0], [0.021, 0.021, 0], [0.021, -0.021, 0], [-0.021, -0.021, 0]]
        )
        self.trunk_rotation = cv2.Rodrigues(np.array([0.03, -0.04, 1.5]))[0]
        self.trunk_position = np.array([0.4, -0.12, 0.3])
        self.rotation = self.trunk_rotation @ pose.R_TRUNK_MARKER
        self.centre = self.trunk_position + self.trunk_rotation @ pose.MARKER_CENTRE
        self.matrix = np.array([[925.0, 0, 640.0], [0, 920.0, 360.0], [0, 0, 1.0]])
        self.distortion = np.array([0.05, -0.02, 0.001, -0.002, 0.003])
        camera_rotation = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
        self.views, self.detections = [], []
        for x in (0.0, 0.8):
            translation = -camera_rotation @ np.array([x, -2.4, 0.9])

            def project(world, translation=translation):
                return cv2.projectPoints(
                    world,
                    cv2.Rodrigues(camera_rotation)[0],
                    translation,
                    self.matrix,
                    self.distortion,
                )[0][:, 0]

            pixels = project(self.local @ self.rotation.T + self.centre)
            self.views.append((camera_rotation, translation, self.matrix, self.distortion, pixels))
            detected = {mid: project(pose._wall_corners(mid)) for mid in (1, 2, 3, 4)}
            detected[5] = pixels
            self.detections.append(detected)
        self.images = [np.zeros((720, 1280, 3), np.uint8)] * 2

    def test_joint_fit_recovers_pose_with_lens_distortion(self):
        initial_rotation = cv2.Rodrigues(np.array([0.08, -0.06, 0.05]))[0] @ self.rotation
        result = pose._joint_pose_fit(
            self.local, initial_rotation, self.centre + [0.008, -0.01, 0.006], self.views
        )
        rotation, centre, errors, initial_errors, _ = result
        np.testing.assert_allclose(centre, self.centre, atol=1e-7)
        np.testing.assert_allclose(rotation, self.rotation, atol=1e-6)
        self.assertLess(max(errors), 1e-6)
        self.assertGreater(max(initial_errors), 1)

    def test_mujoco_pixel_conversion_preserves_supplied_calibration(self):
        before = self.matrix.copy()
        converted = pose.mujoco_pixel_intrinsics(self.matrix)
        np.testing.assert_array_equal(self.matrix, before)
        np.testing.assert_array_equal(converted[:2, 2], before[:2, 2] - 0.5)
        np.testing.assert_array_equal(converted[:, :2], before[:, :2])

    def test_mixed_detection_dtypes_work_in_stereo(self):
        detections = [
            self.detections[0],
            {mid: corners.astype(np.float32) for mid, corners in self.detections[1].items()},
        ]
        with patch.object(pose, "detect_markers", side_effect=detections):
            result = pose.estimate_robot_pose(self.images, [self.matrix] * 2, [self.distortion] * 2)
        self.assertEqual(result["method"], "stereo")
        np.testing.assert_allclose(result["position_m"], self.trunk_position, atol=1e-5)

    def test_full_estimation_recovers_trunk_frame_and_offset(self):
        with patch.object(pose, "detect_markers", side_effect=self.detections):
            result = pose.estimate_robot_pose(self.images, [self.matrix] * 2, [self.distortion] * 2)
        self.assertEqual(result["method"], "stereo")
        np.testing.assert_allclose(result["position_m"], self.trunk_position, atol=1e-6)
        np.testing.assert_allclose(result["rotation_matrix"], self.trunk_rotation, atol=1e-5)

    def test_one_camera_retains_explicit_ambiguity(self):
        for index in (0, 1):
            with self.subTest(camera=index):
                detections = [{}, {}]
                detections[index] = self.detections[index]
                with patch.object(pose, "detect_markers", side_effect=detections):
                    result = pose.estimate_robot_pose(
                        self.images, [self.matrix] * 2, [self.distortion] * 2
                    )
                self.assertEqual(result["method"], "single_camera")
                self.assertEqual(result["cameras_used"], [index])
                self.assertTrue(result["warnings"])
                self.assertTrue(result["alternate_pose_candidates"])
                np.testing.assert_allclose(result["position_m"], self.trunk_position, atol=1e-5)

    def test_supplied_extrinsics_need_no_wall_markers(self):
        detections = [{5: d[5]} for d in self.detections]
        extrinsics = [np.column_stack([v[0], v[1]]) for v in self.views]
        with patch.object(pose, "detect_markers", side_effect=detections):
            with patch.object(
                pose.cv2, "solvePnP", side_effect=AssertionError("Refit calibration")
            ):
                result = pose.estimate_robot_pose(
                    self.images,
                    [self.matrix] * 2,
                    [self.distortion] * 2,
                    camera_extrinsics=extrinsics,
                )
        self.assertEqual(result["camera_calibration_source"], "supplied_extrinsics")
        self.assertEqual(result["wall_reprojection_rms_px"], [None, None])
        np.testing.assert_allclose(result["position_m"], self.trunk_position, atol=1e-6)
        np.testing.assert_allclose(result["rotation_matrix"], self.trunk_rotation, atol=1e-5)
        q = np.array(result["quaternion_wxyz"])
        self.assertAlmostEqual(np.linalg.norm(q), 1)
        # Reconstruct the matrix to verify w,x,y,z order and trunk-to-world direction.
        w, x, y, z = q
        reconstructed = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        np.testing.assert_allclose(reconstructed, self.trunk_rotation, atol=1e-5)

    def test_invalid_calibrated_rotation_is_rejected(self):
        extrinsics = [np.zeros((3, 4))] * 2
        with self.assertRaisesRegex(ValueError, "proper rotation"):
            pose.estimate_robot_pose(self.images, [self.matrix] * 2, camera_extrinsics=extrinsics)

    def test_missing_robot_never_returns_a_pose(self):
        detections = [{mid: c for mid, c in d.items() if mid != 5} for d in self.detections]
        with patch.object(pose, "detect_markers", side_effect=detections):
            with self.assertRaisesRegex(ValueError, "No usable camera"):
                pose.estimate_robot_pose(self.images, [self.matrix] * 2)

    def test_subpixel_refinement_finds_outer_edge(self):
        gray = np.full((200, 200), 255, np.uint8)
        gray[50:146, 50:146] = cv2.aruco.generateImageMarker(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 5, 96
        )
        coarse = np.array([[50.0, 50.0], [145.0, 50.0], [145.0, 145.0], [50.0, 145.0]])
        expected = np.array([[49.5, 49.5], [145.5, 49.5], [145.5, 145.5], [49.5, 145.5]])
        refined = pose.refine_corners(gray, coarse)
        self.assertLess(np.linalg.norm(refined - expected), np.linalg.norm(coarse - expected))
        np.testing.assert_array_equal(pose.refine_corners(gray, np.zeros((4, 2))), np.zeros((4, 2)))

    def test_tiny_marker_refines_border_without_following_internal_bits(self):
        gray = np.full((100, 100), 180, np.uint8)
        gray[40:58, 40:58] = cv2.aruco.generateImageMarker(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 5, 18
        )
        coarse = np.array([[40.0, 40.0], [57.0, 40.0], [57.0, 57.0], [40.0, 57.0]])
        expected = np.array([[39.5, 39.5], [57.5, 39.5], [57.5, 57.5], [39.5, 57.5]])
        np.testing.assert_allclose(pose.refine_corners(gray, coarse), expected, atol=0.05)

    def test_low_contrast_does_not_invent_refined_edges(self):
        gray = np.full((100, 100), 100, np.uint8)
        corners = np.array([[40.0, 40.0], [57.0, 40.0], [57.0, 57.0], [40.0, 57.0]])
        np.testing.assert_array_equal(pose.refine_corners(gray, corners), corners)

    def test_weak_border_section_does_not_tilt_the_whole_edge(self):
        gray = np.full((100, 100), 180, np.uint8)
        gray[40:58, 40:58] = cv2.aruco.generateImageMarker(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 5, 18
        )
        # Simulate lost contrast along part of the top black border.
        gray[40:42, 43:47] = 180
        coarse = np.array([[40.0, 40.0], [57.0, 40.0], [57.0, 57.0], [40.0, 57.0]])
        refined = pose.refine_corners(gray, coarse)
        self.assertLess(abs(refined[0, 1] - refined[1, 1]), 0.1)
        np.testing.assert_allclose(refined[:2, 1], [39.5, 39.5], atol=0.1)


if __name__ == "__main__":
    unittest.main()
