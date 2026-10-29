"""Estimate the trunk pose from two RGB images and camera intrinsics.

From the workshop root:
  MUJOCO_GL=egl third_party/simulator/.venv/bin/python scripts/robot_pose_aruco.py
Add --check to measure error against simulator truth (used only for evaluation).
Add --use-calibrated-extrinsics to use the simulator's configured camera poses
as supplied calibration, rather than estimating them from the wall images.

Import estimate_robot_pose(images, camera_matrices) to use your own RGB images.
Only the demo imports the simulator. Estimation uses no simulator ground truth.
Output frame: x right, y into the wall, z up; metres, quaternion (w,x,y,z).
Detection refines corners on original pixels. Stereo estimation fits a fixed
42 mm square to both images jointly; it does not change cameras or robot geometry.
Real images require measured colour intrinsics, distortion and synchronised views.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

MARKER_SIZE = 0.042
MARKER_CENTRE = np.array([-0.033, 0.0, 0.05475])
R_TRUNK_MARKER = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def _refine_small_marker_edges(gray, corners):
    """Fit the outer black border, avoiding the internal bits of a tiny code.

    Dark-going intensity gradients locate each edge on original source pixels.
    Intersections of robust line fits give the four subpixel outer corners.
    """
    lines = []
    offsets = np.linspace(-1.5, 1.5, 49)
    gradient_positions = (offsets[:-1] + offsets[1:]) / 2
    for a, b in zip(corners, np.roll(corners, -1, axis=0), strict=True):
        edge = b - a
        normal = np.array([-edge[1], edge[0]]) / np.linalg.norm(edge)
        centres = a + np.linspace(0.15, 0.85, 16)[:, None] * edge
        coordinates = centres[:, None, :] + offsets[None, :, None] * normal
        intensities = cv2.remap(
            gray.astype(np.float32),
            coordinates[:, :, 0].astype(np.float32),
            coordinates[:, :, 1].astype(np.float32),
            cv2.INTER_LINEAR,
        )
        weights = np.maximum(-np.diff(intensities, axis=1), 0)
        contrast = weights.sum(axis=1)
        valid = contrast > 15  # insufficient contrast: keep original corners
        if valid.sum() < 6:
            return corners.astype(float)
        shifts = (weights @ gradient_positions) / np.maximum(contrast, 1e-9)
        points = centres[valid] + shifts[valid, None] * normal
        vx, vy, x, y = cv2.fitLine(points.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        line_normal = np.array([-vy, vx])
        lines.append((line_normal, line_normal @ np.array([x, y])))
    refined = []
    for index in range(4):
        matrix = np.stack([lines[index - 1][0], lines[index][0]])
        if abs(np.linalg.det(matrix)) < 0.1:
            return corners.astype(float)
        refined.append(np.linalg.solve(matrix, [lines[index - 1][1], lines[index][1]]))
    refined = np.asarray(refined)
    if np.max(np.linalg.norm(refined - corners, axis=1)) > 1.5:
        return corners.astype(float)
    return refined


def refine_corners(gray, corners):
    """Refine original pixels: border lines for tiny codes, subpixel windows otherwise."""
    corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    edge = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1).min()
    window = max(1, min(3, int(edge / 12)))
    if np.any(corners < window + 1) or np.any(
        corners > np.array([gray.shape[1], gray.shape[0]]) - window - 2
    ):
        return corners.astype(float)
    if edge < 24:
        if edge < 8:
            return corners.astype(float)
        return _refine_small_marker_edges(gray, corners)
    refined = cv2.cornerSubPix(
        gray,
        corners.copy(),
        (window, window),
        (-1, -1),
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001),
    )
    # A refinement that jumps into the code is not a trustworthy outer corner.
    if np.max(np.linalg.norm(refined - corners, axis=1)) > min(1.5, edge / 6):
        return corners.astype(float)
    return refined.astype(float)


def _joint_pose_fit(local, initial_rotation, initial_centre, views):
    """Fit one rigid square to both original images using damped least squares.

    Camera calibration remains fixed. All six free parameters describe only the
    marker's pose; no marker dimensions, robot geometry or camera poses are fitted.
    Uses only NumPy/OpenCV, so the standalone simulator environment needs no SciPy.
    """
    initial = np.r_[cv2.Rodrigues(initial_rotation)[0].ravel(), initial_centre]

    def residual(pose):
        rotation = cv2.Rodrigues(pose[:3])[0]
        world = local @ rotation.T + pose[3:]
        errors = []
        for camera_rotation, camera_translation, matrix, distortion, pixels in views:
            camera = world @ camera_rotation.T + camera_translation
            if np.any(camera[:, 2] <= 0):
                return None
            projected = cv2.projectPoints(
                world,
                cv2.Rodrigues(camera_rotation)[0],
                camera_translation,
                matrix,
                distortion,
            )[0][:, 0]
            errors.append((projected - pixels).ravel())
        return np.concatenate(errors)

    pose = initial.copy()
    error = residual(pose)
    if error is None:
        raise ValueError("Initial marker pose is behind a camera.")
    initial_error = error.copy()
    damping = 1e-3
    iterations = 0
    for iteration in range(1, 61):
        iterations = iteration
        jacobian = np.empty((len(error), 6))
        for column in range(6):
            step = np.zeros(6)
            step[column] = 1e-6
            plus, minus = residual(pose + step), residual(pose - step)
            if plus is None or minus is None:
                raise ValueError("Marker pose refinement reached invalid camera geometry.")
            jacobian[:, column] = (plus - minus) / (2e-6)
        normal = jacobian.T @ jacobian
        diagonal = np.maximum(np.diag(normal), 1e-9)
        delta = np.linalg.solve(normal + damping * np.diag(diagonal), -jacobian.T @ error)
        trial_error = residual(pose + delta)
        if trial_error is not None and trial_error @ trial_error < error @ error:
            improvement = error @ error - trial_error @ trial_error
            pose += delta
            error = trial_error
            damping = max(damping / 3, 1e-12)
            if np.linalg.norm(delta) < 1e-9 or improvement < 1e-12:
                break
        else:
            damping *= 10
            if damping > 1e12:
                break
    return (
        cv2.Rodrigues(pose[:3])[0],
        pose[3:],
        np.sqrt(np.mean(error.reshape(-1, 4, 2) ** 2, axis=(1, 2)) * 2).tolist(),
        np.sqrt(np.mean(initial_error.reshape(-1, 4, 2) ** 2, axis=(1, 2)) * 2).tolist(),
        iterations,
    )


def detect_markers(image):
    """Detect RGB-image markers, retrying small codes at several image scales.

    Enlargement helps OpenCV's threshold/contour stages; it cannot recover an
    occluded code. Returned corners always use the original image coordinates.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detected = {}
    variants = [(1, None, False)] + [
        (scale, interpolation, True)
        for scale in (2, 3, 4)
        for interpolation in (cv2.INTER_CUBIC, cv2.INTER_NEAREST)
    ]
    # Some rendered 15 px codes lose their border under the default threshold.
    variants.append((6, cv2.INTER_NEAREST, True))
    for scale, interpolation, tuned in variants:
        parameters = cv2.aruco.DetectorParameters()
        if tuned:
            parameters.adaptiveThreshWinSizeMax = 53
            parameters.adaptiveThreshWinSizeStep = 4
            parameters.minMarkerPerimeterRate = 0.005
        if scale == 6:
            parameters.adaptiveThreshWinSizeMax = 73
            parameters.adaptiveThreshConstant = 0
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        enlarged = (
            gray
            if scale == 1
            else cv2.resize(gray, None, fx=scale, fy=scale, interpolation=interpolation)
        )
        corners, ids, _ = detector.detectMarkers(enlarged)
        if ids is not None:
            for mid, corner in zip(ids.ravel(), corners, strict=True):
                # resize() maps pixel centres with this half-pixel convention.
                detected.setdefault(int(mid), (corner[0].astype(float) + 0.5) / scale - 0.5)
        if 5 in detected and len(set(detected) & {1, 2, 3, 4}) >= 2:
            break
    return {mid: refine_corners(gray, corner) for mid, corner in detected.items()}


def _wall_corners(marker_id):
    """Black code corners, in canonical OpenCV order, in workshop coordinates."""
    x = 0.0 if marker_id in (1, 3) else 0.800 - 0.136
    z = 1.400 - 0.136 if marker_id in (1, 2) else 0.0
    # The simulator's marker front is 1 mm in front of the board.
    return np.array(
        [
            [x + 0.018, -0.001, z + 0.118],
            [x + 0.118, -0.001, z + 0.118],
            [x + 0.118, -0.001, z + 0.018],
            [x + 0.018, -0.001, z + 0.018],
        ]
    )


def _quaternion(rotation):
    """Rotation matrix -> unit quaternion (w,x,y,z), including 180 degree rotations."""
    rvec = cv2.Rodrigues(rotation)[0].ravel()
    angle = np.linalg.norm(rvec)
    if angle < 1e-12:
        return [1.0, 0.0, 0.0, 0.0]
    return np.r_[np.cos(angle / 2), rvec / angle * np.sin(angle / 2)].tolist()


def estimate_robot_pose(images, camera_matrices, distortion_coeffs=None, camera_extrinsics=None):
    """Estimate trunk pose, using stereo when possible or one camera as a fallback.

    images: two RGB uint8 arrays, synchronised, from different cameras.
    camera_matrices: corresponding calibrated 3x3 OpenCV intrinsic matrices.
    distortion_coeffs: optional two OpenCV distortion arrays; simulator uses none.
    camera_extrinsics: optional two calibrated 3x4 world-to-camera matrices
    [R|t], with translation in metres. World = x right, y into wall, z up.
    With these supplied, camera poses stay fixed and wall markers are optional
    validation observations. Intrinsics alone still require two wall markers.
    Quaternion rotates trunk-frame vectors into this world frame, in w,x,y,z order.
    The vertical-wall assumption defines the frame, not the robot's orientation.
    A single-camera
    result includes an ambiguity warning and any alternate planar-pose solution.
    Raises ValueError when neither image provides enough detections.
    """
    if len(images) != 2 or len(camera_matrices) != 2:
        raise ValueError("Provide exactly two RGB images and their intrinsic matrices.")
    distortion_coeffs = [None, None] if distortion_coeffs is None else distortion_coeffs
    if len(distortion_coeffs) != 2:
        raise ValueError("Provide one distortion array per camera.")
    if camera_extrinsics is not None:
        if len(camera_extrinsics) != 2:
            raise ValueError("Provide one calibrated 3x4 extrinsic matrix per camera.")
        for extrinsic in camera_extrinsics:
            extrinsic = np.asarray(extrinsic, dtype=float)
            if extrinsic.shape != (3, 4) or not np.isfinite(extrinsic).all():
                raise ValueError("Camera extrinsics must be finite 3x4 world-to-camera matrices.")
            rotation = extrinsic[:, :3]
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(
                np.linalg.det(rotation), 1, atol=1e-6
            ):
                raise ValueError("Camera extrinsics must contain a proper rotation matrix.")
    calibration_source = "supplied_extrinsics" if camera_extrinsics is not None else "wall_markers"
    projections, robot_pixels, wall_errors, cameras_used, unavailable = [], [], [], [], []
    views = []
    for index, (image, matrix, distortion) in enumerate(
        zip(images, camera_matrices, distortion_coeffs, strict=True)
    ):
        matrix = np.asarray(matrix, dtype=float)
        detected = detect_markers(image)
        if 5 not in detected:
            unavailable.append(f"Camera {index}: ID 5 not detected (IDs: {sorted(detected)}).")
            continue
        wall_ids = sorted(set(detected) & {1, 2, 3, 4})
        if camera_extrinsics is None and len(wall_ids) < 2:
            unavailable.append(f"Camera {index}: need at least two wall markers (IDs 1–4).")
            continue
        if camera_extrinsics is not None:
            extrinsic = np.asarray(camera_extrinsics[index], dtype=float)
            rotation, tvec = extrinsic[:, :3], extrinsic[:, 3].reshape(3, 1)
            rvec = cv2.Rodrigues(rotation)[0]
        else:
            object_points = np.concatenate([_wall_corners(mid) for mid in wall_ids])
            image_points = np.concatenate([detected[mid] for mid in wall_ids])
            ok, rvec, tvec = cv2.solvePnP(object_points, image_points, matrix, distortion)
            if not ok:
                raise ValueError(f"Camera {index}: wall calibration failed.")
            rotation = cv2.Rodrigues(rvec)[0]
        if wall_ids:
            object_points = np.concatenate([_wall_corners(mid) for mid in wall_ids])
            image_points = np.concatenate([detected[mid] for mid in wall_ids])
            if np.any((object_points @ rotation.T + tvec.ravel())[:, 2] <= 0):
                raise ValueError(f"Camera {index}: wall pose is behind the camera.")
            projected = cv2.projectPoints(object_points, rvec, tvec, matrix, distortion)[0][:, 0]
            wall_errors.append(
                float(np.sqrt(np.mean(np.sum((projected - image_points) ** 2, axis=1))))
            )
        else:
            wall_errors.append(None)
        projections.append(np.column_stack([rotation, tvec]))
        robot_pixels.append(cv2.undistortPoints(detected[5][:, None], matrix, distortion)[:, 0])
        cameras_used.append(index)
        views.append((rotation, tvec.ravel(), matrix, distortion, detected[5]))

    if not projections:
        raise ValueError(
            "No usable camera. "
            + " ".join(unavailable)
            + " Run with --debug-dir ~/robot-pose-debug to save camera images."
        )

    half = MARKER_SIZE / 2
    local = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]])
    if len(projections) == 1:
        # IPPE returns both planar pose hypotheses. Do not conceal the ambiguity.
        solved, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            local, robot_pixels[0], np.eye(3), None, flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        if not solved:
            raise ValueError("Single-camera marker pose could not be solved.")
        camera_rotation = projections[0][:, :3]
        camera_translation = projections[0][:, 3]
        candidates = []
        for rvec, tvec in zip(rvecs, tvecs, strict=True):
            marker_rotation = cv2.Rodrigues(rvec)[0]
            if np.any((local @ marker_rotation.T + tvec.ravel())[:, 2] <= 0):
                continue
            rotation_world = camera_rotation.T @ marker_rotation
            centre_world = camera_rotation.T @ (tvec.ravel() - camera_translation)
            trunk_rotation = rotation_world @ R_TRUNK_MARKER.T
            trunk_position = centre_world - trunk_rotation @ MARKER_CENTRE
            projected = cv2.projectPoints(local, rvec, tvec, np.eye(3), None)[0][:, 0]
            error = float(np.sqrt(np.mean(np.sum((projected - robot_pixels[0]) ** 2, axis=1))))
            candidates.append(
                {
                    "position_m": trunk_position.tolist(),
                    "quaternion_wxyz": _quaternion(trunk_rotation),
                    "rotation_matrix": trunk_rotation.tolist(),
                    "marker_reprojection_rms_normalized": error,
                }
            )
        if not candidates:
            raise ValueError("Single-camera marker pose is behind the camera.")
        candidates.sort(key=lambda candidate: candidate["marker_reprojection_rms_normalized"])
        return {
            **candidates[0],
            "method": "single_camera",
            "camera_calibration_source": calibration_source,
            "cameras_used": cameras_used,
            "warnings": unavailable
            + [
                "Lower confidence: a small planar marker in one camera has depth and "
                "orientation uncertainty. The lowest reprojection-error solution is shown; "
                "check alternate_pose_candidates before relying on orientation."
            ],
            "alternate_pose_candidates": candidates[1:],
            "wall_reprojection_rms_px": wall_errors,
        }

    homogeneous = cv2.triangulatePoints(*projections, robot_pixels[0].T, robot_pixels[1].T)
    if np.any(np.abs(homogeneous[3]) < 1e-10):
        raise ValueError("Stereo geometry is degenerate; use two separated cameras.")
    points = (homogeneous[:3] / homogeneous[3]).T
    for projection in projections:
        if np.any((points @ projection[:, :3].T + projection[:, 3])[:, 2] <= 0):
            raise ValueError("Triangulated robot marker is behind a camera.")

    # Fit the known 42 mm square to stereo corners, giving a proper rigid rotation.
    centre = points.mean(axis=0)
    u, _, vt = np.linalg.svd(local.T @ (points - centre))
    rotation_marker = vt.T @ np.diag([1, 1, np.linalg.det(vt.T @ u.T)]) @ u.T
    rotation_marker, centre, reprojection, initial_reprojection, iterations = _joint_pose_fit(
        local, rotation_marker, centre, views
    )
    rotation_trunk = rotation_marker @ R_TRUNK_MARKER.T
    position_trunk = centre - rotation_trunk @ MARKER_CENTRE
    fit_error = np.sqrt(np.mean(np.sum((local @ rotation_marker.T + centre - points) ** 2, axis=1)))
    return {
        "method": "stereo",
        "camera_calibration_source": calibration_source,
        "cameras_used": cameras_used,
        "position_m": position_trunk.tolist(),
        "quaternion_wxyz": _quaternion(rotation_trunk),
        "rotation_matrix": rotation_trunk.tolist(),
        "wall_reprojection_rms_px": wall_errors,
        "marker_fit_rms_m": float(fit_error),
        "marker_reprojection_rms_px": reprojection,
        "initial_marker_reprojection_rms_px": initial_reprojection,
        "refinement_iterations": iterations,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wall", default="N2", help="Bundled wall name or wall JSON path")
    parser.add_argument(
        "--check", action="store_true", help="Compare estimate to simulator ground truth"
    )
    parser.add_argument("--debug-dir", type=Path, help="Save camera images and detection overlays")
    parser.add_argument(
        "--use-calibrated-extrinsics",
        action="store_true",
        help="Use fixed camera poses from simulator configuration; no wall pose fitting",
    )
    args = parser.parse_args()
    from climbing_sim import ClimbingSim

    sim = ClimbingSim(args.wall)
    try:
        sim.reset_hanging()
        cameras = list(sim.cameras.values())
        extrinsics = None
        if args.use_calibrated_extrinsics:
            # Camera configuration uses x right/y up/z out, whereas output uses
            # x right/y into wall/z up. No robot state is read for this calibration.
            world_to_camera_wall = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])
            extrinsics = [
                np.column_stack(
                    [
                        camera.R_wall_to_cam() @ world_to_camera_wall,
                        np.asarray(camera.tvec, dtype=float),
                    ]
                )
                for camera in cameras
            ]
        images = [sim.render(camera.name) for camera in cameras]
        if args.debug_dir:
            args.debug_dir.mkdir(parents=True, exist_ok=True)
            for camera, image in zip(cameras, images, strict=True):
                detected = detect_markers(image)
                bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(args.debug_dir / f"{camera.name}.png"), bgr)
                if detected:
                    cv2.aruco.drawDetectedMarkers(
                        bgr,
                        [c[None].astype(np.float32) for c in detected.values()],
                        np.array(list(detected), dtype=np.int32)[:, None],
                    )
                cv2.imwrite(str(args.debug_dir / f"{camera.name}_detected.png"), bgr)
        try:
            result = estimate_robot_pose(
                images, [camera.K() for camera in cameras], camera_extrinsics=extrinsics
            )
        except ValueError as error:
            parser.exit(1, f"Pose unavailable: {error}\n")
        result["frame"] = "wall: x right, y into wall, z up"
        if args.check:
            # Ground truth is used only here to evaluate the finished image estimate.
            trunk = sim.data.body("trunk")
            delta = np.asarray(result["rotation_matrix"]).T @ trunk.xmat.reshape(3, 3)
            result["validation"] = {
                "true_position_m": trunk.xpos.tolist(),
                "position_error_m": float(
                    np.linalg.norm(np.array(result["position_m"]) - trunk.xpos)
                ),
                "orientation_error_deg": float(
                    np.degrees(np.arccos(np.clip((np.trace(delta) - 1) / 2, -1, 1)))
                ),
            }
        print(json.dumps(result, indent=2))
    finally:
        sim.close()


if __name__ == "__main__":
    main()
