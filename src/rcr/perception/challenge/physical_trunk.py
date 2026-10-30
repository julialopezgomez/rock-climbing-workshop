"""Experimental rigid-trunk fitting; moving links and encoder readings are excluded."""

import hashlib
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from .robot_pose.contenders.geometry import configured_mesh
from .robot_pose.model_adapter import RobotModel
from .src.cad_silhouette import SilhouetteRenderer


class TrunkRenderer(SilhouetteRenderer):
    def render_pose(self, pose, camera, origin):
        transform = np.asarray(camera["T_camera_wall"]) @ pose
        rotation, translation = transform[:3, :3], transform[:3, 3]
        self.model.cam_pos[0] = -rotation.T @ translation
        axes = np.stack([rotation[0], -rotation[1], -rotation[2]], axis=1)
        mujoco.mju_mat2Quat(self.model.cam_quat[0], axes.ravel())
        matrix = np.asarray(camera["K"])
        self.model.cam_intrinsic[0, :2] = np.diag(matrix)[:2] * self.scale
        self.model.cam_intrinsic[0, 2:] = (
            self.side / 2 - (matrix[:2, 2] + 0.5 - origin)
        ) * self.scale
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera="view")
        self.renderer.enable_segmentation_rendering()
        pixels = self.renderer.render()
        self.renderer.disable_segmentation_rendering()
        return (
            cv2.resize(
                (pixels[..., 0] >= 0).astype(np.float32),
                (self.side, self.side),
                interpolation=cv2.INTER_AREA,
            )
            > 0.5
        )


def seeds_from_label(label, mesh):
    points = np.asarray(label["outer_corners_wall_m"])
    centre = points.mean(0)
    x = (points[1] - points[0] + points[2] - points[3]) / 2
    _, _, rows = np.linalg.svd(points - centre)
    normal = rows[-1]
    if normal[2] < 0:
        normal *= -1
    x -= np.dot(x, normal) * normal
    x /= np.linalg.norm(x)
    bases = [np.column_stack([x, np.cross(normal, x), normal])]
    if label["square"] is not None:
        bases.append(np.asarray(label["square"]["T_wall_marker"])[:3, :3])
    # A wall-facing seed uses label roll but does not trust its occluded corners
    # to determine pitch/yaw. Optimization still evaluates both RGB views.
    angle = np.arctan2(x[1], x[0])
    bases.append(Rotation.from_euler("z", angle).as_matrix())
    back_centre = np.array([mesh.bounds[0, 0], 0, np.mean(mesh.bounds[:, 2])])
    seeds = []
    for basis in bases:
        u, _, v = np.linalg.svd(basis)
        basis = u @ v
        rotation = np.column_stack([-basis[:, 2], -basis[:, 0], basis[:, 1]])
        pose = np.eye(4)
        pose[:3, :3] = rotation
        pose[:3, 3] = centre - rotation @ back_centre
        seeds.append(pose)
    return seeds


def fit_trunk(images, cameras, robot_masks, label, cache):
    model = RobotModel()
    # Only root-body visual geometry is exported. The temporary joint vector
    # places no moving-link geometry into this rigid registration problem.
    mesh = configured_mesh(model, np.zeros(19), trunk_only=True)
    path = Path(cache) / "rigid_trunk.stl"
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)
    observations = []
    for image, camera, mask, corners in zip(
        images, cameras, robot_masks, label["outer_corners"], strict=True
    ):
        origin = np.floor(np.mean(corners, axis=0) - 48).astype(int)
        x, y = origin
        if x < 0 or y < 0 or x + 96 > image.shape[1] or y + 96 > image.shape[0]:
            raise ValueError("Trunk crop extends outside the RGB image")
        observed = mask[y : y + 96, x : x + 96].astype(bool)
        boundary = observed & ~cv2.erode(
            observed.astype(np.uint8), np.ones((3, 3), np.uint8)
        ).astype(bool)
        if boundary.sum() < 10:
            raise ValueError("Robot segmentation does not support the trunk crop")
        distance = cv2.distanceTransform(
            (~boundary).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 80, 160)[y : y + 96, x : x + 96] > 0
        edge_distance = cv2.distanceTransform(
            (~edges).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )
        observations.append(
            dict(
                camera=camera,
                origin=origin,
                mask=observed,
                distance=distance,
                edge_distance=edge_distance,
            )
        )

    renderer = TrunkRenderer(path, side=96, scale=2)

    def score(pose, details=False):
        losses, rows = [], []
        for observation in observations:
            mask = renderer.render_pose(pose, observation["camera"], observation["origin"])
            boundary = mask & ~cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(
                bool
            )
            if boundary.sum() < 10:
                return (100.0, []) if details else 100.0
            a = np.sort(np.minimum(observation["distance"][boundary], 8) ** 2)
            b = np.sort(np.minimum(observation["edge_distance"][boundary], 6) ** 2)
            outside = float(np.mean((~observation["mask"])[mask]))
            loss = float(
                np.mean(a[: max(5, len(a) // 2)])
                + 0.5 * np.mean(b[: max(5, len(b) * 3 // 4)])
                + 10 * outside
            )
            losses.append(loss)
            rows.append(dict(loss=loss, unsupported_area_fraction=outside))
        return (float(np.mean(losses)), rows) if details else float(np.mean(losses))

    candidates = []
    try:
        for index, seed in enumerate(seeds_from_label(label, mesh)):

            def pose(delta, seed=seed):
                value = seed.copy()
                value[:3, :3] = Rotation.from_rotvec(delta[:3]).as_matrix() @ seed[:3, :3]
                value[:3, 3] += delta[3:] / 1000
                return value

            fitted = minimize(
                lambda delta: score(pose(delta)),
                np.zeros(6),
                method="Powell",
                bounds=[(-0.35, 0.35)] * 3 + [(-20, 20), (-20, 20), (-30, 30)],
                options=dict(maxiter=18, xtol=0.05, ftol=1e-3),
            )
            candidates.append(
                dict(
                    seed=index,
                    initial_loss=score(seed),
                    loss=float(fitted.fun),
                    T_wall_trunk=pose(fitted.x).tolist(),
                    evaluations=fitted.nfev,
                )
            )
        best = min(candidates, key=lambda candidate: candidate["loss"])
        transform = np.asarray(best["T_wall_trunk"])
        _, views = score(transform, details=True)
        overlays = []
        for image, observation in zip(images, observations, strict=True):
            mask = renderer.render_pose(transform, observation["camera"], observation["origin"])
            rgb = image.copy()
            x, y = observation["origin"]
            patch = rgb[y : y + 96, x : x + 96]
            patch[mask] = (0.5 * patch[mask] + 0.5 * np.array([0, 255, 100])).astype(np.uint8)
            overlays.append(rgb)
        quaternion = Rotation.from_matrix(transform[:3, :3]).as_quat()[[3, 0, 1, 2]]
        if quaternion[0] < 0:
            quaternion *= -1
        return dict(
            parent_frame="provisional_board",
            child_frame="trunk",
            axes="x right, y up, z out from the wall",
            units="metres",
            quaternion_order="WXYZ",
            hypothesis_available=best["loss"] < 5,
            valid_for_control=False,
            method="two-view rigid-trunk CAD with trimmed RGB boundary support",
            T_wall_trunk=transform.tolist(),
            translation_m=transform[:3, 3].tolist(),
            rotation_wxyz=quaternion.tolist(),
            image_loss=best["loss"],
            view_diagnostics=views,
            candidates=candidates,
            mesh_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            measured_joints_used=False,
            joints_estimated=False,
            endpoints_estimated=False,
            marker_to_trunk_transform_used=False,
            initialization="Visible printed-label geometry and a wall-facing alternative",
            robot_label_used=True,
            markerless=False,
            confidence=None,
            scope=(
                "Experimental physical-image root hypothesis; "
                "no surveyed pose reference or synchronized capture"
            ),
        ), overlays
    finally:
        renderer.close()
