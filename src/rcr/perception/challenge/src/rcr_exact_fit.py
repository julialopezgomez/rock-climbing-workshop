"""Budgeted dual-view CAD fitting with exact final raster verification.

Inputs are RGB images, public stereo calibration and public CAD. Geometry is the
source's wall-or-volume mounting family: normals face forwards and local yaw is
one of three values. Wall z=0 is a competing hypothesis, not a GT surface label.
All final candidates are ranked against observed RGB masks, never true poses.
"""

from __future__ import annotations

import hashlib
import os

from .._paths import CACHE

os.environ.setdefault("MUJOCO_GL", "egl")
import cv2
import numpy as np

from .rcr_methods import (
    normalized_mesh,
)

OUT = CACHE / "hold_fit"


class ExactRenderer:
    """Supersampled soft silhouette; image crop coordinates use OpenCV K."""

    def __init__(
        self,
        scale=4,
        side=64,
        wall=False,
        volumes=None,
        *,
        kind="microban_hold",
        wall_size=(0.8, 1.4),
        volume_height=0.08,
    ):
        import mujoco

        self.mujoco = mujoco
        self.scale = scale
        self.side = side
        mesh = normalized_mesh(kind)
        OUT.mkdir(parents=True, exist_ok=True)
        signature = hashlib.sha256(mesh.vertices.tobytes() + mesh.faces.tobytes()).hexdigest()[:16]
        path = OUT / f"{kind}_{signature}.stl"
        if not path.exists():
            temporary = path.with_name(f"{path.stem}.{os.getpid()}.stl")
            mesh.export(temporary)
            temporary.replace(path)
        spec = mujoco.MjSpec()
        spec.add_mesh(name="hold", file=str(path))
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_MESH, meshname="hold", contype=0, conaffinity=0
        )
        self.wall = wall
        if wall:
            spec.worldbody.add_geom(
                name="wall_occluder",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[wall_size[0] / 2, wall_size[1] / 2, 0.0125],
                pos=[0, wall_size[1] / 2, -0.0125],
                contype=0,
                conaffinity=0,
            )
        for i, v in enumerate(volumes or []):
            x, y, w, d = v
            verts = [
                [x - w / 2, y - volume_height / 2, 0],
                [x + w / 2, y - volume_height / 2, 0],
                [x, y + volume_height / 2, 0],
                [x, y, d],
            ]
            spec.add_mesh(
                name=f"volume_{i}",
                uservert=np.asarray(verts).ravel().tolist(),
                userface=[0, 1, 3, 1, 2, 3, 2, 0, 3, 0, 2, 1],
            )
            spec.worldbody.add_geom(
                type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"volume_{i}", contype=0, conaffinity=0
            )
        size = side * scale
        spec.worldbody.add_camera(
            name="view",
            resolution=[size, size],
            sensor_size=[float(size)] * 2,
            focal_length=[900, 900],
        )
        self.model = spec.compile()
        self.model.vis.quality.offsamples = 0
        self.model.vis.global_.offwidth = self.model.vis.global_.offheight = size
        self.model.vis.map.znear = 0.00001
        self.data = mujoco.MjData(self.model)
        self.context_positions = self.model.geom_pos.copy()
        self.context_rotations = []
        for quat in self.model.geom_quat:
            mat = np.empty(9)
            mujoco.mju_quat2Mat(mat, quat)
            self.context_rotations.append(mat.reshape(3, 3))
        self.renderer = mujoco.Renderer(self.model, height=size, width=size)
        self.center = np.asarray(mesh.vertices).mean(0)
        self.renders = 0

    def render_soft(self, point, R, K, T, origin):
        r = T[:3, :3] @ R
        t = T[:3, :3] @ point + T[:3, 3]
        if t[2] <= 0.05:
            return np.zeros((self.side, self.side), np.float32)
        self.model.cam_pos[0] = -r.T @ t
        axes = np.stack([r[0], -r[1], -r[2]], axis=1)
        self.mujoco.mju_mat2Quat(self.model.cam_quat[0], axes.ravel())
        self.model.cam_intrinsic[0, :2] = [K[0, 0] * self.scale, K[1, 1] * self.scale]
        self.model.cam_intrinsic[0, 2:] = [
            self.side * self.scale / 2 - (K[0, 2] - origin[0]) * self.scale,
            self.side * self.scale / 2 - (K[1, 2] - origin[1]) * self.scale,
        ]
        self.mujoco.mj_forward(self.model, self.data)
        if self.model.ngeom > 1:
            # World-body geom transforms are cached at initialization in
            # MuJoCo; changing model.geom_pos alone does not refresh them.
            for gid in range(1, self.model.ngeom):
                self.data.geom_xpos[gid] = R.T @ (self.context_positions[gid] - point)
                self.data.geom_xmat[gid] = (R.T @ self.context_rotations[gid]).ravel()
        self.renderer.update_scene(self.data, camera="view")
        self.renderer.enable_segmentation_rendering()
        seg = self.renderer.render()
        self.renderer.disable_segmentation_rendering()
        self.renders += 1
        return cv2.resize(
            (seg[..., 0] == 0).astype(np.float32),
            (self.side, self.side),
            interpolation=cv2.INTER_AREA,
        )

    def depth_patch(self, point, R, K, T, origin):
        mask = self.render_soft(point, R, K, T, origin)
        self.renderer.enable_depth_rendering()
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        return cv2.resize(depth, (self.side, self.side), interpolation=cv2.INTER_AREA), mask

    def render(self, point, R, K, T, origin):
        # Compatibility with the cheap initializer; final scoring below rerenders.
        return self.render_soft(point, R, K, T, origin) > 0.5

    def close(self):
        self.renderer.close()


def crop_mask(mask, origin, side=64):
    x, y = map(int, origin)
    result = np.zeros((side, side), np.float32)
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + side, mask.shape[1]), min(y + side, mask.shape[0])
    if x2 > x1 and y2 > y1:
        result[y1 - y : y2 - y, x1 - x : x2 - x] = mask[y1:y2, x1:x2]
    return result
