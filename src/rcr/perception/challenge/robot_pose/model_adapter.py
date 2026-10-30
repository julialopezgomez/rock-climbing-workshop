"""Read-only climbing CAD and FK. No running simulator state is consulted."""

import os
from pathlib import Path

from .._paths import CACHE, model_path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np

from .frame_conventions import to_wxyz

ROOT = CACHE
MODEL = None
JOINTS = [
    "head",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_elbow",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_elbow",
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
]
LIMBS = ["left_hand", "right_hand", "left_foot", "right_foot"]


class RobotModel:
    def __init__(self, path=MODEL):
        self.path = Path(path) if path is not None else model_path()
        self.model = mujoco.MjModel.from_xml_path(str(self.path))
        self.data = mujoco.MjData(self.model)
        self.qadr = np.array([self.model.jnt_qposadr[self.model.joint(j).id] for j in JOINTS])
        self.limits = np.array([self.model.jnt_range[self.model.joint(j).id] for j in JOINTS])
        self.root = self.model.body("trunk").id

    def set_joints(self, q):
        if isinstance(q, dict):
            if set(q) != set(JOINTS):
                raise ValueError("Joint packet must provide exactly the 19 named climbing joints")
            q = [q[j] for j in JOINTS]
        q = np.asarray(q, float)
        if q.shape != (19,) or not np.isfinite(q).all():
            raise ValueError("Expected 19 finite measured joint angles")
        self.data.qpos[:3] = 0
        self.data.qpos[3:7] = [1, 0, 0, 0]
        self.data.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, self.data)
        return q

    def endpoints(self, q):
        self.set_joints(q)
        return {name: self.data.site(name + "_ball").xpos.copy() for name in LIMBS}

    def landmarks(self, q):
        self.set_joints(q)
        result = self.endpoints(q)
        # Fixed CAD coordinates on the rigid trunk; not automatically observable
        # image landmarks. Keypoint frontends must identify these explicitly.
        for i, p in enumerate(
            [
                [-0.030, -0.045, 0.025],
                [-0.030, 0.045, 0.025],
                [-0.030, -0.045, 0.095],
                [-0.030, 0.045, 0.095],
                [0.025, -0.04, 0.02],
                [0.025, 0.04, 0.10],
            ]
        ):
            result[f"trunk_{i}"] = np.array(p)
        return result


class RobotRenderer(RobotModel):
    def __init__(self, side=192, scale=1, path=MODEL):
        path = model_path() if path is None else Path(path)
        spec = mujoco.MjSpec.from_file(str(path))
        side = int(side)
        self.side = side
        self.scale = int(scale)
        size = side * self.scale
        spec.worldbody.add_camera(
            name="pose_camera",
            resolution=[size, size],
            sensor_size=[float(size)] * 2,
            focal_length=[900.0, 900.0],
        )
        self.path = Path(path)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.qadr = np.array([self.model.jnt_qposadr[self.model.joint(j).id] for j in JOINTS])
        self.root = self.model.body("trunk").id
        self.cid = self.model.camera("pose_camera").id
        self.model.vis.quality.offsamples = 0
        self.model.vis.global_.offwidth = self.model.vis.global_.offheight = size
        self.model.vis.map.znear = 0.00001
        self.renderer = mujoco.Renderer(self.model, height=size, width=size)
        self.option = mujoco.MjvOption()
        self.option.geomgroup[:] = [1, 1, 1, 0, 0, 0]
        self.geometry = np.arange(self.model.ngeom)
        self._trunk_mask = self.model.geom_bodyid == self.root
        self.renders = 0

    def render(self, t_wall_trunk, camera, origin, only_trunk=False):
        import cv2

        tc = np.asarray(camera["T_camera_wall"]) @ t_wall_trunk
        r, t = tc[:3, :3], tc[:3, 3]
        c = self.cid
        self.model.cam_pos[c] = -r.T @ t
        self.model.cam_quat[c] = to_wxyz(np.stack([r[0], -r[1], -r[2]], axis=1))
        k = np.asarray(camera["K"])
        self.model.cam_intrinsic[c, :2] = [k[0, 0] * self.scale, k[1, 1] * self.scale]
        # Inputs use OpenCV pixel-centre coordinates. MuJoCo's sensor lengths
        # use raster edges: pixel(0,0) is sampled at(.5,.5).
        self.model.cam_intrinsic[c, 2:] = [
            (self.side / 2 - (k[0, 2] + 0.5 - origin[0])) * self.scale,
            (self.side / 2 - (k[1, 2] + 0.5 - origin[1])) * self.scale,
        ]
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=c, scene_option=self.option)
        if only_trunk:
            for i in range(self.renderer.scene.ngeom):
                g = self.renderer.scene.geoms[i]
                if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and not self._trunk_mask[g.objid]:
                    g.rgba[3] = 0
        self.renderer.enable_segmentation_rendering()
        seg = self.renderer.render()
        self.renderer.disable_segmentation_rendering()
        mask = (seg[..., 0] >= 0).astype(np.float32)
        self.renders += 1
        return (
            cv2.resize(mask, (self.side, self.side), interpolation=cv2.INTER_AREA)
            if self.scale != 1
            else mask
        )

    def close(self):
        self.renderer.close()
