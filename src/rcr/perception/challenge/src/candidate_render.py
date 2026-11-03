"""Render a candidate layout for RGB verification. No evaluation input is read."""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .._paths import CACHE
from .rcr_methods import normalized_mesh

OUT = CACHE / "candidate_render"


def render_scene(holds, calibration, wall=None, volumes=None):
    spec = mujoco.MjSpec()
    mesh_path = OUT / "rcr_normalized.stl"
    if not mesh_path.exists():
        mesh_path.parent.mkdir(parents=True, exist_ok=True)
        normalized_mesh().export(mesh_path)
    spec.add_mesh(name="hold", file=str(mesh_path))
    if wall is not None:
        spec.worldbody.add_geom(
            name="wall",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[wall["width"] / 2, wall["height"] / 2, wall["depth"] / 2],
            pos=[0, wall["height"] / 2, -wall["depth"] / 2],
            contype=0,
            conaffinity=0,
        )
    for i, v in enumerate(volumes or []):
        x, y = v["center"]["x"], v["center"]["y"]
        w, h, z = v["baseWidth"], v["baseHeight"], v["depth"]
        points = [
            [x - w / 2, y - h / 2, 0],
            [x + w / 2, y - h / 2, 0],
            [x, y + h / 2, 0],
            [x, y, z],
        ]
        spec.add_mesh(
            name=f"volume{i}",
            uservert=np.array(points).ravel().tolist(),
            userface=[0, 1, 3, 1, 2, 3, 2, 0, 3, 0, 2, 1],
        )
        spec.worldbody.add_geom(
            name=f"volume{i}",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=f"volume{i}",
            contype=0,
            conaffinity=0,
        )
    for i, h in enumerate(holds.values()):
        R = Rotation.from_euler(
            "XYZ", [h["orientation"][a] for a in ["roll", "pitch", "yaw"]]
        ).as_matrix()
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, R.ravel())
        b = spec.worldbody.add_body(
            name=f"hold{i + 1}", pos=[h["position"][a] for a in "xyz"], quat=q
        )
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="hold", contype=0, conaffinity=0)
    c = calibration["cameras"]["left"]
    K = np.array(c["K"])
    T = np.array(c["T_camera_scene"])
    R, t = T[:3, :3], T[:3, 3]
    p = -R.T @ t
    axes = np.stack([R[0], -R[1], -R[2]], axis=1)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, axes.ravel())
    width, height = c["width"], c["height"]
    spec.worldbody.add_camera(
        name="left",
        pos=p,
        quat=q,
        resolution=[width, height],
        sensor_size=[float(width), float(height)],
        focal_length=[K[0, 0], K[1, 1]],
        principal_length=[width / 2 - K[0, 2], height / 2 - K[1, 2]],
    )
    model = spec.compile()
    model.vis.quality.offsamples = 0
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    model.vis.map.znear = 0.00001
    model.vis.map.zfar = 100
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        renderer.update_scene(data, camera="left")
        renderer.enable_depth_rendering()
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera="left")
        seg = renderer.render().copy()
        ids = np.zeros((height, width), np.int16)
        for gid in range(model.ngeom):
            name = model.body(int(model.geom_bodyid[gid])).name
            if name.startswith("hold"):
                ids[(seg[..., 0] == gid) & (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))] = int(
                    name[4:]
                )
        depth[seg[..., 0] < 0] = 0
        return depth, ids
    finally:
        renderer.close()
