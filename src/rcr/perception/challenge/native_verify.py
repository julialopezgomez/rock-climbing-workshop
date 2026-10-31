"""Check regenerated geometry against both RGB views; never read scene labels."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
import cv2
import mujoco
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

from ._paths import CACHE
from .src.hold_scene import proposals
from .src.rcr_methods import normalized_mesh
from .src.rcr_native_geometry import flat_faces

COLORS = ["#F2554E", "#59C77B", "#6389ED"]


def consensus(estimate, candidate):
    a, b = list(estimate.values()), list(candidate.values())
    if not a or not b:
        return 0
    p = np.array([[h["position"][k] for k in "xyz"] for h in a])
    q = np.array([[h["position"][k] for k in "xyz"] for h in b])
    distances = np.linalg.norm(p[:, None] - q[None], axis=-1)
    valid = (distances < 0.006) & np.array(
        [
            [x["type"] == y["type"] and x["color"].upper() == y["color"].upper() for y in b]
            for x in a
        ]
    )
    i, j = linear_sum_assignment(-valid.astype(float) + np.minimum(distances, 1) * 0.001)
    return int(valid[i, j].sum())


def render_ids(scene, calibration):
    spec = mujoco.MjSpec()
    cache = CACHE / "native_verify"
    cache.mkdir(exist_ok=True, parents=True)
    kinds = sorted({h["type"] for h in scene["holds"].values()})
    for kind in kinds:
        path = cache / (kind + ".stl")
        if not path.exists():
            normalized_mesh(kind).export(path)
        spec.add_mesh(name=kind, file=str(path))
    wall = scene["scene"]["wall"]
    spec.worldbody.add_geom(
        name="wall",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[wall["width"] / 2, wall["height"] / 2, wall["depth"] / 2],
        pos=[0, wall["height"] / 2, -wall["depth"] / 2],
        contype=0,
        conaffinity=0,
    )
    for i, volume in enumerate(scene["scene"]["volumes"]):
        x, y = volume["center"]["x"], volume["center"]["y"]
        w, h, d = volume["baseWidth"], volume["baseHeight"], volume["depth"]
        vertices = [
            [x - w / 2, y - h / 2, 0],
            [x + w / 2, y - h / 2, 0],
            [x, y + h / 2, 0],
            [x, y, d],
        ]
        spec.add_mesh(
            name=f"volume{i}",
            uservert=np.array(vertices).ravel().tolist(),
            userface=[0, 1, 3, 1, 2, 3, 2, 0, 3, 0, 2, 1],
        )
        spec.worldbody.add_geom(
            name=f"volume{i}",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=f"volume{i}",
            contype=0,
            conaffinity=0,
        )
    for i, hold in enumerate(scene["holds"].values()):
        q = Rotation.from_euler(
            "XYZ", [hold["orientation"][a] for a in ("roll", "pitch", "yaw")]
        ).as_quat()[[3, 0, 1, 2]]
        spec.worldbody.add_geom(
            name=f"hold{i + 1}",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=hold["type"],
            pos=[hold["position"][a] for a in "xyz"],
            quat=q,
            contype=0,
            conaffinity=0,
        )
    width, height = calibration["image_width"], calibration["image_height"]
    for eye in ("left", "right"):
        camera = calibration["cameras"][eye]
        k, t = np.array(camera["K"]), np.array(camera["T_camera_scene"])
        r = t[:3, :3]
        q = Rotation.from_matrix(np.stack([r[0], -r[1], -r[2]], axis=1)).as_quat()[[3, 0, 1, 2]]
        spec.worldbody.add_camera(
            name=eye,
            pos=-r.T @ t[:3, 3],
            quat=q,
            resolution=[width, height],
            sensor_size=[float(width), float(height)],
            focal_length=[k[0, 0], k[1, 1]],
            principal_length=[width / 2 - k[0, 2], height / 2 - k[1, 2]],
        )
    model = spec.compile()
    model.vis.quality.offsamples = 0
    model.vis.global_.offwidth, model.vis.global_.offheight = width, height
    model.vis.map.znear = 0.00001
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        renderer.enable_segmentation_rendering()
        outputs = []
        for eye in ("left", "right"):
            renderer.update_scene(data, camera=eye)
            seg = renderer.render().copy()
            ids = np.zeros((height, width), np.int16)
            for i in range(len(scene["scene"]["volumes"])):
                gid = model.geom(f"volume{i}").id
                ids[(seg[..., 0] == gid) & (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))] = -i - 1
            for i in range(len(scene["holds"])):
                gid = model.geom(f"hold{i + 1}").id
                ids[(seg[..., 0] == gid) & (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))] = i + 1
            outputs.append(ids)
        return outputs
    finally:
        renderer.close()


def observation_masks(images):
    """Use detected components globally and allow shading within a known region."""
    observations = []
    for rgb in images:
        detected = np.zeros((*rgb.shape[:2], 3), bool)
        for proposal in proposals(rgb):
            detected[..., proposal["hue_group"]] |= proposal["mask"]
        red, green, blue = np.moveaxis(rgb.astype(np.int16), -1, 0)
        support = np.stack(
            [
                (red >= 200) & (red - green > 20) & (red - blue > 20),
                (green >= 175) & (green - red > 20) & (green - blue > 5),
                (blue >= 200) & (blue - red > 15) & (blue >= green),
            ],
            axis=-1,
        )
        observations.append((detected, support, flat_faces(rgb)[1]))
    return observations


def image_loss(scene, images, calibration, observations=None):
    rendered = render_ids(scene, calibration)
    if observations is None:
        observations = observation_masks(images)
    colors = [COLORS.index(h["color"].upper()) for h in scene["holds"].values()]
    losses = []
    volume_losses = []
    support = [[] for _ in colors]
    for (observed, paint, volume_observed), ids in zip(observations, rendered, strict=True):
        predicted = np.zeros_like(observed)
        for i, color in enumerate(colors, 1):
            mask = ids == i
            predicted[..., color] |= mask
            eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            interior = eroded if eroded.sum() >= 3 else mask
            if interior.sum() >= 8:
                support[i - 1].append(float(paint[..., color][interior].mean()))
        losses.append(
            float(
                np.count_nonzero(observed ^ predicted)
                / max(np.count_nonzero(observed | predicted), 1)
            )
        )
        volume_rendered = ids < 0
        volume_losses.append(
            float(
                np.count_nonzero(volume_observed ^ volume_rendered)
                / max(np.count_nonzero(volume_observed | volume_rendered), 1)
            )
        )
    return dict(
        loss=float(np.mean(losses) + 0.1 * np.mean(volume_losses)),
        eye_losses=losses,
        volume_loss=float(np.mean(volume_losses)),
        minimum_hold_support=min([min(s) for s in support if s] or [0]),
        hold_support=support,
    )
