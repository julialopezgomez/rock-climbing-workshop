"""Resolve partial cup observations with an RGB-estimated configured robot CAD.

The renderer contains public robot, cup, wall and floor geometry. It receives
only a measured joint configuration and estimated trunk/camera transforms.
Candidate sockets come from visible RGB components and the public bolt lattice.
"""

import cv2
import mujoco
import numpy as np

from ._paths import model_path, simulator_asset
from .robot_pose.frame_conventions import project, to_wxyz
from .robot_pose.model_adapter import JOINTS, RobotModel
from .src.cad_silhouette import R_WALL_OBJECT


class CupSceneRenderer(RobotModel):
    def __init__(self, side=64, scale=4):
        self.side = side
        self.scale = scale
        size = side * scale
        spec = mujoco.MjSpec.from_file(str(model_path()))
        spec.add_mesh(
            name="candidate_cup_mesh", file=str(simulator_asset("assets/hold/hold_visual.stl"))
        )
        cup = spec.worldbody.add_body(name="candidate_cup", mocap=True)
        cup.add_geom(
            name="candidate_cup_visual",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname="candidate_cup_mesh",
            group=1,
            contype=0,
            conaffinity=0,
        )
        wall = spec.worldbody.add_body(name="known_wall", mocap=True)
        wall.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.4, 0.7, 0.0054],
            group=1,
            contype=0,
            conaffinity=0,
        )
        floor = spec.worldbody.add_body(name="known_floor", mocap=True)
        floor.add_geom(
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[3.0, 3.0, 0.05],
            group=1,
            contype=0,
            conaffinity=0,
        )
        spec.worldbody.add_camera(
            name="cup_camera",
            resolution=[size, size],
            sensor_size=[float(size)] * 2,
            focal_length=[900.0, 900.0],
        )
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.qadr = np.array([self.model.jnt_qposadr[self.model.joint(name).id] for name in JOINTS])
        self.root = self.model.body("trunk").id
        self.cid = self.model.camera("cup_camera").id
        self.cup_gid = self.model.geom("candidate_cup_visual").id
        self.mocap = {
            name: int(self.model.body_mocapid[self.model.body(name).id])
            for name in ["candidate_cup", "known_wall", "known_floor"]
        }
        bodies = {self.root}
        for body in range(self.root + 1, self.model.nbody):
            if self.model.body_parentid[body] in bodies:
                bodies.add(body)
        self.robot_geoms = np.array([b in bodies for b in self.model.geom_bodyid])
        self.model.vis.quality.offsamples = 0
        self.model.vis.global_.offwidth = self.model.vis.global_.offheight = size
        self.model.vis.map.znear = 0.00001
        self.renderer = mujoco.Renderer(self.model, height=size, width=size)
        self.option = mujoco.MjvOption()
        self.option.geomgroup[:] = [1, 1, 1, 0, 0, 0]
        self.root_pose = None

    def configure(self, joints, root_pose):
        self.set_joints(joints)
        self.root_pose = np.asarray(root_pose)
        rotation = self.root_pose[:3, :3].T
        translation = self.root_pose[:3, 3]
        self.data.mocap_pos[self.mocap["known_wall"]] = rotation @ (
            np.array([0.4, 0.7, -0.005]) - translation
        )
        self.data.mocap_quat[self.mocap["known_wall"]] = to_wxyz(rotation)
        self.data.mocap_pos[self.mocap["known_floor"]] = rotation @ (-translation)
        self.data.mocap_quat[self.mocap["known_floor"]] = to_wxyz(rotation @ R_WALL_OBJECT)
        self.data.mocap_quat[self.mocap["candidate_cup"]] = to_wxyz(rotation @ R_WALL_OBJECT)

    def render(self, point, camera, origin):
        pose = self.root_pose
        rotation = pose[:3, :3].T
        self.data.mocap_pos[self.mocap["candidate_cup"]] = rotation @ (point - pose[:3, 3])
        tc = np.asarray(camera["T_camera_wall"]) @ pose
        r, t = tc[:3, :3], tc[:3, 3]
        self.model.cam_pos[self.cid] = -r.T @ t
        self.model.cam_quat[self.cid] = to_wxyz(np.stack([r[0], -r[1], -r[2]], axis=1))
        k = np.asarray(camera["K"])
        self.model.cam_intrinsic[self.cid, :2] = [k[0, 0] * self.scale, k[1, 1] * self.scale]
        self.model.cam_intrinsic[self.cid, 2:] = [
            (self.side / 2 - (k[0, 2] + 0.5 - origin[0])) * self.scale,
            (self.side / 2 - (k[1, 2] + 0.5 - origin[1])) * self.scale,
        ]
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.cid, scene_option=self.option)
        self.renderer.enable_segmentation_rendering()
        seg = self.renderer.render()
        self.renderer.disable_segmentation_rendering()
        geom = seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
        cup = ((seg[..., 0] == self.cup_gid) & geom).astype(np.float32)
        ids = seg[..., 0]
        valid = (ids >= 0) & geom
        robot = np.zeros(ids.shape, np.float32)
        robot[valid] = self.robot_geoms[ids[valid]]

        def shrink(mask):
            return cv2.resize(mask, (self.side, self.side), interpolation=cv2.INTER_AREA) > 0.5

        return shrink(cup), shrink(robot)

    def close(self):
        self.renderer.close()


def copy_mask(pixels, origin, side):
    output = np.zeros((side, side), bool)
    x, y = pixels["origin"]
    mask = pixels["mask"]
    a, b = max(x, origin[0]), max(y, origin[1])
    c, d = min(x + mask.shape[1], origin[0] + side), min(y + mask.shape[0], origin[1] + side)
    if c > a and d > b:
        output[b - origin[1] : d - origin[1], a - origin[0] : c - origin[0]] = mask[
            b - y : d - y, a - x : c - x
        ]
    return output


def slot(point):
    return int(round((point[0] - 0.0325) / 0.015)), int(round((point[1] + 0.004) / 0.015))


def socket(col, row):
    return np.array([0.0325 + 0.015 * col, -0.004 + 0.015 * row, 0.026])


def overlaps(a, b):
    dc, dr = abs(a[0] - b[0]), abs(a[1] - b[1])
    return dc <= 2 and dr <= 2 and (dc, dr) != (2, 2)


def refine_candidates(candidates, images, cameras, robot_pose, renderer):
    renderer.configure(robot_pose["joints_rad"], robot_pose["T_wall_trunk"])
    local = []
    for gid in np.flatnonzero(renderer.robot_geoms):
        if renderer.model.geom_group[gid] not in (0, 1, 2):
            continue
        mid = renderer.model.geom_dataid[gid]
        if mid < 0:
            continue
        start = renderer.model.mesh_vertadr[mid]
        count = renderer.model.mesh_vertnum[mid]
        vertices = renderer.model.mesh_vert[start : start + count : max(1, count // 20)]
        local.append(
            vertices @ renderer.data.geom_xmat[gid].reshape(3, 3).T + renderer.data.geom_xpos[gid]
        )
    local = np.concatenate(local)
    pose = np.asarray(robot_pose["T_wall_trunk"])
    world = local @ pose[:3, :3].T + pose[:3, 3]
    boxes = []
    for camera in cameras:
        pixels = project(world, camera)
        boxes.append(np.r_[pixels.min(0) - 5, pixels.max(0) + 5])
    confirmed = []
    uncertain = []
    rejected = []
    for value in candidates:
        near = False
        for eye, p in value.get("pixels", {}).items():
            x, y = p["origin"]
            h, w = p["mask"].shape
            b = boxes[eye]
            near |= x <= b[2] and x + w >= b[0] and y <= b[3] and y + h >= b[1]
        strong = len(value["observed_by"]) >= 2 or any(q["cad_accepted"] for q in value["quality"])
        (confirmed if strong and not near else uncertain).append(value)
    uncertain.sort(
        key=lambda v: (-sum(q["cad_accepted"] for q in v["quality"]), -len(v["observed_by"]))
    )
    occupied = {slot(v["point"]) for v in confirmed}
    labels = []
    for image in images:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        paint = (hsv[..., 1] >= 140) & (hsv[..., 2] >= 35)
        labels.append(cv2.connectedComponents(paint.astype(np.uint8))[1])
    ids = [c.get("camera_id", str(i)) for i, c in enumerate(cameras)]
    for value in uncertain:
        col, row = slot(value["point"])
        ranked = []
        reliable = (
            any(q["cad_accepted"] for q in value["quality"])
            and np.linalg.norm(socket(col, row) - value["point"]) < 0.006
        )
        radius = 0 if reliable else 2
        for rr in range(max(0, row - radius), min(92, row + radius) + 1):
            for cc in range(max(0, col - radius), min(49, col + radius) + 1):
                key = (cc, rr)
                if any(other != key and overlaps(key, other) for other in occupied):
                    continue
                point = socket(cc, rr)
                intersection = 0
                union = 0
                visible = 0
                observed = []
                for eye, camera in enumerate(cameras):
                    origin = np.floor(
                        project(np.array([point]), camera)[0] - renderer.side / 2
                    ).astype(int)
                    expected, robot = renderer.render(point, camera, origin)
                    area = int(expected.sum())
                    visible += area
                    if eye in value.get("pixels", {}):
                        actual = copy_mask(value["pixels"][eye], origin, renderer.side)
                    else:
                        label_crop = copy_label_crop(labels[eye], origin, renderer.side)
                        hit, counts = np.unique(label_crop[expected], return_counts=True)
                        valid = hit > 0
                        chosen = int(hit[valid][np.argmax(counts[valid])]) if valid.any() else -1
                        actual = label_crop == chosen
                    actual = actual & ~robot
                    intersection += int((expected & actual).sum())
                    union += int((expected | actual).sum())
                    if (expected & actual).sum() >= 5:
                        observed.append(eye)
                if visible >= 12:
                    ranked.append((intersection / max(union, 1), key, point, observed))
        ranked.sort(key=lambda r: -r[0])
        best = ranked[0] if ranked else None
        gap = best[0] - ranked[1][0] if len(ranked) > 1 else 1.0
        if best is not None and best[0] >= 0.55 and gap >= 0.025:
            score, key, point, seen = best
            occupied.add(key)
            confirmed.append(
                dict(
                    value,
                    point=point,
                    observed_by=[ids[i] for i in seen] or value["observed_by"],
                    quality=value["quality"]
                    + [
                        dict(
                            cad_accepted=True,
                            scene_cup_iou=score,
                            scene_score_gap=gap,
                            hypotheses=len(ranked),
                            robot_pose_source=robot_pose["method"],
                            occlusion_model="RGB-estimated configured robot, public wall/floor",
                        )
                    ],
                )
            )
        else:
            rejected.append(
                dict(
                    point_m=value["point"].tolist(),
                    reason="No uniquely supported cup/robot scene hypothesis",
                    best_scene_iou=best[0] if best else None,
                    scene_score_gap=gap,
                )
            )
    return confirmed, rejected


def copy_label_crop(image, origin, side):
    x, y = origin
    result = np.zeros((side, side), np.int32)
    a, b = max(0, x), max(0, y)
    c, d = min(image.shape[1], x + side), min(image.shape[0], y + side)
    if c > a and d > b:
        result[b - y : d - y, a - x : c - x] = image[b:d, a:c]
    return result
