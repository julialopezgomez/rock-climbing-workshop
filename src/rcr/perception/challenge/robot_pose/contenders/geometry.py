"""Official visual meshes and labelled kinematic landmarks, without rendering.

Mesh coordinates are compiled MuJoCo visual coordinates, including its mesh
preprocessing transform. Never export collision hulls as the visual robot.
"""

import hashlib
from pathlib import Path

import numpy as np
import trimesh

from ..frame_conventions import transform

JOINT_LANDMARKS = (
    "right_shoulder_pitch",
    "left_shoulder_pitch",
    "right_elbow",
    "left_elbow",
    "right_hip_yaw",
    "left_hip_yaw",
    "right_knee",
    "left_knee",
    "right_ankle_pitch",
    "left_ankle_pitch",
)
LANDMARK_NAMES = tuple(
    ["left_hand", "right_hand", "left_foot", "right_foot"]
    + [f"trunk_{i}" for i in range(6)]
    + list(JOINT_LANDMARKS)
)


def landmarks(model, joints):
    """Twenty consistently named points; hidden joint centres remain inferred."""
    points = model.landmarks(joints)
    for name in JOINT_LANDMARKS:
        jid = model.model.joint(name).id
        points[name] = model.data.xanchor[jid].copy()
    return np.array([points[name] for name in LANDMARK_NAMES])


def visual_parts(model, joints):
    """Return (body id, mesh in trunk coordinates) for each visual geom."""
    model.set_joints(joints)
    result = []
    for gid in range(model.model.ngeom):
        if model.model.geom_group[gid] not in (0, 1, 2):
            continue
        mid = int(model.model.geom_dataid[gid])
        if mid < 0 or int(model.model.geom_type[gid]) != 7:
            raise ValueError("Unexpected non-mesh visual geom: add explicit support")
        va = model.model.mesh_vertadr[mid]
        vn = model.model.mesh_vertnum[mid]
        fa = model.model.mesh_faceadr[mid]
        fn = model.model.mesh_facenum[mid]
        vertices = model.model.mesh_vert[va : va + vn].astype(float)
        faces = model.model.mesh_face[fa : fa + fn].copy()
        if faces.min() < 0 or faces.max() >= vn:
            raise ValueError("Compiled mesh face indices are outside their mesh")
        rotation = model.data.geom_xmat[gid].reshape(3, 3)
        vertices = vertices @ rotation.T + model.data.geom_xpos[gid]
        mesh = trimesh.Trimesh(vertices, faces, process=False)
        mesh.visual.vertex_colors = np.tile([170, 170, 170, 255], (len(vertices), 1))
        result.append((int(model.model.geom_bodyid[gid]), mesh))
    if not result:
        raise ValueError("No official visual geometry")
    return result


def configured_mesh(model, joints, *, trunk_only=False):
    parts = [
        mesh for body, mesh in visual_parts(model, joints) if not trunk_only or body == model.root
    ]
    return trimesh.util.concatenate(parts)


def export_links(model, joints, output):
    """Reusable per-link OBJ plus known-q transforms for M3T fixed links.

    Each mesh is centred in its own local frame. T_trunk_mesh updates with
    encoders; the mesh and its sparse viewpoint model can be cached unchanged.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    parts = visual_parts(model, joints)
    bodies = sorted({body for body, _ in parts})
    links = []
    for body in bodies:
        rotation = model.data.xmat[body].reshape(3, 3).copy()
        position = model.data.xpos[body].copy()
        mesh = trimesh.util.concatenate([m.copy() for b, m in parts if b == body])
        mesh.vertices = (mesh.vertices - position) @ rotation
        center = mesh.bounds.mean(0)
        mesh.vertices -= center
        pose = transform(rotation, position + rotation @ center)
        name = model.model.body(body).name
        path = output / (name + ".obj")
        mesh.export(path)
        links.append(
            dict(
                name=name,
                body_id=body,
                geometry=str(path.resolve()),
                T_trunk_mesh=pose.tolist(),
                center_body_m=center.tolist(),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                vertices=len(mesh.vertices),
                faces=len(mesh.faces),
            )
        )
    return links
