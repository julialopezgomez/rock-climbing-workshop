"""Four-type schema, geometry and scoring-boundary checks without a GPU."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

cv2 = pytest.importorskip("cv2")
pytest.importorskip("mujoco")
trimesh = pytest.importorskip("trimesh")
from rcr.perception.challenge import cli  # noqa: E402
from rcr.perception.challenge.src import rcr_methods  # noqa: E402
from rcr.perception.challenge.src.hold_geometry import SceneGeometry  # noqa: E402
from rcr.perception.challenge.src.hold_scene import proposals  # noqa: E402
from rcr.perception.challenge.src.rcr_contract import TYPES, score_holds  # noqa: E402


def records():
    return {
        f"item_{i}": dict(
            type=kind,
            color="#F2554E",
            position=dict(x=i * 0.2, y=0.5, z=0.0),
            orientation=dict(roll=0.0, pitch=0.0, yaw=0.0),
        )
        for i, kind in enumerate(sorted(TYPES))
    }


def test_all_four_types_keep_their_identity_in_matching():
    expected = records()
    result = score_holds(expected, expected)
    assert result["tp"] == 4
    changed = json.loads(json.dumps(expected))
    changed["item_0"]["type"] = changed["item_1"]["type"]
    assert score_holds(changed, expected)["tp"] == 3


def test_wall_width_changes_the_volume_height_and_reference_square():
    geometry = SceneGeometry.from_calibration(dict(wall_width_m=3.0, wall_height_m=1.4))
    assert geometry.volume_height == pytest.approx(0.3)
    vertices = geometry.vertices([0.2, 0.6, 0.7, 0.23])
    np.testing.assert_allclose(vertices[:, 1], [0.45, 0.45, 0.75, 0.6])
    assert vertices[3, 2] == 0.23
    with pytest.raises(ValueError, match="positive"):
        SceneGeometry.from_calibration(dict(wall_width_m=-3.0))


def test_mesh_frame_is_blender_x_z_minus_y_and_backplate_zero(tmp_path, monkeypatch):
    folder = tmp_path / "assets/rcr"
    folder.mkdir(parents=True)
    model = trimesh.creation.box(extents=[0.04, 0.05, 0.06])
    model.apply_translation([1.0, 2.0, 3.0])
    model.export(folder / "fingerpocket_hold.stl")
    monkeypatch.setattr(rcr_methods, "CACHE", tmp_path)
    normalized = rcr_methods.normalized_mesh("fingerpocket_hold")
    np.testing.assert_allclose(normalized.extents, [0.04, 0.06, 0.05], atol=3e-7)
    np.testing.assert_allclose(normalized.bounds.mean(0)[:2], [0.0, 0.0], atol=1e-8)
    assert normalized.bounds[0, 2] == 0
    with pytest.raises(KeyError):
        rcr_methods.normalized_mesh("unknown")


def test_nearby_same_color_holds_are_not_merged_by_the_old_clearance_prior():
    image = np.zeros((25, 30, 3), np.uint8)
    for center in [(8, 12), (18, 12)]:
        cv2.circle(image, center, 3, (242, 85, 78), thickness=-1)
    detected = proposals(image)
    assert len(detected) == 2


def test_reference_is_opened_after_prediction_and_failures_keep_the_denominator(
    tmp_path, monkeypatch
):
    folder = tmp_path / "inputs/case"
    folder.mkdir(parents=True)
    Image.fromarray(np.zeros((2, 4, 3), np.uint8)).save(folder / "stereo.png")
    (folder / "calibration.json").write_text(
        json.dumps(dict(image_width=2, image_height=2, wall_width_m=3.0, wall_height_m=1.4))
    )
    (folder / "expected_holds.json").write_text(json.dumps(records()))
    output = tmp_path / "outputs"
    original = Path.read_text
    inference_failed = False

    def read(path, *args, **kwargs):
        if path.name == "expected_holds.json":
            assert inference_failed or (output / "case/holds.json").exists()
        return original(path, *args, **kwargs)

    def fitted(images, calibration):
        assert all(image.shape == (2, 2, 3) for image in images)
        return records(), dict(runtime_s=0.0, volumes=[])

    from rcr.perception.challenge.src import hold_constraints, hold_scene

    monkeypatch.setattr(hold_scene, "infer", fitted)
    monkeypatch.setattr(
        hold_constraints, "recover_scene", lambda holds, diagnostics, calibration: (holds, {})
    )
    monkeypatch.setattr(Path, "read_text", read)
    task = (str(folder.parent), str(output), dict(id="case"), None, None)
    result = cli.evaluate_hold_case(task)
    assert result["completed"] is True and result["strict_poses"] == 4
    exported = json.loads((output / "case/prediction.json").read_text())
    assert exported["scene"]["wall"]["width"] == 3.0
    assert exported["scene"]["wall"]["originMarker"]["width"] == 1.0

    def failed(images, calibration):
        nonlocal inference_failed
        inference_failed = True
        raise RuntimeError("Render failure")

    monkeypatch.setattr(hold_scene, "infer", failed)
    result = cli.evaluate_hold_case(task)
    assert result["completed"] is False
    assert not (output / "case/holds.json").exists()
    assert result["gt_holds"] == result["fn"] == 4
    assert result["tp"] == result["strict_poses"] == 0


def test_hull_recovery_separates_conflicts_along_the_support_plane():
    from scipy.spatial import ConvexHull

    from rcr.perception.challenge.src.hold_constraints import (
        collision_axes,
        shape,
        tangent_correction,
    )

    points = trimesh.creation.box(extents=[0.04, 0.04, 0.04]).vertices
    first = shape(points, ConvexHull(points).simplices, 0.0008)
    shifted = points + [0.039, 0.0, 0.0]
    second = shape(shifted, ConvexHull(shifted).simplices, 0.0008)
    collision = collision_axes(first, second)
    assert collision is not None
    delta = tangent_correction(collision, np.array([0.0, 0.0, 1.0]))
    assert delta[2] == 0
    first["points"] = points + delta
    assert collision_axes(first, second) is None


def test_volume_recovery_aligns_backplate_and_preserves_source_clearance(monkeypatch):
    from scipy.spatial.transform import Rotation

    from rcr.perception.challenge.src import hold_constraints
    from rcr.perception.challenge.src.rcr_native_geometry import FACES

    mesh = trimesh.creation.box(extents=[0.02, 0.02, 0.02])
    mesh.apply_translation([0, 0, 0.01])
    monkeypatch.setattr(hold_constraints, "normalized_mesh", lambda kind: mesh)
    hold_constraints.local_hull.cache_clear()
    calibration = dict(wall_width_m=3.0, wall_height_m=1.4)
    parameters = [0.0, 0.7, 0.5, 0.2]
    vertices = SceneGeometry.from_calibration(calibration).vertices(parameters)
    a, b, c = vertices[list(FACES[0])]
    normal = np.cross(b - a, c - a)
    normal /= np.linalg.norm(normal)
    if normal[2] < 0:
        normal *= -1
    rotation = rcr_methods._rotation(normal, 0.0) @ Rotation.from_euler("x", 0.02).as_matrix()
    point = (a + b + c) / 3 + normal * 0.0001
    hold = dict(
        type="cube_hold",
        color="#F2554E",
        position=dict(zip("xyz", point, strict=True)),
        orientation=dict(
            zip(
                ["roll", "pitch", "yaw"],
                Rotation.from_matrix(rotation).as_euler("XYZ"),
                strict=True,
            )
        ),
    )
    diagnostics = dict(
        objects=[dict(accepted=True, family="volume_0_face_0_pose", loss=0.1)],
        volumes=[dict(parameters=parameters)],
    )
    try:
        recovered, debug = hold_constraints.recover_scene(
            {"hold_1": hold}, diagnostics, calibration
        )
        assert not debug["rejected"]
        result = recovered["hold_1"]
        point = np.array([result["position"][key] for key in "xyz"])
        angles = [result["orientation"][key] for key in ["roll", "pitch", "yaw"]]
        np.testing.assert_allclose(Rotation.from_euler("XYZ", angles).as_matrix()[:, 2], normal)
        assert (point - a) @ normal == pytest.approx(hold_constraints.CONTACT_CLEARANCE)
    finally:
        hold_constraints.local_hull.cache_clear()
