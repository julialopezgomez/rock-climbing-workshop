"""CPU inference must recover metric anchors without a rendering context."""

import itertools
import json

import numpy as np
import pytest
from PIL import Image

cv2 = pytest.importorskip("cv2")
pytest.importorskip("trimesh")

from rcr.perception.challenge import cpu_frontend  # noqa: E402
from rcr.perception.challenge.src.rcr_exact_fit import ExactRenderer  # noqa: E402


def cube_scene():
    vertices = np.array(list(itertools.product([-0.025, 0.025], [-0.025, 0.025], [0, 0.04])))
    point = np.array([0.12, 0.78, 0])
    k = np.array([[600, 0, 160], [0, 600, 120], [0, 0, 1]])
    calibration = dict(image_width=320, image_height=240, wall_width_m=3, wall_height_m=1.4)
    cameras, images = {}, []
    for name, x in [("left", 0.032), ("right", -0.032)]:
        transform = np.diag([1.0, -1.0, -1.0, 1.0])
        transform[:3, 3] = [x, 0.7, 1]
        camera_vertices = (vertices + point) @ transform[:3, :3].T + transform[:3, 3]
        pixels = camera_vertices @ k.T
        pixels = pixels[:, :2] / pixels[:, 2:] - 0.5
        hull = cv2.convexHull(np.rint(pixels).astype(np.int32))
        rgb = np.full((240, 320, 3), 150, np.uint8)
        cv2.fillConvexPoly(rgb, hull, (255, 100, 90))
        images.append(rgb)
        cameras[name] = dict(K=k.tolist(), T_camera_scene=transform.tolist())
    calibration["cameras"] = cameras
    return np.concatenate(images, axis=1), calibration, vertices, point


def test_cpu_wrapper_recovers_cube_without_gpu_context(tmp_path, monkeypatch):
    rgb, calibration, vertices, expected = cube_scene()
    monkeypatch.setattr(cpu_frontend, "shapes", lambda: {"cube_hold": vertices})

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU estimation opened the GPU renderer")

    monkeypatch.setattr(ExactRenderer, "__init__", forbidden)
    stereo = tmp_path / "stereo.png"
    Image.fromarray(rgb).save(stereo)
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps(calibration))
    output = tmp_path / "prediction"
    holds, debug = cpu_frontend.hold_prediction(stereo, calibration_path, output)

    assert len(holds) == 1
    hold = next(iter(holds.values()))
    assert hold["type"] == "cube_hold"
    actual = np.array([hold["position"][axis] for axis in "xyz"])
    assert np.linalg.norm(actual - expected) < 0.003
    assert debug["volumes"] == []
    native = json.loads((output / "native_calibration.json").read_text())
    assert native["native_partial_paint_masks"]
    assert native["native_ignore_volume_mask"]
    assert native["native_volume_counts"] == [0, 3]
    assert json.loads((output / "prediction.json").read_text())["holds"] == holds
    assert json.loads((output / "image_holds.json").read_text()) == holds


def test_blank_pair_reports_no_holds():
    rgb, calibration, _, _ = cube_scene()
    rgb[:] = 150
    prediction, debug = cpu_frontend.estimate(rgb, calibration)
    assert prediction["holds"] == {}
    assert debug["proposals"] == 0
