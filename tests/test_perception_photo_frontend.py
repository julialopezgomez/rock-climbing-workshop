"""Controls for the matte-paint normalization used by competition inference."""

import json

import numpy as np
import pytest

pytest.importorskip("cv2")
from rcr.perception.challenge.photo_frontend import appearance, canonical_rgb, component_masks  # noqa: E402


def test_dim_paint_is_retained_without_bright_seed():
    rgb = np.full((80, 80, 3), 180, dtype=np.uint8)
    rgb[20:35, 20:35] = [30, 130, 65]
    rgb[45:60, 45:60] = [30, 60, 145]
    masks = component_masks(rgb)
    assert masks[1][25, 25]
    assert masks[2][50, 50]
    assert not any(mask[0, 0] for mask in masks)
    assert appearance(rgb)[0] == "real"


def test_red_reference_square_does_not_erase_local_hold():
    rgb = np.full((120, 120, 3), 180, dtype=np.uint8)
    rgb[15:110, 15:110] = [225, 125, 115]
    rgb[45:55, 55:65] = [210, 55, 50]
    red = component_masks(rgb)[0]
    assert red[50, 60]
    assert not red[80, 80]
    assert red.sum() < 0.1 * (95 * 95)


def test_canonical_pair_uses_declared_paint_colours():
    rgb = np.full((80, 160, 3), 180, dtype=np.uint8)
    rgb[20:35, 20:35] = [30, 130, 65]
    rgb[45:60, 125:140] = [30, 60, 145]
    result = canonical_rgb(rgb)
    assert result[25, 25].tolist() == [89, 199, 123]
    assert result[50, 130].tolist() == [99, 137, 237]
    assert result[0, 0].tolist() == [150, 150, 150]


def test_wrapper_exports_native_observations_and_flags(tmp_path, monkeypatch):
    from PIL import Image

    from rcr.perception.challenge import cli, photo_frontend

    rgb = np.full((80, 160, 3), 180, dtype=np.uint8)
    rgb[20:35, 20:35] = [30, 130, 65]
    stereo = tmp_path / "input.png"
    Image.fromarray(rgb).save(stereo)
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({"image_width": 80, "image_height": 80}))
    called = []

    def fit(stereo_path, calibration_path, out):
        called.append((stereo_path, calibration_path, out))
        return {}, {"runtime_s": 0.0}

    monkeypatch.setattr(cli, "hold_prediction", fit)
    output = tmp_path / "output"
    photo_frontend.hold_prediction(stereo, calibration, output)
    spec = json.loads((output / "native_calibration.json").read_text())
    assert spec["appearance"] == "real"
    assert spec["native_volume_counts"] == [0, 3]
    assert spec["native_ignore_volume_mask"]
    assert spec["native_partial_paint_masks"]
    assert called[0][0] == output / "native_stereo.png"
    assert called[0][1] == calibration
