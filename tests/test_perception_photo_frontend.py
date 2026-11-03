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


def test_image_only_keeps_unrefined_pairs_and_saves_baseline_first(tmp_path, monkeypatch):
    from PIL import Image

    from rcr.perception.challenge import cpu_frontend, photo_frontend
    from rcr.perception.challenge.src import hold_constraints, hold_scene

    def hold(x):
        return dict(
            type="microban_hold",
            color="#59C77B",
            position=dict(x=x, y=0.5, z=0.0),
            orientation=dict(roll=0.0, pitch=0.0, yaw=0.0),
        )

    def record(i, loss):
        return dict(
            left_proposal=i,
            right_proposal=i,
            hold_id=f"hold_{i + 1}",
            accepted=True,
            loss=loss,
            family="wall",
            kinds=["microban_hold"],
        )

    raw = np.full((40, 80, 3), 180, dtype=np.uint8)
    source = tmp_path / "source.png"
    Image.fromarray(raw).save(source)
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps(dict(image_width=40, image_height=40)))
    output = tmp_path / "out"
    baseline = {"hold_1": hold(-0.1), "hold_2": hold(0.2)}
    records = [record(0, 0.1), record(1, 0.4)]
    monkeypatch.setattr(
        cpu_frontend,
        "estimate",
        lambda *_: ({"holds": baseline}, {"objects": records, "losses": [0.1, 0.4]}),
    )

    def refine(images, rig, **kwargs):
        checkpoint = json.loads((output / "prediction.json").read_text())
        assert checkpoint["holds"] == baseline
        assert kwargs["pair_losses"] == [-0.1, -0.4]
        assert np.array_equal(kwargs["geometry_images"][0], raw[:, :40])
        return {"hold_1": hold(-0.12)}, dict(
            objects=[records[0]], volumes=[], partial=True, runtime_s=0.0
        )

    monkeypatch.setattr(hold_scene, "infer", refine)
    monkeypatch.setattr(photo_frontend, "volume_rgb", lambda image: image)
    monkeypatch.setattr(hold_constraints, "recover_scene", lambda holds, *_: (holds, {}))
    monkeypatch.setenv("RCR_IMAGE_FIT_SECONDS", "24")
    holds, diagnostics = photo_frontend.image_only_hold_prediction(source, calibration, output)
    assert holds == {"hold_1": hold(-0.12), "hold_2": hold(0.2)}
    assert diagnostics["cpu_retained_pairs"] == 1
    assert diagnostics["seed_search"] is False
    assert diagnostics["source_exact"] is False
    assert not (output / "native_stereo.png").exists()
    assert not (output / "native_calibration.json").exists()
