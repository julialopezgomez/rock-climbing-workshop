"""Keep raw photometry separate from normalized paint observations."""

import json

import numpy as np
from PIL import Image

from rcr.perception.challenge import improved_frontend, improved_masks, improved_volumes


def test_raw_views_reach_photometry_and_normalized_views_reach_cad(tmp_path, monkeypatch):
    rgb = np.full((40, 120, 3), [210, 125, 115], np.uint8)
    image = tmp_path / "image.png"
    Image.fromarray(rgb).save(image)
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({"image_width": 60, "image_height": 40}))
    monkeypatch.setenv("RCR_IMPROVED_MASKS", "1")
    monkeypatch.setenv("RCR_IMPROVED_VOLUMES", "1")
    monkeypatch.setenv("RCR_IMPROVED_CLASSIFIER", "0")
    monkeypatch.setattr(improved_frontend, "appearance", lambda _: ("real", 0.0))
    monkeypatch.setattr(improved_masks, "canonical_rgb", lambda _: np.full_like(rgb, 150))
    monkeypatch.setattr(
        improved_frontend,
        "cpu_estimate",
        lambda *args: ({"holds": {}}, {"objects": [], "losses": []}),
    )
    monkeypatch.setattr(improved_volumes, "estimate", lambda *args: [])
    seen = {}

    def infer(images, calibration, **kwargs):
        seen["images"] = images
        seen.update(kwargs)
        return {}, {"objects": [], "volumes": [], "partial": False}

    monkeypatch.setattr(improved_frontend, "infer", infer)
    monkeypatch.setattr(improved_frontend, "recover_scene", lambda holds, *args: (holds, {}))
    improved_frontend.hold_prediction(image, calibration, tmp_path / "output")
    assert all(np.all(image == 150) for image in seen["images"])
    np.testing.assert_array_equal(seen["photo_images"][0], rgb[:, :60])
    np.testing.assert_array_equal(seen["photo_images"][1], rgb[:, 60:])
    assert seen["estimated_volumes"] == []
