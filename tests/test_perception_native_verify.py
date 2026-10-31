"""Image checks must retain shaded paint and reject an absent visible hold."""

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("mujoco")
pytest.importorskip("trimesh")
from rcr.perception.challenge import native_verify  # noqa: E402


def test_local_colour_support_under_shading_and_highlights():
    image = np.full((60, 90, 3), 128, dtype=np.uint8)
    image[15:45, 5:25] = [105, 230, 149]  # Shaded green side face.
    image[15:45, 35:55] = [190, 255, 255]  # Blue with clipped illumination.
    image[15:45, 65:85] = [225, 130, 125]  # Red boundary blended with background.
    _, support, _ = native_verify.observation_masks([image])[0]
    assert support[30, 15, 1]
    assert support[30, 45, 2]
    assert support[30, 75, 0]
    assert not support[0, 0].any()


def test_visible_missing_hold_has_no_support(monkeypatch):
    image = np.full((60, 60, 3), 128, dtype=np.uint8)
    ids = np.zeros((60, 60), dtype=np.int16)
    ids[20:40, 20:40] = 1
    monkeypatch.setattr(native_verify, "render_ids", lambda *args: [ids, ids])
    scene = {"holds": {"a": {"color": "#F2554E"}}}
    loss = native_verify.image_loss(scene, [image, image], {})
    assert loss["minimum_hold_support"] == 0.0
    assert loss["eye_losses"] == [1.0, 1.0]
