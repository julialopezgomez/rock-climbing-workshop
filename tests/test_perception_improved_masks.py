"""Controls for the learned local-colour paint frontend."""

import numpy as np
import pytest

pytest.importorskip("cv2")

from rcr.perception.challenge.improved_masks import MODEL, component_masks, pixel_features


def test_features_do_not_include_pixel_coordinates():
    image = np.full((90, 100, 3), [210, 120, 110], np.uint8)
    features = pixel_features(image)
    assert features.shape == (90, 100, 15)
    np.testing.assert_array_equal(features[30, 30], features[60, 70])


@pytest.mark.skipif(not MODEL.exists(), reason="Optional paint model is not installed")
def test_hold_is_retained_and_square_interior_is_suppressed():
    image = np.full((120, 120, 3), 180, np.uint8)
    image[15:110, 15:110] = [225, 125, 115]
    image[45:57, 55:69] = [210, 55, 50]
    masks = component_masks(image)
    assert masks[0][50, 60]
    assert not masks[0][80, 80]
    assert masks[0].sum() < 0.1 * (95 * 95)
