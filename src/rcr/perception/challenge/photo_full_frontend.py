"""Broaden image-only CAD fitting to every supported hold type."""

from .photo_frontend import image_only_hold_prediction


def hold_prediction(stereo, calibration, out):
    """Use four CAD types without generator recovery or scene-seed search."""
    return image_only_hold_prediction(stereo, calibration, out, full_types=True)
