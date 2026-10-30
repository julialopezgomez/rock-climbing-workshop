"""Apply calibrated intrinsics to the same color capture profile used to fit them."""

import numpy as np


def apply_intrinsics(packet, spec):
    if spec.get("pixel_convention") != "opencv_centres":
        raise ValueError("Calibration must use OpenCV pixel centres")
    if [c["camera_id"] for c in packet["cameras"]] != spec["camera_ids"]:
        raise ValueError("Calibration serial numbers/order do not match capture")
    if not np.isfinite(spec["stereo_rms_px"]) or spec["stereo_rms_px"] > 2.5:
        raise ValueError("Calibration RMS exceeds 2.5px; inspect/repeat calibration")
    for camera, calibrated, original in zip(
        packet["cameras"], spec["cameras"], spec["input_intrinsics"], strict=True
    ):
        if [camera["width"], camera["height"]] != spec["image_size"]:
            raise ValueError("Calibrated resolution does not match capture")
        if not np.allclose(camera["K"], original, atol=1e-5):
            raise ValueError("Source RGB intrinsics changed; calibrate this capture profile")
        camera["source_K"] = camera["K"]
        camera["K"] = calibrated["K"]
        camera["distortion"] = calibrated["distortion"]
        camera["distortion_model"] = "opencv_brown_conrady"
        camera["intrinsics_source"] = "Measured residual calibration on SDK-rectified color images"
    return packet
