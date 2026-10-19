"""ArUco markers on the wall corners [Rules §5.2].

DICT_4X4_50, 0.100 m code inside a 0.018 m white border, outer border corner on
the wall corner. Centres are tabulated in INTERFACES.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rcr.types import WALL_HEIGHT, WALL_WIDTH

DICT_NAME = "DICT_4X4_50"
CODE = 0.100
BORDER = 0.018
PLATE = CODE + 2 * BORDER  # 0.136 m

CORNERS = {1: "top-left", 2: "top-right", 3: "bottom-left", 4: "bottom-right"}


def _aruco():
    import cv2

    return cv2.aruco


def plate_centre(marker_id: int) -> tuple[float, float]:
    """(x, z) of the plate centre in the wall frame [Rules §5.2]."""
    half = PLATE / 2
    corner = CORNERS[marker_id]
    x = half if "left" in corner else WALL_WIDTH - half
    z = WALL_HEIGHT - half if "top" in corner else half
    return x, z


def marker_png(marker_id: int, out_dir: Path, px_per_mm: int = 4) -> Path:
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"aruco_{marker_id}.png"
    code_px = int(CODE * 1000) * px_per_mm
    border_px = int(BORDER * 1000) * px_per_mm
    dictionary = _aruco().getPredefinedDictionary(getattr(_aruco(), DICT_NAME))
    code = _aruco().generateImageMarker(dictionary, marker_id, code_px)
    plate = np.full((code_px + 2 * border_px, code_px + 2 * border_px), 255, dtype=np.uint8)
    plate[border_px : border_px + code_px, border_px : border_px + code_px] = code
    cv2.imwrite(str(path), plate)
    return path


def detect(image: np.ndarray) -> dict[int, np.ndarray]:
    """Detect markers in an RGB image. Returns {id: 4x2 corner pixels}."""
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    detector = _aruco().ArucoDetector(
        _aruco().getPredefinedDictionary(getattr(_aruco(), DICT_NAME)),
        _aruco().DetectorParameters(),
    )
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2) for i, c in zip(ids.flatten(), corners, strict=False)}
