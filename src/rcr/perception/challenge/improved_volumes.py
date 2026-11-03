"""Estimate textured pyramid supports from stereo color regions and geometry."""

import cv2
import numpy as np

from .src.hold_geometry import estimate_volumes, refine_volume_faces


def flatten_pair(images, clusters=4, maximum_red=200, contrast=4, closing=0, neutral=False):
    """Cluster both views together so corresponding faces keep the same color."""
    values = [cv2.medianBlur(image, 3).astype(np.int16) for image in images]
    masks = []
    for value in values:
        red, green, blue = np.moveaxis(value, -1, 0)
        if neutral:
            maximum = value.max(axis=-1)
            mask = (maximum < maximum_red) & (maximum - value.min(axis=-1) < 0.35 * maximum + 8)
        else:
            mask = (blue > red + contrast) & (blue > green + contrast) & (red < maximum_red)
        if closing:
            mask = (
                cv2.morphologyEx(
                    mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((closing, closing), np.uint8)
                )
                > 0
            )
        masks.append(mask)
    pixels = np.concatenate([value[mask] for value, mask in zip(values, masks, strict=True)])
    result = [np.full_like(image, 150) for image in images]
    if len(pixels) < 100:
        return result
    # This initializes color clustering, independently of any scene generator.
    cv2.setRNGSeed(0)
    _, labels, centers = cv2.kmeans(
        pixels.astype(np.float32),
        clusters,
        None,
        (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.1),
        1,
        cv2.KMEANS_PP_CENTERS,
    )
    offset = 0
    for image, mask in zip(result, masks, strict=True):
        count = int(mask.sum())
        image[mask] = np.rint(centers[labels[offset : offset + count, 0]]).astype(np.uint8)
        offset += count
    return result


def estimate_variant(
    images,
    calibration,
    geometry,
    *,
    clusters=4,
    maximum_red=200,
    contrast=4,
    closing=0,
    neutral=False,
):
    flattened = flatten_pair(images, clusters, maximum_red, contrast, closing, neutral)
    volumes = estimate_volumes(
        flattened, calibration, geometry, base_tolerance=0.03, vertex_loss_limit=9.0
    )
    return refine_volume_faces(flattened, calibration, geometry, volumes)


def estimate(images, calibration, geometry):
    """Return image-supported volume parameters without reading scene labels."""
    candidates = []
    for clusters, maximum_red in ((3, 220), (6, 220), (5, 170)):
        candidates.extend(
            estimate_variant(
                images,
                calibration,
                geometry,
                clusters=clusters,
                maximum_red=maximum_red,
                contrast=4 if maximum_red == 220 else 5,
                closing=3 if clusters == 6 else 0,
            )
        )
    candidates.extend(
        estimate_variant(
            images, calibration, geometry, clusters=6, maximum_red=190, closing=3, neutral=True
        )
    )
    result = []
    for candidate in sorted(candidates, key=lambda v: v.get("joint_face_loss", v["loss"])):
        if any(
            np.linalg.norm(candidate["parameters"][:2] - v["parameters"][:2]) < 0.04 for v in result
        ):
            continue
        result.append(candidate)
    return result
