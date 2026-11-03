"""Image-only hold fitting with independently evaluated observation modules."""

import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .cpu_frontend import estimate as cpu_estimate
from .photo_frontend import appearance, canonical_rgb, volume_rgb
from .src.hold_constraints import recover_scene
from .src.hold_geometry import SceneGeometry
from .src.hold_scene import infer
from .src.rcr_contract import TYPES, write_prediction


def settings():
    """Explicit experiment switches; no image identifier changes the policy."""
    return dict(
        masks=os.environ.get("RCR_IMPROVED_MASKS", "0") == "1",
        volumes=os.environ.get("RCR_IMPROVED_VOLUMES", "1") == "1",
        classifier=os.environ.get("RCR_IMPROVED_CLASSIFIER", "0") == "1",
        allow_cube=os.environ.get("RCR_ALLOW_CUBE", "1") == "1",
        finger_yaw_scale=float(os.environ.get("RCR_FINGER_YAW_SCALE", "1")),
        finger_yaw_policy=os.environ.get("RCR_FINGER_YAW_POLICY", "half"),
    )


def hold_prediction(stereo, calibration, out):
    started = time.perf_counter()
    budget = float(os.environ.get("RCR_IMAGE_FIT_SECONDS", "24"))
    deadline = started + max(0.1, budget - 4.0)
    options = settings()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    raw = np.asarray(Image.open(stereo).convert("RGB"))
    calibration = json.loads(Path(calibration).read_text())
    width = raw.shape[1] // 2
    if raw.shape[1] % 2 or (width, raw.shape[0]) != (
        calibration["image_width"],
        calibration["image_height"],
    ):
        raise ValueError("Image dimensions do not match the declared stereo calibration")
    style, fraction = appearance(raw)
    if options["masks"] and style == "real":
        from .improved_masks import canonical_rgb as improved_rgb

        normalized = improved_rgb(raw)
    else:
        normalized = canonical_rgb(raw) if style == "real" else raw
    baseline, cpu_debug = cpu_estimate(normalized, calibration)
    debug = dict(
        method="image-only CAD with calibrated observations",
        seed_search=False,
        source_exact=False,
        options=options,
        partial=True,
        volumes=[],
        appearance=dict(style=style, bright_paint_fraction=fraction),
        inputs=["stereo RGB", "declared calibration", "CAD", "development-trained models"],
    )
    write_prediction(out, baseline["holds"], calibration, 0, debug, volumes=[])
    images = [raw[:, :width], raw[:, width:]]
    volumes = None
    if options["volumes"]:
        from .improved_volumes import estimate

        volumes = estimate(images, calibration, SceneGeometry.from_calibration(calibration))
    exact, fitted = infer(
        [normalized[:, :width], normalized[:, width:]],
        calibration,
        estimated_volumes=volumes,
        geometry_images=[volume_rgb(im) for im in images],
        photo_images=images,
        deadline=deadline,
        fast=True,
        pair_losses=[-loss for loss in cpu_debug["losses"]],
        hold_types=TYPES if options["allow_cube"] else TYPES - {"cube_hold"},
    )
    merged = {
        (r["left_proposal"], r["right_proposal"]): (baseline["holds"][r["hold_id"]], r)
        for r in cpu_debug["objects"]
        if r["accepted"]
    }
    accepted = [r for r in fitted["objects"] if r["accepted"]]
    for hold, record in zip(exact.values(), accepted, strict=True):
        merged[(record["left_proposal"], record["right_proposal"])] = (hold, record)
    holds = {f"hold_{i + 1}": item[0] for i, item in enumerate(merged.values())}
    debug.update(fitted)
    debug["objects"] = [item[1] for item in merged.values()]
    debug["method"] = "image-only CAD with calibrated observations"
    debug["options"] = options
    debug["seed_search"] = False
    debug["source_exact"] = False
    debug["pre_classifier_holds"] = len(holds)
    if options["classifier"]:
        from .improved_pose import postprocess

        holds, debug = postprocess(
            holds,
            debug,
            raw,
            calibration,
            finger_yaw_policy=options["finger_yaw_policy"],
        )
    elif options["finger_yaw_scale"] != 1:
        for hold in holds.values():
            if hold["type"] == "fingerpocket_hold" and abs(hold["position"]["z"]) < 0.002:
                hold["orientation"]["yaw"] *= options["finger_yaw_scale"]
    volume_records = [
        dict(
            name=f"Volume_{i}",
            center=dict(x=v["parameters"][0], y=v["parameters"][1]),
            baseWidth=v["parameters"][2],
            baseHeight=v["base_height"],
            depth=v["parameters"][3],
        )
        for i, v in enumerate(debug["volumes"])
    ]
    debug["runtime_s"] = time.perf_counter() - started
    write_prediction(out, holds, calibration, 0, debug, volumes=volume_records)
    if time.perf_counter() < started + budget - 2:
        holds, debug["scene_recovery"] = recover_scene(holds, debug, calibration)
        debug["runtime_s"] = time.perf_counter() - started
        write_prediction(out, holds, calibration, 0, debug, volumes=volume_records)
    return holds, debug
