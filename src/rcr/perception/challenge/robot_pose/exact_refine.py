"""Warm-start the preserved MuJoCo raster fitter and select by RGB evidence."""

import time

import numpy as np
from scipy.ndimage import gaussian_filter

from .configured_fit import estimate
from .differentiable_fit import observations


def image_losses(images, cameras, joints, renderer, poses):
    values = observations(images, cameras, renderer.side)
    renderer.set_joints(joints)
    losses = []
    for pose in poses:
        loss = 0.0
        for camera, value in zip(cameras, values, strict=False):
            rendered = gaussian_filter(renderer.render(pose, camera, value["origin"]), 1.0)
            loss += float(
                (((rendered - value["soft"]) ** 2) * value["valid"]).sum()
                / max(value["mask"].sum(), 1)
            )
        losses.append(loss)
    return losses


def refine_exact(images, cameras, joints, renderer, initial):
    started = time.perf_counter()
    pose, info = estimate(images, cameras, joints, renderer, initial=initial)
    losses = image_losses(images, cameras, joints, renderer, [initial, pose])
    selected = int(np.argmin(losses))
    return [initial, pose][selected], dict(
        info,
        runtime_s=time.perf_counter() - started,
        initial_loss=losses[0],
        refined_loss=losses[1],
        selected=selected,
        selection="Minimum common observed RGB silhouette loss; no pose labels",
    )
