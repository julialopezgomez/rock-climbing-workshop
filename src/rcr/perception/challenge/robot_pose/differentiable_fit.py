"""Two-view differentiable fitting of the public, encoder-configured visual CAD.

This optional GPU contender refines an autonomous pose using RGB silhouettes.
It does not use simulator depth, camera truth, masks or pose labels.
"""

import time

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from .configured_fit import crop, segment_robot
from .contenders.geometry import configured_mesh
from .frame_conventions import validate_transform
from .model_adapter import RobotModel


def observations(images, cameras, side):
    values = []
    for image, camera in zip(images, cameras, strict=False):
        mask, box, region = segment_robot(image, camera, return_region=True)
        origin = np.floor((np.asarray(box[:2]) + box[2:]) / 2 - side / 2).astype(int)
        observed = crop(mask.astype(np.float32), origin, side)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        occluder = (hsv[..., 1] > 100) & (hsv[..., 0] < 30) & (hsv[..., 2] > 70)
        valid = (~crop(occluder, origin, side)) & (crop(region, origin, side) > 0)
        values.append(
            dict(origin=origin, mask=observed, soft=gaussian_filter(observed, 1.0), valid=valid)
        )
    return values


class DifferentiableCAD:
    def __init__(self, model=None, side=224, gpu_confirmed=False):
        if not gpu_confirmed:
            raise ValueError("Differentiable CAD fitting requires an available GPU")
        import nvdiffrast.torch as dr
        import torch

        self.torch = torch
        self.dr = dr
        self.context = dr.RasterizeCudaContext()
        self.model = model or RobotModel()
        self.side = side
        self.device = "cuda"
        self.dtype = torch.float32
        x = torch.arange(-4, 5, device="cuda", dtype=self.dtype)
        kernel = torch.exp(-x * x / 2)
        kernel /= kernel.sum()
        self.kernel_x = kernel.reshape(1, 1, 1, -1)
        self.kernel_y = kernel.reshape(1, 1, -1, 1)
        self.eye = torch.eye(3, device="cuda", dtype=self.dtype)

    def tensor(self, value, dtype=None):
        return self.torch.as_tensor(value, device=self.device, dtype=dtype or self.dtype)

    def configure(self, joints, cameras, values):
        mesh = configured_mesh(self.model, joints)
        self.vertices = self.tensor(mesh.vertices)
        self.faces = self.tensor(mesh.faces, self.torch.int32).contiguous()
        self.topology = self.dr.antialias_construct_topology_hash(self.faces)
        self.tc = self.tensor(np.stack([c["T_camera_wall"] for c in cameras]))
        projections = []
        for camera, value in zip(cameras, values, strict=False):
            k = np.asarray(camera["K"])
            ox, oy = value["origin"]
            n, f = 0.01, 10.0
            projections.append(
                [
                    [2 * k[0, 0] / self.side, 0, 2 * (k[0, 2] - ox + 0.5) / self.side - 1, 0],
                    [0, -2 * k[1, 1] / self.side, 1 - 2 * (k[1, 2] - oy + 0.5) / self.side, 0],
                    [0, 0, (f + n) / (f - n), -2 * f * n / (f - n)],
                    [0, 0, 1, 0],
                ]
            )
        self.projections = self.tensor(projections)

    def rotation(self, vector):
        t = self.torch
        x, y, z = vector.unbind()
        zero = x * 0
        skew = t.stack([zero, -z, y, z, zero, -x, -y, x, zero]).reshape(3, 3)
        angle = t.linalg.vector_norm(vector)
        a = t.sinc(angle / t.pi)
        b = 0.5 * t.sinc(angle / (2 * t.pi)) ** 2
        return self.eye + a * skew + b * (skew @ skew)

    def render(self, rotation, translation):
        t = self.torch
        wall = self.vertices @ rotation.T + translation
        camera = t.einsum("nv,bkv->bnk", wall, self.tc[:, :3, :3]) + self.tc[:, None, :3, 3]
        homogeneous = t.cat([camera, t.ones_like(camera[..., :1])], dim=-1)
        clip = t.einsum("bnv,bkv->bnk", homogeneous, self.projections).contiguous()
        raster, _ = self.dr.rasterize(
            self.context, clip, self.faces, resolution=[self.side, self.side]
        )
        mask = (raster[..., 3:] > 0).to(t.float32)
        mask = self.dr.antialias(mask, raster, clip, self.faces, topology_hash=self.topology)
        return mask.flip(1).permute(0, 3, 1, 2)

    def smooth(self, image):
        f = self.torch.nn.functional
        return f.conv2d(
            f.conv2d(image, self.kernel_x, padding=(0, 4)), self.kernel_y, padding=(4, 0)
        )

    def fit(self, images, cameras, joints, initial, iterations=35):
        started = time.perf_counter()
        t = self.torch
        initial = validate_transform(initial)
        values = observations(images, cameras, self.side)
        self.configure(joints, cameras, values)
        target = self.tensor(np.stack([v["soft"] for v in values]))[:, None]
        valid = self.tensor(np.stack([v["valid"] for v in values]))[:, None]
        denominator = self.tensor([max(v["mask"].sum(), 1) for v in values]).reshape(-1, 1, 1, 1)
        base_r = self.tensor(initial[:3, :3])
        base_t = self.tensor(initial[:3, 3])
        scale = self.tensor([0.03] * 3 + [0.3] * 3)
        delta = t.zeros(6, device="cuda", requires_grad=True)
        optimizer = t.optim.LBFGS(
            [delta],
            lr=1,
            max_iter=iterations,
            max_eval=iterations * 3,
            tolerance_grad=1e-7,
            tolerance_change=1e-10,
            line_search_fn="strong_wolfe",
        )
        best = [float("inf"), initial.copy()]
        evaluations = 0

        def decode():
            step = t.tanh(delta) * scale
            return self.rotation(step[3:]) @ base_r, base_t + step[:3]

        def objective(backward=True):
            nonlocal evaluations
            if backward:
                optimizer.zero_grad()
            rotation, translation = decode()
            mask = self.render(rotation, translation)
            loss = (((self.smooth(mask) - target) ** 2) * valid / denominator).sum()
            number = float(loss.detach())
            if not np.isfinite(number):
                raise ValueError("Non-finite differentiable silhouette loss")
            if number < best[0]:
                pose = np.eye(4)
                pose[:3, :3] = rotation.detach().cpu().numpy()
                pose[:3, 3] = translation.detach().cpu().numpy()
                best[:] = [number, pose]
            evaluations += 1
            if backward:
                loss.backward()
            return loss

        initial_loss = float(objective(False).detach())
        optimizer.step(objective)
        # Re-orthogonalize only float32 numerical roundoff, as for M3T.
        from .contenders.m3t_collect import normalize_float32_pose

        result, roundoff = normalize_float32_pose(best[1])
        return result, dict(
            loss=best[0],
            initial_loss=initial_loss,
            render_evaluations=evaluations,
            runtime_s=time.perf_counter() - started,
            optimizer="six-parameter L-BFGS; multi-view RGB silhouettes",
            float32_rotation_orthogonality_error=roundoff,
            mask_pixels=[int(v["mask"].sum()) for v in values],
        )
