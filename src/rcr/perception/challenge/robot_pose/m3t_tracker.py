"""Persistent RGB M3T process with fixed measured joints and six free root DOFs.

Only public CAD is cached. RGB arrays cross the process boundary as raw BGR in
private shared-memory files; inference never opens evaluation data.
"""

import hashlib
import json
import os
import select
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from .._paths import CACHE
from .contenders.geometry import export_links
from .contenders.m3t_collect import normalize_float32_pose
from .frame_conventions import transform, validate_transform
from .model_adapter import RobotModel

DEFAULT_CACHE = CACHE / "m3t"
EXECUTABLE = CACHE / "bin/microban_m3t"


def model_signature(model):
    digest = hashlib.sha256()
    for field in [
        "mesh_vert",
        "mesh_face",
        "geom_pos",
        "geom_quat",
        "geom_dataid",
        "geom_group",
        "geom_bodyid",
        "body_pos",
        "body_quat",
        "jnt_pos",
        "jnt_axis",
    ]:
        digest.update(np.asarray(getattr(model.model, field)).tobytes())
    return digest.hexdigest()


def cached_geometry(model, cache):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    signature = model_signature(model)
    manifest = cache / "geometry.json"
    if manifest.exists():
        value = json.loads(manifest.read_text())
        if value["model_sha256"] == signature and all(
            Path(link["geometry"]).exists() for link in value["links"]
        ):
            return value["links"]
    links = export_links(model, np.zeros(19), cache / "links")
    value = dict(
        model_sha256=signature,
        links=links,
        provenance=(
            "Compiled public climbing CAD only; zero-joint local mes"
            "h export; no image or pose labels"
        ),
    )
    manifest.write_text(json.dumps(value, indent=2))
    return links


def frame_offsets(model, joints, links):
    model.set_joints(joints)
    result = []
    for link in links:
        body = link["body_id"]
        rotation = model.data.xmat[body].reshape(3, 3)
        position = model.data.xpos[body] + rotation @ np.asarray(link["center_body_m"])
        result.append(transform(rotation, position).tolist())
    return result


class M3TTracker:
    def __init__(
        self,
        *,
        model=None,
        cache=DEFAULT_CACHE,
        executable=EXECUTABLE,
        gpu_confirmed=False,
        startup_timeout_s=300,
        frame_timeout_s=20,
        correspondence_iterations=7,
        update_iterations=2,
    ):
        if not gpu_confirmed:
            raise ValueError(
                "M3T uses hardware GPU rendering; pass gpu_confirmed=Tru"
                "e when GPU resources are available"
            )
        self.model = model or RobotModel()
        self.cache = Path(cache)
        self.links = cached_geometry(self.model, self.cache)
        self.executable = Path(executable)
        if not self.executable.is_file():
            raise FileNotFoundError(
                "Build the private M3T adapter with scripts/perception/setup.py --robot"
            )
        self.temp = tempfile.TemporaryDirectory(prefix="microban-m3t-", dir="/dev/shm")
        self.directory = Path(self.temp.name)
        self.process = None
        self.stderr = None
        self.index = 0
        self.last_pose = None
        self.startup_timeout_s = startup_timeout_s
        self.frame_timeout_s = frame_timeout_s
        self.messages = []
        self.buffer = b""
        if not 1 <= correspondence_iterations <= 30 or not 1 <= update_iterations <= 8:
            raise ValueError("Invalid M3T iteration budget")
        self.correspondence_iterations = correspondence_iterations
        self.update_iterations = update_iterations

    def _frame(self, images, cameras, joints, initial):
        if len(images) != 2 or len(cameras) != 2:
            raise ValueError("M3T requires two calibrated RGB observations")
        eye_packets = []
        for eye, (image, camera) in enumerate(zip(images, cameras, strict=False)):
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("RGB must be an HxWx3 uint8 array")
            if np.any(camera.get("distortion", [0] * 5)):
                raise ValueError("Undistort images before M3T and update the intrinsics")
            path = self.directory / f"eye_{eye}.bgr"
            np.ascontiguousarray(image[..., ::-1]).tofile(path)
            eye_packets.append(
                dict(
                    rgb_bgr_raw=str(path),
                    width=image.shape[1],
                    height=image.shape[0],
                    K=np.asarray(camera["K"]).tolist(),
                    T_camera_wall=validate_transform(camera["T_camera_wall"]).tolist(),
                )
            )
        return dict(
            id=f"{self.index:08d}",
            cameras=eye_packets,
            T_wall_trunk_initial=validate_transform(initial).tolist(),
            T_trunk_meshes=frame_offsets(self.model, joints, self.links),
        )

    def _read(self, expected, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if b"\n" not in self.buffer:
                if self.process.poll() is not None:
                    raise RuntimeError(
                        "M3T process exited: " + (self.directory / "stderr.log").read_text()[-4000:]
                    )
                ready, _, _ = select.select(
                    [self.process.stdout], [], [], max(0, deadline - time.monotonic())
                )
                if not ready:
                    break
                self.buffer += os.read(self.process.stdout.fileno(), 65536)
                continue
            line, self.buffer = self.buffer.split(b"\n", 1)
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                self.messages.append(line.decode(errors="replace").rstrip())
                self.messages = self.messages[-20:]
                continue
            if value.get("case") != expected:
                raise RuntimeError("M3T stream returned an unexpected frame identifier")
            return value
        raise TimeoutError(f"M3T did not return frame {expected} within {timeout}s")

    def update(self, images, cameras, joints, *, initial=None, reset=False):
        started = time.perf_counter()
        if initial is None:
            if self.last_pose is None:
                raise ValueError("The first M3T frame needs an autonomous initial pose")
            initial = self.last_pose
        frame = self._frame(images, cameras, joints, initial)
        if self.process is None:
            config = dict(
                links=self.links,
                frames=[frame],
                independent=0,
                correspondence_iterations=self.correspondence_iterations,
                update_iterations=self.update_iterations,
                model_cache=str((self.cache / "region_models").resolve()),
            )
            config_path = self.directory / "config.json"
            config_path.write_text(json.dumps(config))
            self.stderr = (self.directory / "stderr.log").open("w")
            self.process = subprocess.Popen(
                [
                    str(self.executable),
                    str(config_path),
                    str(self.directory / "raw.jsonl"),
                    "--gpu-confirmed",
                    "--stream",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.stderr,
                bufsize=0,
            )
            value = self._read(frame["id"], self.startup_timeout_s)
        else:
            self.process.stdin.write(
                (json.dumps(dict(frame=frame, reset=int(reset))) + "\n").encode()
            )
            self.process.stdin.flush()
            value = self._read(frame["id"], self.frame_timeout_s)
        pose, roundoff = normalize_float32_pose(value["T_wall_trunk"])
        self.last_pose = pose
        self.index += 1
        return pose, dict(
            m3t_update_seconds=value["seconds"],
            client_seconds=time.perf_counter() - started,
            float32_rotation_orthogonality_error=roundoff,
            persistent_process=True,
            raw_rgb_transport=True,
            measured_links_fixed=True,
            reset=bool(reset or self.index == 1),
            frame_index=self.index - 1,
            correspondence_iterations=self.correspondence_iterations,
            update_iterations=self.update_iterations,
            free_parameters="six global trunk parameters",
        )

    def close(self):
        if self.process is not None:
            try:
                if self.process.poll() is None:
                    self.process.stdin.write(b'{"stop":1}\n')
                    self.process.stdin.flush()
                    self.process.wait(timeout=5)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            for stream in [self.process.stdin, self.process.stdout]:
                stream.close()
            self.process = None
        if self.stderr is not None:
            self.stderr.close()
            self.stderr = None
        self.temp.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
