"""Two RealSense COLOR-only capture with explicit clock and lens conventions.

No depth or infrared stream is enabled. Hardware capture is not validated until
two devices are attached. SDK global timestamps are matched, not claimed to be
hardware-synchronized exposures.
"""

import hashlib
import importlib.metadata
import json
import queue
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .robot_pose.model_adapter import JOINTS, ROOT


def intrinsic_record(intrinsics):
    return dict(
        width=intrinsics.width,
        height=intrinsics.height,
        fx=intrinsics.fx,
        fy=intrinsics.fy,
        ppx=intrinsics.ppx,
        ppy=intrinsics.ppy,
        model=str(intrinsics.model),
        coeffs=list(intrinsics.coeffs),
    )


def projection_maps(rs, intrinsics, cache=None):
    """Use the installed SDK's own color projection for every distortion model.

    Inverse/modified Brown coefficients are not silently re-labelled OpenCV
    Brown coefficients. The resulting RGB is an undistorted pinhole image.
    """
    record = intrinsic_record(intrinsics)
    signature = json.dumps(
        dict(intrinsics=record, sdk=importlib.metadata.version("pyrealsense2")), sort_keys=True
    )
    key = hashlib.sha256(signature.encode()).hexdigest()
    path = Path(cache) / (key + ".npz") if cache else None
    if path and path.exists():
        with np.load(path) as data:
            return data["map_x"].copy(), data["map_y"].copy()
    width, height = intrinsics.width, intrinsics.height
    if intrinsics.model == rs.distortion.none:
        yy, xx = np.mgrid[:height, :width]
        mx, my = xx.astype(np.float32), yy.astype(np.float32)
    else:
        allowed = {
            rs.distortion.brown_conrady,
            rs.distortion.inverse_brown_conrady,
            rs.distortion.modified_brown_conrady,
            rs.distortion.ftheta,
            rs.distortion.kannala_brandt4,
        }
        if intrinsics.model not in allowed:
            raise ValueError("Unsupported RealSense color distortion model")
        mapping = np.empty((height, width, 2), np.float32)
        for y in range(height):
            mapping[y] = [
                rs.rs2_project_point_to_pixel(
                    intrinsics,
                    [
                        (x - intrinsics.ppx) / intrinsics.fx,
                        (y - intrinsics.ppy) / intrinsics.fy,
                        1.0,
                    ],
                )
                for x in range(width)
            ]
        if not np.isfinite(mapping).all():
            raise ValueError("SDK color projection produced invalid lens coordinates")
        mx, my = mapping[..., 0].copy(), mapping[..., 1].copy()
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, map_x=mx, map_y=my)
        path.with_suffix(".json").write_text(signature)
    return mx, my


def matching_telemetry(path, timestamp):
    """Interpolate a bracketing host-clock encoder log; never invent joint data."""
    with Path(path).open("rb") as stream:
        stream.seek(0, 2)
        size = stream.tell()
        offset = max(0, size - 256 * 1024)
        stream.seek(offset)
        lines = stream.read().decode().splitlines()
    if offset:
        lines = lines[1:]
    rows = []
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                continue
            raise ValueError("Malformed encoder log record") from None
        if row.get("clock") != "host_monotonic":
            raise ValueError("Encoder log must use the same host_monotonic clock")
        if set(row.get("joints_rad", {})) != set(JOINTS):
            raise ValueError("Encoder log must provide19 named joint angles")
        t = float(row["timestamp_s"])
        q = np.array([row["joints_rad"][name] for name in JOINTS], float)
        if not np.isfinite(t) or not np.isfinite(q).all():
            raise ValueError("Nonfinite encoder log sample")
        rows.append((t, q))
    rows.sort(key=lambda x: x[0])
    before = [r for r in rows if r[0] <= timestamp]
    after = [r for r in rows if r[0] >= timestamp]
    if before and after and after[0][0] - before[-1][0] <= 0.05:
        a, b = before[-1], after[0]
        if abs(a[0] - timestamp) > 0.03 or abs(b[0] - timestamp) > 0.03:
            raise ValueError("Encoder interpolation is too far from the RGB timestamp")
        alpha = 0.0 if b[0] == a[0] else (timestamp - a[0]) / (b[0] - a[0])
        q = a[1] * (1 - alpha) + b[1] * alpha
        return dict(
            joints_rad=dict(zip(JOINTS, q.tolist(), strict=False)),
            joint_timestamp_s=timestamp,
            joint_provenance=dict(
                method="linear interpolation of measured encoder samples",
                bracket_timestamps_s=[a[0], b[0]],
            ),
        )
    if not rows:
        raise ValueError("No encoder log samples")
    t, q = min(rows, key=lambda r: abs(r[0] - timestamp))
    if abs(t - timestamp) > 0.015:
        raise ValueError("No encoder sample within15ms of RGB")
    return dict(
        joints_rad=dict(zip(JOINTS, q.tolist(), strict=False)),
        joint_timestamp_s=t,
        joint_provenance=dict(method="nearest measured encoder sample", timestamp_s=t),
    )


class DualColorCapture:
    def __init__(self, serials=None, width=1280, height=720, fps=30, max_skew_s=0.015):
        if not 0 < max_skew_s <= 0.015:
            raise ValueError("The perception contract allows at most15ms stereo skew")
        import pyrealsense2 as rs

        self.rs = rs
        self.context = rs.context()
        devices = list(self.context.query_devices())
        available = {d.get_info(rs.camera_info.serial_number): d for d in devices}
        self.serials = serials or sorted(available)
        if (
            len(self.serials) != 2
            or len(set(self.serials)) != 2
            or any(s not in available for s in self.serials)
        ):
            raise ValueError(
                "Connect exactly two visible RGB devices or select two --serial values; visible: "
                + ", ".join(sorted(available))
            )
        self.devices = [available[s] for s in self.serials]
        self.width = width
        self.height = height
        self.fps = fps
        self.max_skew_s = max_skew_s
        self.pipelines = []
        self.maps = []
        self.intrinsics = []
        self.restore = []
        self.stop_event = threading.Event()
        self.queues = [queue.Queue(maxsize=5) for _ in range(2)]
        self.errors = queue.Queue()
        self.threads = []
        self.buffers = [deque(maxlen=6), deque(maxlen=6)]
        self.epoch_offset = time.time() - time.monotonic()

    def start(self):
        rs = self.rs
        try:
            for serial, device in zip(self.serials, self.devices, strict=False):
                for sensor in device.query_sensors():
                    if any(
                        p.stream_type() == rs.stream.color for p in sensor.get_stream_profiles()
                    ) and sensor.supports(rs.option.global_time_enabled):
                        old = sensor.get_option(rs.option.global_time_enabled)
                        self.restore.append((sensor, old))
                        sensor.set_option(rs.option.global_time_enabled, 1.0)
                pipeline = rs.pipeline(self.context)
                config = rs.config()
                config.enable_device(serial)
                config.enable_stream(
                    rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps
                )
                profile = pipeline.start(config)
                self.pipelines.append(pipeline)
                if any(p.stream_type() != rs.stream.color for p in profile.get_streams()):
                    raise RuntimeError("Refusing an unexpected non-RGB stream")
                intrinsics = (
                    profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
                )
                self.intrinsics.append(intrinsics)
                self.maps.append(projection_maps(rs, intrinsics, ROOT / "camera_maps"))
            for eye in range(2):
                worker = threading.Thread(target=self._worker, args=(eye,), daemon=True)
                worker.start()
                self.threads.append(worker)
            return self
        except Exception:
            self.close()
            raise

    def _worker(self, eye):
        rs = self.rs
        warm = 15
        while not self.stop_event.is_set():
            try:
                frame = self.pipelines[eye].wait_for_frames(1000).get_color_frame()
                if not frame:
                    continue
                if warm:
                    warm -= 1
                    continue
                domain = frame.get_frame_timestamp_domain()
                if domain not in {rs.timestamp_domain.global_time, rs.timestamp_domain.system_time}:
                    continue
                # A hardware_clock from each device is never treated as a
                # common clock merely because the enum names are identical.
                offset = time.time() - time.monotonic()
                if abs(offset - self.epoch_offset) > 0.005:
                    raise RuntimeError("Host system clock changed; restart timestamp mapping")
                stamp = frame.get_timestamp() / 1000 - self.epoch_offset
                value = dict(
                    timestamp_s=stamp,
                    rgb=np.asanyarray(frame.get_data()).copy(),
                    sdk_timestamp_ms=frame.get_timestamp(),
                    sdk_domain=str(domain),
                    frame_number=frame.get_frame_number(),
                    host_arrival_s=time.monotonic(),
                )
                try:
                    self.queues[eye].put_nowait(value)
                except queue.Full:
                    try:
                        self.queues[eye].get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self.queues[eye].put_nowait(value)
                    except queue.Full:
                        pass
            except RuntimeError as error:
                if not self.stop_event.is_set():
                    self.errors.put(str(error))
                break

    def pair(self, timeout_s=15, latest=True, max_age_s=0.15):
        until = time.monotonic() + timeout_s
        while time.monotonic() < until:
            if not self.errors.empty():
                raise RuntimeError(self.errors.get())
            for eye in range(2):
                while True:
                    try:
                        self.buffers[eye].append(self.queues[eye].get_nowait())
                    except queue.Empty:
                        break
            if all(self.buffers):
                pairs = [
                    (abs(a["timestamp_s"] - b["timestamp_s"]), i, j)
                    for i, a in enumerate(self.buffers[0])
                    for j, b in enumerate(self.buffers[1])
                ]
                if latest:
                    now = time.monotonic()
                    pairs = [
                        (skew, i, j)
                        for skew, i, j in pairs
                        if skew <= self.max_skew_s
                        and now
                        - min(self.buffers[0][i]["timestamp_s"], self.buffers[1][j]["timestamp_s"])
                        <= max_age_s
                    ]
                    if not pairs:
                        time.sleep(0.002)
                        continue
                    skew, i, j = min(
                        pairs,
                        key=lambda p: (
                            -min(
                                self.buffers[0][p[1]]["timestamp_s"],
                                self.buffers[1][p[2]]["timestamp_s"],
                            ),
                            p[0],
                        ),
                    )
                else:
                    skew, i, j = min(pairs)
                if skew <= self.max_skew_s:
                    values = []
                    for eye, index in enumerate([i, j]):
                        value = self.buffers[eye][index]
                        for _ in range(index + 1):
                            self.buffers[eye].popleft()
                        value["rgb"] = cv2.remap(value["rgb"], *self.maps[eye], cv2.INTER_LINEAR)
                        values.append(value)
                    return values
            time.sleep(0.002)
        raise TimeoutError(
            "No RGB pair with common-clock timestamps within15ms; ha"
            "rdware synchronization remains unvalidated"
        )

    def close(self):
        self.stop_event.set()
        for pipeline in self.pipelines:
            try:
                pipeline.stop()
            except RuntimeError:
                pass
        for thread in self.threads:
            thread.join(timeout=2)
        for sensor, old in self.restore:
            try:
                sensor.set_option(self.rs.option.global_time_enabled, old)
            except RuntimeError:
                pass
        self.pipelines = []
        self.restore = []

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.close()


def main(args):
    import pyrealsense2 as rs

    if args.list:
        print(
            json.dumps(
                dict(
                    sdk_version=importlib.metadata.version("pyrealsense2"),
                    devices=[
                        dict(
                            serial=d.get_info(rs.camera_info.serial_number),
                            name=d.get_info(rs.camera_info.name),
                        )
                        for d in rs.context().query_devices()
                    ],
                ),
                indent=2,
            )
        )
        return
    if args.out is None:
        raise ValueError("Capture requires --out DIRECTORY")
    cases = []
    with DualColorCapture(
        args.serial, args.width, args.height, args.fps, args.max_skew_ms / 1000
    ) as capture:
        for index in range(args.count):
            frames = capture.pair(args.timeout)
            stamp = float(np.mean([f["timestamp_s"] for f in frames]))
            packet = dict(
                schema="robot-input-v1",
                clock="host_monotonic",
                timestamp_s=stamp,
                cameras=[],
                capture_provenance=(
                    "Two color streams only; SDK timestamp matching, no hard"
                    "ware exposure-lock claim"
                ),
            )
            if args.telemetry_log:
                packet.update(matching_telemetry(args.telemetry_log, stamp))
            scene = args.out / "scenes" / f"{index:06d}"
            scene.mkdir(parents=True, exist_ok=True)
            for eye, (frame, intrinsics) in enumerate(
                zip(frames, capture.intrinsics, strict=False)
            ):
                Image.fromarray(frame["rgb"]).save(scene / f"rgb_{eye}.png")
                packet["cameras"].append(
                    dict(
                        camera_id=capture.serials[eye],
                        rgb=f"rgb_{eye}.png",
                        width=intrinsics.width,
                        height=intrinsics.height,
                        K=[
                            [intrinsics.fx, 0, intrinsics.ppx],
                            [0, intrinsics.fy, intrinsics.ppy],
                            [0, 0, 1],
                        ],
                        distortion=[0.0] * 5,
                        distortion_model="none",
                        pixel_convention="opencv_centres",
                        timestamp_s=frame["timestamp_s"],
                        clock_domain="host_monotonic",
                        sdk_timestamp_ms=frame["sdk_timestamp_ms"],
                        sdk_domain=frame["sdk_domain"],
                        frame_number=frame["frame_number"],
                        source_intrinsics=intrinsic_record(intrinsics),
                        undistortion="SDK color projection map; no depth input",
                    )
                )
            (scene / "input.json").write_text(json.dumps(packet, indent=2, allow_nan=False))
            cases.append(
                dict(id=f"{index:06d}", split="unlabelled_real_capture", timestamp_s=stamp)
            )
            (args.out / "manifest.json").write_text(
                json.dumps(
                    dict(
                        cases=cases,
                        scope=(
                            "Real RGB capture; no evaluation labels; physical accura"
                            "cy and exposure synchronization unvalidated"
                        ),
                    ),
                    indent=2,
                )
            )
            print(
                json.dumps(
                    dict(
                        input=str(scene / "input.json"),
                        skew_ms=abs(frames[0]["timestamp_s"] - frames[1]["timestamp_s"]) * 1000,
                    )
                ),
                flush=True,
            )
