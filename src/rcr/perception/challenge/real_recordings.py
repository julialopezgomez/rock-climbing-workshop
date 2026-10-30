"""Decode RGB-only RealSense bag files without assuming simultaneous capture."""

import hashlib
import json
import shutil
import tarfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

NAMES = ("l_without_rob", "r_without_rob", "l_with_rob", "r_with_rob")


def source_directory(source, output):
    source = Path(source)
    if source.is_dir():
        return source
    target = Path(output) / "source"
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(source) as archive:
        for name in NAMES:
            matches = [m for m in archive.getmembers() if Path(m.name).name == name + ".bag"]
            if len(matches) != 1 or not matches[0].isfile():
                raise ValueError(f"Archive must contain one regular {name}.bag")
            path = target / (name + ".bag")
            with archive.extractfile(matches[0]) as incoming, path.open("wb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)
    return target


def decode_recording(path, folder):
    import pyrealsense2 as rs

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    metadata_path = folder / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("source_sha256") == digest and (folder / "rgb.png").exists():
            return metadata
    pipeline, config = rs.pipeline(), rs.config()
    rs.config.enable_device_from_file(config, str(Path(path).resolve()), repeat_playback=False)
    config.enable_stream(rs.stream.color)
    profile = pipeline.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)
    video = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = video.get_intrinsics()
    serial = profile.get_device().get_info(rs.camera_info.serial_number)
    images, records = [], []
    try:
        while True:
            try:
                frames = pipeline.wait_for_frames(2000)
            except RuntimeError:
                if playback.current_status() != rs.playback_status.stopped:
                    raise
                break
            frame = frames.get_color_frame()
            if not frame:
                continue
            image = np.asarray(frame.get_data()).copy()
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            images.append(image)
            records.append(
                dict(
                    frame_number=frame.get_frame_number(),
                    timestamp_ms=frame.get_timestamp(),
                    timestamp_domain=str(frame.get_frame_timestamp_domain()),
                    sharpness=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                )
            )
    finally:
        pipeline.stop()
    if not images:
        raise ValueError(f"No color frames in {path}")
    indices = range(len(images) // 5, max(1, len(images) - len(images) // 5))
    selected = max(indices, key=lambda index: records[index]["sharpness"])
    Image.fromarray(images[selected]).save(folder / "rgb.png")
    stability_frames = []
    for index in sorted({len(images) // 4, len(images) // 2, 3 * len(images) // 4}):
        filename = f"frame_{index:04d}.png"
        Image.fromarray(images[index]).save(folder / filename)
        stability_frames.append(dict(filename=filename, **records[index]))
    metadata = dict(
        source_sha256=digest,
        serial=serial,
        width=intrinsics.width,
        height=intrinsics.height,
        K=[
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1],
        ],
        distortion=list(intrinsics.coeffs),
        distortion_model=str(intrinsics.model),
        frame_count=len(images),
        selected_index=selected,
        selected=records[selected],
        records=records,
        stability_frames=stability_frames,
        streams_read=["RGB color only"],
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
