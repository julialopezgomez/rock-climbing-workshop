"""Stereo-to-JSON entrypoint with optional, explicit native-generator recovery."""

import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .rcr_contract import write_prediction
from .rcr_maxfit import close_renderer, infer
from .rcr_seed_recovery import BIN, recover_batch


def seed_scanner(backend, gpu_confirmed=False):
    if backend == "cpu":
        if not BIN.exists():
            subprocess.run(
                [
                    "g++",
                    "-O3",
                    "-std=c++17",
                    "-fopenmp",
                    str(Path(__file__).with_name("rcr_seedscan.cpp")),
                    "-o",
                    str(BIN),
                ],
                check=True,
            )
        return BIN
    if backend != "cuda":
        raise ValueError("Unknown native seed scanner")
    if not gpu_confirmed:
        raise ValueError("CUDA scanning requires --gpu-confirmed and available GPU resources")
    binary = BIN.with_name("rcr-seedscan-cuda")
    if not binary.exists():
        subprocess.run(
            [
                "/usr/local/cuda/bin/nvcc",
                "-O3",
                "--fmad=false",
                "-std=c++17",
                "-arch=sm_89",
                str(Path(__file__).with_name("rcr_seedscan.cu")),
                "-o",
                str(binary),
            ],
            check=True,
        )
    return binary


def main(a):
    started = time.perf_counter()
    rgb = np.array(Image.open(a.stereo).convert("RGB"))
    if rgb.shape[1] % 2:
        raise ValueError("Stereo image must contain two equal-width panels")
    cal = json.loads(a.calibration.read_text())
    w = rgb.shape[1] // 2
    images = [rgb[:, :w], rgb[:, w:]]
    if (w, rgb.shape[0]) != (cal["image_width"], cal["image_height"]):
        raise ValueError("RGB dimensions do not match calibration")
    try:
        holds, debug = infer(images, cal, deep_search=True)
    finally:
        close_renderer()
    write_prediction(a.out / "cad", holds, cal, 0, debug)
    recovery = None
    output_seed = 0
    if a.native_generator:
        scanner = seed_scanner(
            getattr(a, "seed_backend", "cpu"), getattr(a, "gpu_confirmed", False)
        )
        first_limit = min(a.seed_limit, 2**20)
        results, config = recover_batch(
            [holds],
            limit=first_limit,
            hold_count=a.hold_count,
            maximum_volumes=a.maximum_volumes,
            images=[images],
            calibrations=[cal],
            scanner=scanner,
            candidate_filter=getattr(a, "fast_verify", False),
        )
        attempts = [config]
        if not results[0][1]["accepted"] and first_limit < a.seed_limit:
            results, config = recover_batch(
                [holds],
                limit=a.seed_limit,
                hold_count=a.hold_count,
                maximum_volumes=a.maximum_volumes,
                images=[images],
                calibrations=[cal],
                scanner=scanner,
                candidate_filter=getattr(a, "fast_verify", False),
            )
            attempts.append(config)
        holds, recovery = results[0]
        recovery["config"] = config
        recovery["search_attempts"] = attempts
        if recovery["accepted"]:
            output_seed = recovery["recovered_seed"]
    debug = dict(
        method="native_generator_recovery"
        if recovery and recovery["accepted"]
        else "native_inverse_graphics",
        native_generator_assumption=a.native_generator,
        recovery=recovery,
        runtime_s=time.perf_counter() - started,
        input_sha256={
            str(p.name): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [a.stereo, a.calibration]
        },
        inputs=(
            "RGB, declared calibration and public source/CAD; no lab"
            "el, input seed, metadata or answer table"
        ),
    )
    write_prediction(a.out, holds, cal, output_seed, debug)
    print(
        json.dumps(
            dict(
                method=debug["method"],
                holds=len(holds),
                seconds=debug["runtime_s"],
                recovery_accepted=bool(recovery and recovery["accepted"]),
            )
        )
    )
