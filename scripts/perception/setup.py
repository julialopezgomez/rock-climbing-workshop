#!/usr/bin/env python3
"""Install an isolated, repository-backed perception runtime and pinned resources."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PACKAGE = REPO / "src/rcr/perception/challenge"
M3T_COMMIT = "f02106186a1e756450226b3b793e309506817c58"
MODEL_COMMIT = "8e42ef70673b5747dbdca3efaf60bc0f8ec7ea6d"
BENCHMARK_COMMIT = "b8c607e8dc2749ae2f512ce7265ef3018f32e9f5"


def run(command, **kwargs):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True, **kwargs)


def download(url, target, expected=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and (
        expected is None or hashlib.sha256(target.read_bytes()).hexdigest() == expected
    ):
        return
    temporary = target.with_suffix(target.suffix + ".partial")
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    if expected and hashlib.sha256(temporary.read_bytes()).hexdigest() != expected:
        temporary.unlink()
        raise ValueError(f"Download checksum mismatch: {url}")
    temporary.replace(target)


def clone(url, path, commit, lfs=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not (path / ".git").exists():
        run(["git", "clone", "--no-checkout", url, path])
    if lfs:
        run(["git", "-C", path, "lfs", "install", "--local"])
    run(["git", "-C", path, "checkout", "--detach", commit])
    if lfs:
        run(["git", "-C", path, "lfs", "pull"])


def native(cache, micromamba=None):
    prefix = cache / "native"
    cmake = prefix / "bin/cmake"
    if not cmake.exists():
        binary = Path(micromamba or shutil.which("micromamba") or cache / "tools/micromamba")
        if not binary.exists():
            archive = cache / "downloads/micromamba-2.10.0.tar.bz2"
            download("https://micro.mamba.pm/api/micromamba/linux-64/2.10.0", archive)
            binary.parent.mkdir(parents=True, exist_ok=True)
            with tarfile.open(archive) as tar:
                member = tar.extractfile("bin/micromamba")
                if member is None:
                    raise ValueError("Micromamba archive has no binary")
                with binary.open("wb") as output:
                    shutil.copyfileobj(member, output)
            binary.chmod(0o755)
        run(
            [
                binary,
                "create",
                "-y",
                "-p",
                prefix,
                "--file",
                REPO / "scripts/perception/m3t-linux-64.lock",
            ]
        )
    source = cache / "sources/3DObjectTracking"
    clone("https://github.com/DLR-RM/3DObjectTracking.git", source, M3T_COMMIT)
    build = cache / "build"
    env = dict(
        os.environ,
        LD_LIBRARY_PATH=str(prefix / "lib") + ":" + os.environ.get("LD_LIBRARY_PATH", ""),
    )
    run(
        [
            cmake,
            "-S",
            source / "M3T",
            "-B",
            build / "m3t",
            "-G",
            "Ninja",
            "-DUSE_AZURE_KINECT=OFF",
            "-DUSE_REALSENSE=OFF",
            "-DUSE_GTEST=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_PREFIX_PATH={prefix}",
            f"-DCMAKE_BUILD_RPATH={prefix / 'lib'}",
        ],
        env=env,
    )
    run([cmake, "--build", build / "m3t", "--target", "m3t", "-j", "4"], env=env)
    run(
        [
            cmake,
            "-S",
            PACKAGE / "native",
            "-B",
            build / "adapter",
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_PREFIX_PATH={prefix}",
            f"-DCMAKE_BUILD_RPATH={prefix / 'lib'}",
            f"-DM3T_ROOT={source / 'M3T'}",
            f"-DM3T_LIBRARY={build / 'm3t/src/libm3t.a'}",
        ],
        env=env,
    )
    run([cmake, "--build", build / "adapter", "-j", "2"], env=env)
    shutil.copy2(build / "adapter/microban_m3t", cache / "bin/microban_m3t")


def benchmark(cache, node=None):
    """Keep the optional sampler tools in the perception cache."""
    node = Path(node or shutil.which("node") or "node").absolute()
    if not node.is_file():
        raise RuntimeError("Install Node.js 20+ or pass --node /path/to/node")
    version = subprocess.check_output([str(node), "--version"], text=True).strip()
    if int(version.lstrip("v").split(".")[0]) < 20:
        raise RuntimeError("Native benchmark recovery requires Node.js 20+")
    env = dict(os.environ, PATH=str(node.parent) + os.pathsep + os.environ.get("PATH", ""))
    npm = shutil.which("npm", path=env["PATH"])
    if not npm:
        raise RuntimeError("The Node.js installation must include npm")
    source = cache / "sources/rcr-perception-score"
    if (source / ".git").exists() and subprocess.run(
        ["git", "-C", str(source), "cat-file", "-e", BENCHMARK_COMMIT + "^{commit}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode:
        run(["git", "-C", source, "fetch", "origin", BENCHMARK_COMMIT])
    clone("https://github.com/jackybaltes/rcr-perception.git", source, BENCHMARK_COMMIT)
    run(
        [
            npm,
            "install",
            "--prefix",
            cache / "benchmark_js",
            "--ignore-scripts",
            "--no-audit",
            "--no-fund",
            "three@0.180.0",
        ],
        env=env,
    )
    scanner = cache / "bin/rcr-four-seedscan"
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    run([nvcc, "-O3", "-arch=native", PACKAGE / "native/hold_seedscan.cu", "-o", scanner])
    spec = dict(
        node=str(node),
        source=str(source),
        scanner=str(scanner),
        three=str(cache / "benchmark_js/node_modules/three"),
        source_commit=BENCHMARK_COMMIT,
        scanner_source_sha256=hashlib.sha256(
            (PACKAGE / "native/hold_seedscan.cu").read_bytes()
        ).hexdigest(),
    )
    (cache / "benchmark_runtime.json").write_text(json.dumps(spec, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", type=Path, default=REPO / ".venv-perception")
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(os.environ.get("RCR_PERCEPTION_HOME", Path.home() / ".cache/rcr-perception")),
    )
    parser.add_argument(
        "--robot", action="store_true", help="Install the learned robot stack, model and native M3T"
    )
    parser.add_argument(
        "--cameras", action="store_true", help="Install color-only RealSense capture support"
    )
    parser.add_argument("--native", action="store_true", help="Build the M3T executable")
    parser.add_argument("--cuda", action="store_true", help="Check the CUDA development toolchain")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Prepare optional native-scene recovery for the scored hold benchmark",
    )
    parser.add_argument("--node", type=Path, help="Node.js 20+ executable for --benchmark")
    parser.add_argument(
        "--cuda-arch", default="8.9+PTX", help="PyTorch extension target architectures"
    )
    parser.add_argument("--micromamba", type=Path)
    parser.add_argument(
        "--inherit-python",
        type=Path,
        help="Reuse this existing Python's packages read-only in the new private environment",
    )
    parser.add_argument(
        "--wheel-dir", type=Path, help="Optional cache of compatible nvdiffrast/RealSense wheels"
    )
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        parser.error("Use Python 3.12 for the tested runtime")
    environment = args.env.resolve()
    if environment == Path(sys.prefix).resolve():
        parser.error("The runtime environment must differ from the invoking environment")
    python = environment / "bin/python"
    if not python.exists():
        run([sys.executable, "-m", "venv", environment])
    if args.inherit_python:
        code = "import sysconfig; print(sysconfig.get_paths()['purelib'])"
        base = subprocess.check_output([str(args.inherit_python), "-c", code], text=True).strip()
        target = subprocess.check_output([str(python), "-c", code], text=True).strip()
        (Path(target) / "perception_base.pth").write_text(
            "import site; site.addsitedir(" + repr(base) + ")\n"
        )
    pip = [python, "-m", "pip", "install"]
    run([*pip, "-r", REPO / "scripts/perception/requirements-core.txt"])
    run([*pip, "--no-deps", "-e", REPO])
    cache = args.cache.expanduser().resolve()
    (cache / "bin").mkdir(parents=True, exist_ok=True)
    for name, spec in json.loads((PACKAGE / "assets/downloads.json").read_text()).items():
        download(spec["url"], cache / "assets/rcr" / name, spec["sha256"])
    if args.cuda:
        nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
        run([nvcc, "--version"])
    if args.cameras or args.robot:
        run([*pip, "pyrealsense2==2.59.1.11053"])
    if args.robot:
        run([*pip, "setuptools", "wheel", "ninja"])
        valid_torch = (
            subprocess.run(
                [
                    str(python),
                    "-c",
                    (
                        "import torch; assert torch.__version__.startswith('2.9."
                        "1') and torch.version.cuda=='12.8'"
                    ),
                ],
                capture_output=True,
            ).returncode
            == 0
        )
        if not valid_torch:
            run(
                [
                    *pip,
                    "--force-reinstall",
                    "torch==2.9.1",
                    "torchvision==0.24.1",
                    "--index-url",
                    "https://download.pytorch.org/whl/cu128",
                ]
            )
        run([*pip, "-r", REPO / "scripts/perception/requirements-robot.txt"])
        wheels = list(args.wheel_dir.glob("nvdiffrast-0.4.0-*.whl")) if args.wheel_dir else []
        env = dict(os.environ, TORCH_CUDA_ARCH_LIST=args.cuda_arch, MAX_JOBS="2")
        run(
            [
                *pip,
                "--no-build-isolation",
                "--no-deps",
                str(wheels[0]) if wheels else "git+https://github.com/NVlabs/nvdiffrast.git@v0.4.0",
            ],
            env=env,
        )
        model = cache / "sources/microban-model"
        clone("https://github.com/Rhoban/microban_model.git", model, MODEL_COMMIT, lfs=True)
        run([*pip, "--no-deps", model])
    if args.native or args.robot:
        native(cache, args.micromamba)
    if args.benchmark:
        benchmark(cache, args.node)
    (environment / "runtime.json").write_text(
        json.dumps(
            {
                "cache": str(cache),
                "robot": args.robot,
                "cuda": args.cuda,
                "python": str(python),
                "source": str(REPO),
            },
            indent=2,
        )
    )
    print(f"Ready: RCR_PERCEPTION_HOME={cache} {python} scripts/perception/run.py doctor")


if __name__ == "__main__":
    main()
