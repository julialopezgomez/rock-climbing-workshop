"""Fast necessary positional condition before unchanged native pose verification."""

import hashlib
import json
import subprocess
import time
from pathlib import Path

from .._paths import CACHE

PAINTS = ["#f2554e", "#59c77b", "#6389ed"]
SOURCE = Path(__file__).with_name("rcr_seedverify.cpp")
BINARY = CACHE / "bin/rcr-seedverify"


def ensure_binary():
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    metadata = BINARY.with_suffix(".build.json")
    if (
        not BINARY.exists()
        or not metadata.exists()
        or json.loads(metadata.read_text()).get("source_sha256") != digest
    ):
        command = ["g++", "-O3", "-std=c++17", "-fopenmp", str(SOURCE), "-o", str(BINARY)]
        subprocess.run(command, check=True, capture_output=True)
        metadata.write_text(json.dumps(dict(source_sha256=digest, command=command), indent=2))
    return BINARY


def filter_candidates(estimates, options, hold_count=10, maximum_volumes=3, binary=None):
    started = time.perf_counter()
    binary = Path(binary) if binary else ensure_binary()
    observations = []
    candidates = []
    for scene, holds in enumerate(estimates):
        for hold in holds.values():
            color = hold["color"].lower()
            if color not in PAINTS:
                continue
            xyz = [float(hold["position"][axis]) for axis in "xyz"]
            observations.append(" ".join(map(str, [scene, *xyz, PAINTS.index(color)])))
        candidates.extend((scene, seed) for seed in sorted(options[scene]))
    header = [len(estimates), len(observations), len(candidates), hold_count, maximum_volumes]
    data = " ".join(map(str, header)) + "\n" + "\n".join(observations) + "\n"
    data += "\n".join(f"{scene} {seed}" for scene, seed in candidates) + "\n"
    result = subprocess.run([str(binary)], input=data, text=True, capture_output=True, check=True)
    filtered = [set() for _ in estimates]
    for line in result.stdout.splitlines():
        scene, seed = map(int, line.split())
        if not 0 <= scene < len(options) or seed not in options[scene]:
            raise ValueError("Candidate filter returned an unknown seed")
        filtered[scene].add(seed)
    return filtered, dict(
        before=len(candidates),
        after=sum(map(len, filtered)),
        runtime_s=time.perf_counter() - started,
        necessary_condition=(
            "At least six distinct same-color positions within3.002m"
            "m before rounding; all ambiguous numeric boundaries pas"
            "s through"
        ),
        final_verification="Unchanged rounded poses, SO(3), consensus, uniqueness and RGB checks",
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    )
