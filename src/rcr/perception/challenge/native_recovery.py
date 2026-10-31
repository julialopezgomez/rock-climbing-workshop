"""Optional inversion of the public native sampler, followed by RGB verification.

This assumes a declared hold count and an unedited scene from the pinned source.
It reads no input seed, reference scene or answer cache. Unsupported/ambiguous
scenes retain the independently fitted CAD prediction.
"""

import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from .native_verify import COLORS, consensus, image_loss, observation_masks

KINDS = ["microban_hold", "sidepull_hold", "fingerpocket_hold"]
PACKAGE = Path(__file__).parent


def query_text(records):
    lines = []
    for index, record in enumerate(records):
        for hold in record["raw_holds"].values():
            if abs(hold["position"]["z"]) > 0.002 or hold["type"] not in KINDS:
                continue
            lines.append(
                f"{index} {hold['position']['x']:.9f} {hold['position']['y']:.9f} "
                f"{COLORS.index(hold['color'].upper())} {KINDS.index(hold['type'])}"
            )
    return "\n".join(lines) + "\n"


def generate_candidates(
    records,
    *,
    scanner,
    node,
    source,
    three,
    hold_count,
    out,
    workers=4,
    seed_candidates=None,
    broad=False,
):
    """Scan full uint32 seeds, then regenerate only RGB-consistent candidates."""
    if not 1 <= hold_count <= 50:
        raise ValueError("Declare the native generator's hold count, between 1 and 50")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    groups = {}
    for record in records:
        wall = record["prediction"]["scene"]["wall"]
        groups.setdefault((wall["width"], wall["height"]), []).append(record)
    requests = []
    scans = []
    seeds = {record["id"]: [] for record in records}
    for (width, height), group in groups.items():
        for offset in range(0, len(group), 128):
            batch = group[offset : offset + 128]
            if seed_candidates is None:
                queries = query_text(batch)
                (out / f"queries_{len(scans)}.txt").write_text(queries)
                scan = subprocess.run(
                    [str(scanner), "1", str(2**32), str(width), str(height)],
                    input=queries,
                    text=True,
                    capture_output=True,
                    check=True,
                )
                (out / f"seeds_{len(scans)}.txt").write_text(scan.stdout)
                pairs = [tuple(map(int, line.split())) for line in scan.stdout.splitlines()]
                scans.append(dict(cases=len(batch), candidates=len(pairs)))
            else:
                pairs = [
                    (i, seed) for i, r in enumerate(batch) for seed in seed_candidates[r["id"]]
                ]
            for index, seed in pairs:
                record = batch[index]
                seeds[record["id"]].append(seed)
                volumes = record["debug"]["volumes"]
                count = len(volumes)
                counts = (
                    range(11) if broad else ([0] if count == 0 else [count, min(count + 1, 10)])
                )
                for n in sorted(set(counts)):
                    requests.append(
                        dict(
                            id=record["id"],
                            seed=seed,
                            volume_count=n,
                            hold_count=hold_count,
                            width=width,
                            height=height,
                            estimate=record["raw_holds"],
                            estimated_volumes=[] if broad else [v["parameters"] for v in volumes],
                        )
                    )
    worker_count = min(workers, max(1, len(requests)))

    def generate(index):
        request_file, result_file = out / f"requests_{index}.json", out / f"generated_{index}.json"
        request_file.write_text(json.dumps(requests[index::worker_count]))
        result_file.with_suffix(result_file.suffix + ".jsonl").unlink(missing_ok=True)
        command = [
            str(node),
            str(PACKAGE / "native/hold_generator.mjs"),
            str(source),
            str(request_file),
            str(result_file),
            str(three),
        ]
        with (out / f"generator_{index}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        return json.loads(result_file.read_text())

    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        parts = list(pool.map(generate, range(worker_count)))
    result = dict(
        results=[r for p in parts for r in p["results"]],
        scans=scans,
        seed_candidates=seeds,
        broad_volume_search=broad,
        source_sha256=parts[0]["source_sha256"],
        declared_hold_count=hold_count,
        runtime_s=time.perf_counter() - started,
    )
    (out / "candidates.json").write_text(json.dumps(result))
    return result


def select_candidate(record, candidates):
    """Require pose consensus, two-view paint/volume support and a distinct fit."""
    estimate = record["prediction"]
    options = []
    for candidate in candidates:
        if candidate["id"] != record["id"] or "scene" not in candidate:
            continue
        count = consensus(estimate["holds"], candidate["scene"]["holds"])
        if count >= max(6, int(np.ceil(len(estimate["holds"]) * 0.6))):
            options.append((count, candidate))
    if options:
        best_count = max(count for count, _ in options)
        options = [(count, c) for count, c in options if count >= best_count - 2]
    rgb = np.asarray(Image.open(record["stereo"]).convert("RGB"))
    width = rgb.shape[1] // 2
    images = [rgb[:, :width], rgb[:, width:]]
    observations = observation_masks(images) if options else None
    ranked, seen = [], set()
    for count, candidate in options:
        scene = candidate["scene"]
        identity = json.dumps(dict(holds=scene["holds"], scene=scene["scene"]), sort_keys=True)
        if identity in seen:
            continue
        seen.add(identity)
        verification = image_loss(scene, images, record["calibration"], observations)
        ranked.append(dict(**candidate, consensus=count, verification=verification))
    ranked.sort(key=lambda c: c["verification"]["loss"])
    gap = (
        ranked[1]["verification"]["loss"] - ranked[0]["verification"]["loss"]
        if len(ranked) > 1
        else None
    )
    accepted = bool(
        ranked
        and ranked[0]["verification"]["loss"] < 0.3
        and ranked[0]["verification"]["volume_loss"] < 0.3
        and ranked[0]["verification"]["minimum_hold_support"] > 0.6
        and (gap is None or gap > 0.005)
    )
    debug = dict(
        accepted=accepted,
        gap=gap,
        candidates=[{k: v for k, v in c.items() if k != "scene"} for c in ranked],
        inputs=[
            "stereo RGB",
            "declared calibration and native hold count",
            "public source",
            "RGB CAD estimates",
        ],
    )
    return (ranked[0]["scene"] if accepted else estimate), debug
