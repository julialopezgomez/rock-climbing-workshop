"""Native CPU concurrency preserves the per-case verifier inputs and results."""

import importlib
import json
from pathlib import Path

import pytest

Image = pytest.importorskip("PIL.Image")
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/perception"


@pytest.fixture
def benchmark(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return importlib.import_module("benchmark_max")


def test_verification_tasks_contain_only_their_own_candidates(benchmark, monkeypatch):
    tasks = []

    class Future:
        def __init__(self, task):
            self.task = task

        def result(self):
            record, candidates = self.task
            return record["id"], candidates, {}

    class Pool:
        def __init__(self, **kwargs):
            assert kwargs["max_workers"] == 2

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, function, task):
            tasks.append(task)
            return Future(task)

    monkeypatch.setattr(benchmark, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(benchmark, "as_completed", reversed)
    records = [{"id": "a"}, {"id": "b"}]
    candidates = [{"id": "a", "index": 2}, {"id": "b"}, {"id": "a", "index": 1}]
    results = list(benchmark.verify_records(records, candidates, workers=8))
    assert [item[0] for item in results] == ["b", "a"]
    assert tasks[0][1] == [candidates[0], candidates[2]]
    assert tasks[1][1] == [candidates[1]]


def test_spawned_verification_matches_sequential_results(benchmark, tmp_path, monkeypatch):
    for dependency in ("cv2", "mujoco", "trimesh"):
        pytest.importorskip(dependency)
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    image = tmp_path / "stereo.png"
    Image.new("RGB", (40, 20), "gray").save(image)
    records = [
        dict(id=name, stereo=str(image), prediction={"holds": {}}, calibration={})
        for name in ["a", "b"]
    ]
    sequential = sorted(benchmark.verify_records(records, [], workers=1))
    parallel = sorted(benchmark.verify_records(records, [], workers=2))
    assert json.dumps(sequential, sort_keys=True) == json.dumps(parallel, sort_keys=True)
