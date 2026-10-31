"""Grouped delivery preserves source images while sharing one inference budget."""

import importlib.util
import io
import json
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

Image = pytest.importorskip("PIL.Image")
SCRIPT = Path(__file__).resolve().parents[1] / "scripts/perception/submit_folders.py"


@pytest.fixture
def grouped(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("perception_submit_folders", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_inputs(root):
    buffer = io.BytesIO()
    Image.new("RGB", (40, 20), "gray").save(buffer, format="PNG")
    content = buffer.getvalue()
    for category, name in [("real", "a.zip"), ("real", "b.zip"), ("sim", "a.zip")]:
        folder = root / category
        folder.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(folder / name, "w") as archive:
            archive.writestr("examples/scene_cam1_0001.png", content)
            archive.writestr("examples/scene_0001.json", "DO NOT READ THIS ANSWER")
            archive.writestr("__MACOSX/._scene_cam1_0001.png", b"metadata")
    return content


def test_grouped_delivery_preserves_images_and_paths_without_reading_answers(
    grouped, tmp_path, monkeypatch
):
    root, output = tmp_path / "inputs", tmp_path / "outputs"
    original_image = make_inputs(root)
    original_read = zipfile.ZipFile.read

    def image_only(archive, name, *args, **kwargs):
        assert not str(name).endswith(".json")
        return original_read(archive, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "read", image_only)
    calls = []

    def inference(settings, started, on_package):
        calls.append(settings)
        assert settings.seconds == 72
        assert settings.case_seconds == 24
        assert settings.native_reserve == 18
        internal = sorted(settings.inputs[0].glob("*.png"))
        assert len(internal) == len({image.name for image in internal}) == 3
        for i, image in enumerate(internal):
            grouped.submit.write_json(
                settings.out / "json" / image.with_suffix(".json").name, {"predicted_image": i}
            )
        on_package(
            dict(
                state="complete",
                inferred=3,
                native_recovered=3,
                cases=[
                    dict(filename=image.with_suffix(".json").name, status="refined")
                    for image in internal
                ],
                uncertain_cases=[],
            )
        )
        return 0

    monkeypatch.setattr(grouped.submit, "run", inference)
    args = SimpleNamespace(
        input_root=root, out=output, seconds=480, seconds_per_sample=24, workers=4
    )
    assert grouped.run(args, time.monotonic()) == 0
    assert len(calls) == 1
    for category, name in [("real", "a.zip"), ("real", "b.zip"), ("sim", "a.zip")]:
        with zipfile.ZipFile(output / category / name) as archive:
            assert set(archive.namelist()) == {
                "examples/scene_cam1_0001.png",
                "examples/scene_0001.json",
            }
            assert original_read(archive, "examples/scene_cam1_0001.png") == original_image
            assert "predicted_image" in json.loads(
                original_read(archive, "examples/scene_0001.json")
            )
    report = json.loads((output / "perception_manifest.json").read_text())
    assert report["state"] == "complete" and report["inferred"] == 3


def test_failed_inference_keeps_safe_initial_archives(grouped, tmp_path, monkeypatch):
    root, output = tmp_path / "inputs", tmp_path / "outputs"
    make_inputs(root)

    def unavailable(*args, **kwargs):
        raise RuntimeError("GPU unavailable")

    monkeypatch.setattr(grouped.submit, "run", unavailable)
    args = SimpleNamespace(
        input_root=root, out=output, seconds=480, seconds_per_sample=24, workers=4
    )
    with pytest.raises(RuntimeError, match="GPU unavailable"):
        grouped.run(args, time.monotonic())
    for category, name in [("real", "a.zip"), ("real", "b.zip"), ("sim", "a.zip")]:
        with zipfile.ZipFile(output / category / name) as archive:
            assert json.loads(archive.read("examples/scene_0001.json"))["holds"] == {}
    assert json.loads((output / "perception_manifest.json").read_text())["state"] == "failed"


def test_shared_budget_settings(grouped, tmp_path):
    args = SimpleNamespace(workers=4, seconds_per_sample=24)
    settings = grouped.runner_args(args, 40, 480, tmp_path)
    assert settings.seconds == 480 and settings.case_seconds == 12
    assert settings.native_reserve == 90
    assert settings.native_workers == 8
    assert settings.gpu_fallback_engine.endswith(":budgeted_hold_prediction")


def test_exhausted_preparation_budget_skips_preflight(grouped, tmp_path, monkeypatch):
    root, output = tmp_path / "inputs", tmp_path / "outputs"
    make_inputs(root)
    now = [0.0]
    monkeypatch.setattr(grouped.time, "monotonic", lambda: now[0])
    original_prepare = grouped.prepare

    def prepare(*args):
        original_prepare(*args)
        now[0] = 68.0

    def no_inference(*args, **kwargs):
        pytest.fail("GPU preflight must not begin after the inference deadline")

    monkeypatch.setattr(grouped, "prepare", prepare)
    monkeypatch.setattr(grouped.submit, "run", no_inference)
    args = SimpleNamespace(
        input_root=root, out=output, seconds=480, seconds_per_sample=24, workers=4
    )
    assert grouped.run(args, 0.0) == 2
    report = json.loads((output / "perception_manifest.json").read_text())
    assert report["state"] == "partial" and report["inferred"] == 0
    assert (output / "real/a.zip").exists()


def test_inventory_rejects_conflicting_prediction_names(grouped, tmp_path):
    folder = tmp_path / "real"
    folder.mkdir()
    with zipfile.ZipFile(folder / "conflict.zip", "w") as archive:
        archive.writestr("scene_cam1_0001.png", b"image")
        archive.writestr("scene_0001.png", b"image")
    with pytest.raises(ValueError, match="Duplicate image or prediction"):
        grouped.inventory(tmp_path)
