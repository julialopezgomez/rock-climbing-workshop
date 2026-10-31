"""Submission packaging must ignore labels and survive bounded inference failure."""

import importlib.util
import io
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

Image = pytest.importorskip("PIL.Image")
SCRIPT = Path(__file__).resolve().parents[1] / "scripts/perception/submit.py"


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("perception_submit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rig(**overrides):
    return SimpleNamespace(
        wall_width=3.0,
        wall_height=1.4,
        camera_height=0.7,
        distance=1.0,
        baseline=0.064,
        fov=74.0,
        **overrides,
    )


def png():
    buffer = io.BytesIO()
    Image.new("RGB", (40, 20), "gray").save(buffer, format="PNG")
    return buffer.getvalue()


def test_preparation_never_opens_reference_json(runner, tmp_path, monkeypatch):
    archive_path = tmp_path / "sample.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("sample/real_zero_volumes_cam1_0001.png", png())
        archive.writestr("sample/real_zero_volumes_0001.json", "malformed secret answer")
    original = zipfile.ZipFile.read

    def image_only(self, member, *args, **kwargs):
        name = member.filename if isinstance(member, zipfile.ZipInfo) else member
        assert not name.endswith(".json")
        return original(self, member, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "read", image_only)
    output = tmp_path / "out"
    cases = runner.prepare_inputs([archive_path], output, rig())
    assert len(cases) == 1
    assert cases[0]["filename"] == "real_zero_volumes_0001.json"
    assert not list((output / "work").rglob("*expected*"))
    prediction = json.loads((output / "json/real_zero_volumes_0001.json").read_text())
    assert prediction["holds"] == {}


def test_duplicate_names_are_rejected(runner, tmp_path):
    paths = []
    for name in ["a", "b"]:
        folder = tmp_path / name
        folder.mkdir()
        image = folder / "same.png"
        image.write_bytes(png())
        paths.append(image)
    with pytest.raises(ValueError, match="Duplicate submission"):
        runner.prepare_inputs(paths, tmp_path / "out", rig())


def test_zip_contains_predictions_only(runner, tmp_path):
    image = tmp_path / "scene.png"
    image.write_bytes(png())
    output = tmp_path / "out"
    cases = runner.prepare_inputs([image], output, rig())
    runner.package(output, cases, time.monotonic(), "prepared")
    with zipfile.ZipFile(output / "submission.zip") as archive:
        assert archive.namelist() == ["scene.json"]
        assert json.loads(archive.read("scene.json"))["formatVersion"] == 1
    assert json.loads((output / "manifest.json").read_text())["inferred"] == 0


def test_timed_out_worker_keeps_a_complete_upload(runner, tmp_path, monkeypatch):
    # This substitutes a sleeping CPU-only frontend; no GPU work is launched.
    engine = tmp_path / "slow_frontend.py"
    engine.write_text("import time\ndef predict(*args): time.sleep(30)\n")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setattr(runner, "gpu_preflight", lambda _: (tmp_path / "lock").open("w"))
    image = tmp_path / "scene.png"
    image.write_bytes(png())
    args = rig(
        inputs=[image],
        out=tmp_path / "out",
        check=False,
        seconds=6,
        native_hold_count=None,
        native_reserve=0,
        workers=1,
        case_seconds=0.1,
        engine="slow_frontend:predict",
    )
    started = time.monotonic()
    result = runner.run(args, started)
    assert time.monotonic() - started < 5
    assert result == 2
    report = json.loads((args.out / "manifest.json").read_text())
    assert report["inferred"] == 0
    assert report["state"] == "partial"
    assert report["errors"]
    with zipfile.ZipFile(args.out / "submission.zip") as archive:
        assert archive.namelist() == ["scene.json"]


def test_mixed_directory_includes_archives_and_loose_images(runner, tmp_path):
    (tmp_path / "loose.png").write_bytes(png())
    with zipfile.ZipFile(tmp_path / "images.zip", "w") as archive:
        archive.writestr("archived.png", png())
    names = [Path(name).name for name, _ in runner.image_members([tmp_path])]
    assert sorted(names) == ["archived.png", "loose.png"]


def test_macos_metadata_is_ignored_in_archives_and_directories(runner, tmp_path):
    (tmp_path / "loose.png").write_bytes(png())
    (tmp_path / "._loose.png").write_bytes(b"metadata")
    (tmp_path / "._images.zip").write_bytes(b"metadata")
    metadata = tmp_path / "__MACOSX"
    metadata.mkdir()
    (metadata / "image.png").write_bytes(b"metadata")
    with zipfile.ZipFile(tmp_path / "images.zip", "w") as archive:
        archive.writestr("archived.png", png())
        archive.writestr("._archived.png", b"metadata")
        archive.writestr("__MACOSX/archived.png", b"metadata")
    names = [Path(name).name for name, _ in runner.image_members([tmp_path])]
    assert sorted(names) == ["archived.png", "loose.png"]


def test_manifest_reports_native_uncertainty_without_changing_predictions(runner, tmp_path):
    image = tmp_path / "scene.png"
    image.write_bytes(png())
    output = tmp_path / "out"
    cases = runner.prepare_inputs([image], output, rig())
    case = cases[0]
    case.update(status="refined", native_accepted=True, native_pass="initial", native_loss=0.4)
    for evidence in [
        {"native_loss": 0.4},
        {"native_loss": None},
        {"native_loss": 0.1, "native_accepted": False},
        {"native_loss": 0.1, "native_accepted": True, "native_pending": True},
    ]:
        case.update(evidence)
        report = runner.package(output, cases, time.monotonic(), "complete", native_enabled=True)
        assert report["uncertain_cases"] == ["scene.json"]
        assert report["state"] == "complete_with_warnings"
    case.pop("native_pending")
    for stage, loss in [("initial", 0.1), ("gpu", 0.4)]:
        case.update(native_pass=stage, native_loss=loss)
        report = runner.package(output, cases, time.monotonic(), "complete", native_enabled=True)
        assert report["uncertain_cases"] == []
        assert report["state"] == "complete"
    report = runner.package(output, cases, time.monotonic(), "complete", native_enabled=False)
    assert "uncertain_cases" not in report


def test_group_cleanup_reaches_child_after_leader_exit(runner):
    child = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"
    leader = (
        "import subprocess,sys,time; "
        "p=subprocess.Popen([sys.executable,'-c',sys.argv[1]]); "
        "print(p.pid,flush=True); time.sleep(.2)"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", leader, child],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    child_pid = int(process.stdout.readline())
    process.wait(timeout=3)
    try:
        runner.stop_process(process)
        stat = Path(f"/proc/{child_pid}/stat")
        until = time.monotonic() + 1
        while stat.exists() and stat.read_text().split()[2] != "Z" and time.monotonic() < until:
            time.sleep(0.01)
        # A reparented zombie is already terminated and no longer consumes GPU/CPU.
        assert not stat.exists() or stat.read_text().split()[2] == "Z"
    finally:
        try:
            os.kill(child_pid, 9)
        except ProcessLookupError:
            pass


def test_empty_estimator_result_is_not_reported_as_inferred(runner, tmp_path):
    image = tmp_path / "scene.png"
    image.write_bytes(png())
    output = tmp_path / "out"
    cases = runner.prepare_inputs([image], output, rig())
    source = output / "json/scene.json"
    assert runner.publish(source, output, cases[0], "image_fit")
    report = runner.package(output, cases, time.monotonic(), "partial")
    assert report["inferred"] == 0
    assert cases[0]["status"] == "empty_prediction"


def test_worker_forwards_image_fit_budget(runner, tmp_path, monkeypatch):
    received = []

    def predict(*args):
        received.append((os.environ["RCR_IMAGE_FIT_SECONDS"], args))

    monkeypatch.setitem(sys.modules, "budget_frontend", SimpleNamespace(predict=predict))
    monkeypatch.setenv("RCR_IMAGE_FIT_SECONDS", "old")
    task = tmp_path / "task.json"
    runner.write_json(
        task,
        dict(
            engine="budget_frontend:predict",
            fit_seconds=24,
            stereo="rgb.png",
            calibration="rig.json",
            output="out",
        ),
    )
    assert runner.run_worker(SimpleNamespace(worker=task)) == 0
    assert received == [("24", ("rgb.png", "rig.json", "out"))]


def test_adaptive_pass_refines_uncertain_case_and_protects_native_output(
    runner, tmp_path, monkeypatch
):
    hold = dict(
        type="microban_hold",
        color="#F2554E",
        position=dict(x=0.3, y=0.5, z=0),
        orientation=dict(roll=0, pitch=0, yaw=0),
    )
    document = dict(
        formatVersion=1,
        seed=0,
        materialSeed=0,
        scene=dict(wall=dict(width=3, height=1.4, depth=0.025), volumes=[]),
        holds={"hold_1": hold},
    )
    engine = tmp_path / "fixture_engine.py"
    engine.write_text(
        "import json\nfrom pathlib import Path\n"
        f"DOCUMENT={document!r}\n"
        "def predict(stereo,calibration,out):\n"
        " p=Path(out);p.mkdir(parents=True,exist_ok=True)\n"
        " (p/'prediction.json').write_text(json.dumps(DOCUMENT))\n"
        "def gpu(stereo,calibration,out):\n"
        " DOCUMENT['holds']['hold_1']['position']['x']=.9\n"
        " predict(stereo,calibration,out)\n"
    )
    fake_native = tmp_path / "native_fixture.py"
    fake_native.write_text(
        "import json,sys\nfrom pathlib import Path\n"
        f"DOCUMENT={document!r}\n"
        "root=Path(sys.argv[1]);manifest=json.loads(Path(sys.argv[2]).read_text())\n"
        "gpu=root.name.endswith('_gpu')\n"
        "for case in manifest['cases']:\n"
        " p=root/case['id'];p.mkdir(parents=True,exist_ok=True)\n"
        " DOCUMENT['holds']['hold_1']['position']['x']=.2 if gpu else .1\n"
        " (p/'prediction.json').write_text(json.dumps(DOCUMENT))\n"
        " loss=.4 if case['id']=='scene_0000' and not gpu else .1\n"
        " native={'accepted':True,'candidates':[{'verification':{'loss':loss}}]}\n"
        " (p/'debug.json').write_text(json.dumps({'native':native}))\n"
    )
    actual_popen = subprocess.Popen

    def launch(command, **kwargs):
        if Path(command[1]).name == "benchmark_max.py":
            output = command[command.index("--out") + 1]
            dataset = command[command.index("--dataset") + 1]
            command = [
                sys.executable,
                str(fake_native),
                output,
                str(Path(dataset) / "manifest.json"),
            ]
        return actual_popen(command, **kwargs)

    monkeypatch.setattr(runner.subprocess, "Popen", launch)
    monkeypatch.setattr(runner, "gpu_preflight", lambda _: (tmp_path / "lock").open("w"))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    images = []
    for name in ["one.png", "two.png"]:
        path = tmp_path / name
        path.write_bytes(png())
        images.append(path)
    args = rig(
        inputs=images,
        out=tmp_path / "out",
        check=False,
        seconds=30,
        native_hold_count=30,
        native_reserve=8,
        workers=2,
        case_seconds=5,
        engine="fixture_engine:predict",
        gpu_fallback=True,
        gpu_fallback_engine="fixture_engine:gpu",
    )
    package = runner.package
    published_x = []

    def record(*params, **kwargs):
        output = args.out / "json/one.json"
        if output.exists():
            holds = json.loads(output.read_text())["holds"]
            if holds:
                published_x.append(holds["hold_1"]["position"]["x"])
        return package(*params, **kwargs)

    monkeypatch.setattr(runner, "package", record)
    assert runner.run(args, time.monotonic()) == 0
    assert (args.out / "work/gpu_cad/scene_0000/prediction.json").exists()
    assert not (args.out / "work/gpu_cad/scene_0001").exists()
    assert 0.9 not in published_x
    assert published_x[-1] == 0.2
    report = json.loads((args.out / "manifest.json").read_text())
    assert report["native_recovered"] == 2
    assert [c["native_pass"] for c in report["cases"]] == ["gpu", "initial"]
