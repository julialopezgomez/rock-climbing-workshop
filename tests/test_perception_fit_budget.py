"""Controls for render reuse and deadline-bounded exact CAD anchors."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("cv2")
pytest.importorskip("trimesh")

from rcr.perception.challenge import cpu_frontend, photo_frontend  # noqa: E402
from rcr.perception.challenge.src import hold_fit, hold_scene  # noqa: E402
from rcr.perception.challenge.src.rcr_methods import _project  # noqa: E402


class FakeRenderer:
    center = np.zeros(3)

    def __init__(self):
        self.calls = 0
        self.yy, self.xx = np.mgrid[:64, :64]

    def render_soft(self, point, rotation, k, transform, origin):
        self.calls += 1
        center = _project(point, k, transform)[0] - origin
        return np.exp(-((self.xx - center[0]) ** 2 + (self.yy - center[1]) ** 2) / 30)


def test_rejected_plane_update_preserves_previous_cached_render(monkeypatch):
    k = [np.array([[500.0, 0, 32], [0, 500.0, 32], [0, 0, 1]])] * 2
    transforms = [np.eye(4), np.eye(4)]
    transforms[0][:3, 3] = [0.005, 0, 1]
    transforms[1][:3, 3] = [-0.005, 0, 1]
    target = np.array([0.01, 0, 0])
    renderer = FakeRenderer()
    observations = [
        dict(
            origin=np.zeros(2),
            mask=renderer.render_soft(target, np.eye(3), ki, ti, np.zeros(2)),
            weight=np.ones((64, 64)),
        )
        for ki, ti in zip(k, transforms, strict=True)
    ]
    renderer.calls = 0
    updates = iter([np.array([1.0, 0]), np.array([2.0, 0])])
    monkeypatch.setattr(hold_fit, "minimize", lambda *a, **kw: SimpleNamespace(x=next(updates)))
    point, _, loss = hold_fit.fit_plane(
        np.zeros(3),
        np.eye(3),
        np.array([0, 0, 1]),
        np.zeros(3),
        observations,
        k,
        transforms,
        rounds=2,
        render=renderer,
    )
    np.testing.assert_array_equal(point, target)
    assert loss == 0
    assert renderer.calls == 6


def test_pair_priority_keeps_colour_coverage_and_low_loss_first():
    pairs = [(i, i) for i in range(5)]
    parts = [[dict(hue_group=hue) for hue in [0, 0, 1, 2, 1]], []]
    ordered = hold_scene.prioritize_pairs(pairs, parts, [0.5, 0.1, 0.2, 0.3, 0.05])
    assert ordered == [pairs[i] for i in [1, 4, 3, 0, 2]]
    assert sorted(ordered) == pairs
    with pytest.raises(ValueError, match="Pair losses"):
        hold_scene.prioritize_pairs(pairs, parts, [0.1])


@pytest.fixture
def scene_stubs(monkeypatch):
    calls = dict(fits=0, opened=0, closed=0)
    geometry = SimpleNamespace(width=3.0, height=1.4, volume_height=0.14)
    monkeypatch.setattr(
        hold_scene, "SceneGeometry", SimpleNamespace(from_calibration=lambda c: geometry)
    )
    monkeypatch.setattr(hold_scene, "_calibration", lambda c: ([np.eye(3)] * 2, [np.eye(4)] * 2))
    props = [dict(color=np.array([99, 137, 237]), hue_group=2)] * 2
    monkeypatch.setattr(hold_scene, "proposals", lambda image: props)
    monkeypatch.setattr(hold_scene, "estimate_volumes", lambda *a: [])
    monkeypatch.setattr(hold_scene, "refine_volume_faces", lambda *a: [])
    monkeypatch.setattr(hold_scene, "match_proposals", lambda *a: [(0, 0), (1, 1)])
    monkeypatch.setattr(hold_scene, "observations", lambda *a: None)
    monkeypatch.setattr(hold_scene, "plane_initial", lambda *a, **kw: np.array([0.1, 0.8, 0]))
    monkeypatch.setattr(hold_scene, "TYPES", {"cube_hold"})

    def fit(point, rotation, *args, **kwargs):
        calls["fits"] += 1
        return point, rotation, 0.1

    class Renderer:
        def __init__(self, **kwargs):
            calls["opened"] += 1

        def close(self):
            calls["closed"] += 1

    monkeypatch.setattr(hold_scene, "fit_plane", fit)
    monkeypatch.setattr(hold_scene, "ExactRenderer", Renderer)
    return calls


def test_deadline_finishes_all_hypotheses_for_started_pair(monkeypatch, scene_stubs):
    clock = iter([0, 0, 0, 3, 3])
    monkeypatch.setattr(hold_scene.time, "perf_counter", lambda: next(clock))
    holds, debug = hold_scene.infer([np.zeros((10, 10, 3))] * 2, {}, deadline=1)
    assert len(holds) == 1
    assert debug["partial"]
    assert (debug["processed_pairs"], debug["available_pairs"]) == (1, 2)
    assert scene_stubs == dict(fits=3, opened=1, closed=1)


def test_expired_deadline_does_not_open_renderer(monkeypatch, scene_stubs):
    monkeypatch.setattr(hold_scene.time, "perf_counter", lambda: 2)
    holds, debug = hold_scene.infer([np.zeros((10, 10, 3))] * 2, {}, deadline=1)
    assert holds == {}
    assert debug["partial"]
    assert debug["available_pairs"] == 2
    assert scene_stubs == dict(fits=0, opened=0, closed=0)


def test_budgeted_wrapper_passes_order_and_exports_completed_anchors(tmp_path, monkeypatch):
    stereo = tmp_path / "stereo.png"
    Image.fromarray(np.full((80, 160, 3), 150, np.uint8)).save(stereo)
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps(dict(image_width=80, image_height=80)))
    monkeypatch.setenv("RCR_IMAGE_FIT_SECONDS", "12")
    monkeypatch.setattr(cpu_frontend, "estimate", lambda *a: ({}, dict(losses=[0.2, 0.1])))
    captured = {}

    def fit(images, calibration, **kwargs):
        captured.update(kwargs)
        return {}, dict(runtime_s=0.01, volumes=[], partial=True)

    monkeypatch.setattr(hold_scene, "infer", fit)
    output = tmp_path / "output"
    holds, debug = photo_frontend.budgeted_hold_prediction(stereo, calibration, output)
    assert holds == {}
    assert captured["pair_losses"] == [0.2, 0.1]
    assert np.isfinite(captured["deadline"])
    assert debug["requested_engine_budget_s"] == 12
    assert debug["reserve_s"] == 2
    assert debug["partial"]
    assert json.loads((output / "image_holds.json").read_text()) == {}
    native = json.loads((output / "native_calibration.json").read_text())
    assert native["native_partial_paint_masks"]
