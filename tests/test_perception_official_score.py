"""Analytic controls for the released scene comparison, independent of fitting."""

import copy
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from rcr.perception.challenge.official_score import position_matches, score_scene


def hold(x=0.0, yaw=0.0, kind="microban_hold"):
    return dict(
        type=kind,
        color="#F2554E",
        position=dict(x=x, y=0.5, z=0.0),
        orientation=dict(roll=0.0, pitch=0.0, yaw=yaw),
    )


def scene(holds, volumes=()):
    return dict(formatVersion=1, holds=holds, scene=dict(volumes=list(volumes)))


def test_scored_weights_and_wrapping():
    reference = scene({"a": hold()})
    prediction = scene({"b": hold(0.1, math.pi / 2, "cube_hold")})
    # Final badge sums both sides: position 4, angle 2, type 4.
    assert score_scene(reference, prediction)["final_score"] == pytest.approx(10.0)
    prediction["holds"]["b"] = hold(yaw=2 * math.pi)
    assert score_scene(reference, prediction)["final_score"] == pytest.approx(0.0)


def test_missing_and_extra_objects_cost_ten_each():
    result = score_scene(scene({"a": hold()}), scene({"b": hold(0.1001)}))
    assert result["final_score"] == 20.0
    assert result["holds"]["matches"] == 0
    assert score_scene(scene({"a": hold()}), scene({}))["final_score"] == 10.0


def test_matcher_is_ordered_and_does_not_use_type():
    # The later item displaces the first through the nearest occupied match.
    # A minimum-cost assignment would give a different pair list.
    active = [hold(0.0), hold(0.08)]
    test = [hold(0.01, kind="cube_hold"), hold(0.02)]
    assert position_matches(active, test) == [(0, 1), (1, 0)]
    assert position_matches(active, list(reversed(test))) == [(0, 1), (1, 0)]


def test_volume_toggle_and_full_scene_requirement():
    volume = dict(name="v", center=dict(x=0.0, y=0.5), baseWidth=0.3, baseHeight=0.3, depth=0.1)
    reference = scene({}, [volume])
    prediction = scene({})
    assert score_scene(reference, prediction, volumes=True)["final_score"] == 10.0
    assert score_scene(reference, prediction)["final_score"] == 0.0
    assert score_scene(reference, prediction, volumes=False)["final_score"] == 0.0
    changed = copy.deepcopy(reference)
    changed["scene"]["volumes"][0]["depth"] = 0.05
    assert score_scene(reference, changed, volumes=True)["final_score"] == 0.0
    with pytest.raises(ValueError, match="full scene"):
        score_scene(reference, {}, volumes=True)


def test_dataset_cli_does_not_drop_missing_predictions(tmp_path):
    dataset, predictions = tmp_path / "data", tmp_path / "predictions"
    dataset.mkdir()
    predictions.mkdir()
    (dataset / "manifest.json").write_text(json.dumps({"cases": [{"id": "one"}, {"id": "two"}]}))
    for name in ["one", "two"]:
        folder = dataset / name
        folder.mkdir()
        (folder / "expected_scene.json").write_text(json.dumps(scene({"a": hold()})))
    (predictions / "one").mkdir()
    (predictions / "one/prediction.json").write_text(json.dumps(scene({"a": hold()})))
    output = tmp_path / "score.json"
    script = Path(__file__).resolve().parents[1] / "scripts/perception/score_scene.py"
    command = [
        sys.executable,
        str(script),
        "--dataset",
        str(dataset),
        "--predictions",
        str(predictions),
        "--out",
        str(output),
    ]
    failed = subprocess.run(command, capture_output=True, text=True)
    assert failed.returncode != 0
    assert not output.exists()
    (predictions / "two").mkdir()
    (predictions / "two/prediction.json").write_text(json.dumps(scene({})))
    subprocess.run(command, capture_output=True, text=True, check=True)
    result = json.loads(output.read_text())
    assert result["cases"] == 2
    assert result["final_score"] == 10.0
    assert result["mean_score"] == 5.0
