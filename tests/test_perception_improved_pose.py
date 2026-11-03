"""Controls for local-only features and post-classification record alignment."""

from types import SimpleNamespace

import numpy as np

from rcr.perception.challenge import improved_pose


def test_crop_features_do_not_encode_image_position():
    first = np.full((100, 100, 3), 140, np.uint8)
    second = first.copy()
    first[25:36, 25:36] = [30, 160, 60]
    second[65:76, 65:76] = [30, 160, 60]
    np.testing.assert_array_equal(
        improved_pose.features(first, [30, 30]), improved_pose.features(second, [70, 70])
    )


def test_rejection_preserves_hold_record_alignment(monkeypatch):
    from rcr.perception.challenge.src import rcr_methods

    monkeypatch.setattr(
        rcr_methods, "normalized_mesh", lambda kind: SimpleNamespace(vertices=np.zeros((3, 3)))
    )
    monkeypatch.setattr(
        improved_pose,
        "probabilities",
        lambda image, center: (
            np.array([0.99, 0.01, 0, 0]) if center[0] == 10 else np.array([0, 1, 0, 0])
        ),
    )
    hold = dict(
        type="microban_hold",
        color="#F2554E",
        position=dict(x=0, y=0, z=0),
        orientation=dict(roll=0, pitch=0, yaw=0),
    )
    debug = {
        "objects": [
            dict(centers=[[10, 10], [10, 10]], accepted=True, identity="rejected"),
            dict(centers=[[20, 20], [20, 20]], accepted=True, identity="retained"),
        ]
    }
    transform = np.eye(4)
    transform[2, 3] = 1
    camera = dict(K=np.eye(3).tolist(), T_camera_scene=transform.tolist())
    result, diagnostics = improved_pose.postprocess(
        {"hold_1": hold.copy(), "hold_2": hold.copy()},
        debug,
        np.zeros((40, 80, 3), np.uint8),
        {"cameras": {"left": camera, "right": camera}},
    )
    assert list(result) == ["hold_1"]
    assert [r["identity"] for r in diagnostics["objects"]] == ["retained"]
    assert [r["action"] for r in diagnostics["crop_classifier"]] == ["reject", "retain"]
