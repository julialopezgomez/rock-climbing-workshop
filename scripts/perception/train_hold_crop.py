#!/usr/bin/env python3
"""Train the local hold-crop classifier from development CAD annotations.

This command requires scikit-learn for training. The exported NumPy model runs
without scikit-learn. Each input NPZ contains rgb, instance_ids and hold_types;
files ending in _00 through _11 form the declared development set.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from rcr.perception.challenge.improved_pose import CLASSES, features
from rcr.perception.challenge.photo_frontend import component_masks


def samples(path):
    with np.load(path) as arrays:
        rgb = arrays["rgb"]
        ids = arrays["instance_ids"]
        types = arrays["hold_types"].tolist()
    index = int(path.stem.rsplit("_", 1)[1])
    if not 0 <= index < 12:
        raise ValueError(f"Refusing a scene outside the declared development set: {path}")
    internal_validation = index >= 8
    width = rgb.shape[1] // 2
    for eye in range(2):
        image = rgb[:, eye * width : (eye + 1) * width]
        segmentation = ids[:, eye * width : (eye + 1) * width]
        points = []
        for mask in component_masks(image):
            count, labels, _, centers = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
            for instance in range(1, count):
                values = segmentation[labels == instance]
                foreground = values[values > 0]
                label = 0
                if len(foreground) > max(3, len(values) * 0.2):
                    identities, counts = np.unique(foreground, return_counts=True)
                    identity = int(identities[np.argmax(counts)])
                    label = CLASSES.index(types[identity - 1])
                points.append((centers[instance], label))
        for identity, kind in enumerate(types, 1):
            yy, xx = np.where(segmentation == identity)
            if len(xx) >= 5:
                points.append((np.array([xx.mean(), yy.mean()]), CLASSES.index(kind)))
        for center, label in points:
            offsets = [(0, 0)] if internal_validation else [(0, 0), (-2, -1), (2, 1)]
            for offset in offsets:
                yield features(image, center + offset), label, internal_validation


def export_model(model, destination):
    arrays = {"trees": len(model.estimators_)}
    for index, estimator in enumerate(model.estimators_):
        tree = estimator.tree_
        values = tree.value[:, 0, :]
        values = values / np.maximum(values.sum(axis=1, keepdims=True), 1e-8)
        for name, value in {
            "feature": tree.feature.astype(np.int32),
            "threshold": tree.threshold.astype(np.float32),
            "left": tree.children_left.astype(np.int32),
            "right": tree.children_right.astype(np.int32),
            "value": values.astype(np.float32),
        }.items():
            arrays[f"{name}_{index}"] = value
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    files = sorted(args.labels.glob("*.npz"))
    if not files:
        parser.error("No development label archives found")
    records = [record for path in files for record in samples(path)]
    x = np.array([r[0] for r in records])
    y = np.array([r[1] for r in records])
    validation = np.array([r[2] for r in records])
    if not validation.any() or validation.all():
        parser.error("Both internal development training and validation scenes are required")
    model = ExtraTreesClassifier(
        n_estimators=96,
        max_depth=18,
        min_samples_leaf=2,
        max_features=0.35,
        n_jobs=8,
        random_state=0,
        class_weight="balanced",
    )
    model.fit(x[~validation], y[~validation])
    predicted = model.predict(x[validation])
    report = dict(
        development_images=len(files),
        internal_validation=classification_report(y[validation], predicted, output_dict=True),
        confusion=confusion_matrix(y[validation], predicted).tolist(),
        classes=CLASSES,
        features=x.shape[1],
        training_labels="Development-only CAD renders from previously recovered scenes",
        inference_inputs="Local HOG and saturation/value pixels; no scene coordinates or identity",
        input_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
    )
    model.fit(x, y)
    export_model(model, args.model)
    report["model_sha256"] = hashlib.sha256(args.model.read_bytes()).hexdigest()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"model": str(args.model), "model_sha256": report["model_sha256"]}))


if __name__ == "__main__":
    main()
