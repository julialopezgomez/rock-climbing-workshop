#!/usr/bin/env python3
"""Train the paint frontend from development RGB/semantic-mask NPZ files.

Run in an environment containing scikit-learn. The exported runtime model
requires NumPy only. Annotation files must contain stereo ``rgb`` and ``labels``
arrays, with labels 0=background, 1=red, 2=green and 3=blue.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from rcr.perception.challenge.improved_masks import pixel_features


def train(dataset, output):
    cv2.setNumThreads(1)
    rng = np.random.default_rng(1207)
    inputs, targets, sources = [], [], []
    paths = sorted(dataset.glob("real_v[03]_[0-9][0-9].npz"))
    if not paths:
        raise ValueError("No real-style development annotations found")
    for path in paths:
        with np.load(path) as data:
            rgb, labels = data["rgb"], data["labels"]
        if rgb.shape[:2] != labels.shape or rgb.shape[1] % 2:
            raise ValueError(f"Invalid stereo annotation: {path}")
        width = rgb.shape[1] // 2
        for offset in (0, width):
            features = pixel_features(rgb[:, offset : offset + width]).reshape(-1, 15)
            classes = labels[:, offset : offset + width].ravel()
            for label in range(4):
                candidates = np.flatnonzero(classes == label)
                count = min(len(candidates), 4000 if label == 0 else 1200)
                if count:
                    selected = rng.choice(candidates, count, replace=False)
                    inputs.append(features[selected])
                    targets.append(classes[selected])
        sources.append(
            dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        )
    x = np.concatenate(inputs).astype(np.float32)
    y = np.concatenate(targets).astype(np.int32)
    model = ExtraTreesClassifier(
        n_estimators=24, max_depth=16, min_samples_leaf=4, n_jobs=2, random_state=1207
    )
    model.fit(x, y)
    if list(model.classes_) != [0, 1, 2, 3]:
        raise ValueError("Development annotations must contain every paint class")
    arrays = {"trees": np.array(len(model.estimators_))}
    for index, estimator in enumerate(model.estimators_):
        tree = estimator.tree_
        value = tree.value[:, 0, :]
        value = value / np.maximum(value.sum(axis=1, keepdims=True), 1e-12)
        for name, data in (
            ("feature", tree.feature.astype(np.int16)),
            ("threshold", tree.threshold.astype(np.float32)),
            ("left", tree.children_left.astype(np.int32)),
            ("right", tree.children_right.astype(np.int32)),
            ("value", value.astype(np.float32)),
        ):
            arrays[f"{name}_{index}"] = data
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    metadata = dict(
        model="ExtraTrees local-colour segmentation",
        classes=["background", "red", "green", "blue"],
        development_images=len(paths),
        training_pixels=len(x),
        features="RGB, HSV, RGB-minus-median5, RGB-minus-median31, normalized RGB",
        excluded_features=["pixel coordinates", "image identity", "scene seed", "geometry"],
        trees=24,
        maximum_depth=16,
        minimum_leaf_samples=4,
        random_state=1207,
        annotation_sources=sources,
        model_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
    )
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output path to preserve the selected model")
    result = train(args.annotations, args.output)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "annotation_sources"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
