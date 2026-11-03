#!/usr/bin/env python3
"""Evaluate saved delivery ZIPs without running or changing their predictions."""

import argparse
import csv
import hashlib
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from rcr.perception.challenge.official_score import score_scene
from rcr.perception.challenge.src.rcr_contract import validate_holds


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def distribution(values):
    if not values:
        return dict(count=0, mean=None, median=None, p90=None, maximum=None)
    return dict(
        count=len(values),
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        p90=float(np.percentile(values, 90)),
        maximum=float(max(values)),
    )


def evaluate_pair(reference, prediction):
    validate_holds(reference["holds"])
    validate_holds(prediction["holds"])
    official = score_scene(reference, prediction, volumes=True)
    angles, eulers = [], []
    typed = 0
    for pair in official["holds"]["pairs"]:
        first = reference["holds"][pair["active"]]
        second = prediction["holds"][pair["test"]]
        a = np.array([first["orientation"][k] for k in ("roll", "pitch", "yaw")])
        b = np.array([second["orientation"][k] for k in ("roll", "pitch", "yaw")])
        eulers.append(np.degrees(np.arctan2(np.sin(a - b), np.cos(a - b))).tolist())
        angle = float(
            np.degrees(
                (Rotation.from_euler("XYZ", a) * Rotation.from_euler("XYZ", b).inv()).magnitude()
            )
        )
        angles.append(angle)
        pair["rotation_geodesic_deg"] = angle
        pair["reference_type"] = first["type"]
        pair["predicted_type"] = second["type"]
        pair["reference_support"] = "wall" if abs(first["position"]["z"]) < 0.002 else "volume"
        typed += first["type"] == second["type"]
    return dict(
        official=official,
        correctly_typed_matches=typed,
        hold_translation_mm=[p["distance_m"] * 1000 for p in official["holds"]["pairs"]],
        hold_rotation_deg=angles,
        wrapped_euler_deg=eulers,
    )


def summarize(rows):
    hold_pairs = [p for row in rows for p in row["official"]["holds"]["pairs"]]
    volume_pairs = [p for row in rows for p in row["official"]["volumes"]["pairs"]]
    missing = sum(len(r["official"]["holds"]["missing"]) for r in rows)
    extra = sum(len(r["official"]["holds"]["extra"]) for r in rows)
    components = dict(
        position=sum(p["position_score"] for p in hold_pairs),
        orientation=sum(p["orientation_score"] for p in hold_pairs),
        type=sum(p["type_score"] for p in hold_pairs),
        missing=10 * missing,
        extra=10 * extra,
    )
    hold_scores = [r["official"]["holds"]["final_score"] for r in rows]
    all_scores = [r["official"]["final_score"] for r in rows]
    reference_count = sum(r["official"]["holds"]["active_count"] for r in rows)
    prediction_count = sum(r["official"]["holds"]["test_count"] for r in rows)
    typed = sum(r["correctly_typed_matches"] for r in rows)
    return dict(
        scenes=len(rows),
        hold_score_total=sum(hold_scores),
        hold_score_mean=float(np.mean(hold_scores)),
        hold_score_distribution=distribution(hold_scores),
        with_volumes_score_total=sum(all_scores),
        with_volumes_score_mean=float(np.mean(all_scores)),
        reference_holds=reference_count,
        predicted_holds=prediction_count,
        matched_holds=len(hold_pairs),
        correctly_typed_matches=typed,
        type_errors=len(hold_pairs) - typed,
        missing_holds=missing,
        extra_holds=extra,
        position_match_recall=len(hold_pairs) / reference_count,
        position_match_precision=len(hold_pairs) / prediction_count,
        correctly_typed_recall=typed / reference_count,
        translation_mm=distribution([x for r in rows for x in r["hold_translation_mm"]]),
        rotation_geodesic_deg=distribution([x for r in rows for x in r["hold_rotation_deg"]]),
        hold_score_components=components,
        reference_volumes=sum(r["official"]["volumes"]["active_count"] for r in rows),
        predicted_volumes=sum(r["official"]["volumes"]["test_count"] for r in rows),
        matched_volumes=len(volume_pairs),
        missing_volumes=sum(len(r["official"]["volumes"]["missing"]) for r in rows),
        extra_volumes=sum(len(r["official"]["volumes"]["extra"]) for r in rows),
        volume_center_error_mm=distribution([p["distance_m"] * 1000 for p in volume_pairs]),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference-description", required=True)
    args = parser.parse_args()
    rows, archives = [], []
    for category in ("real", "sim"):
        predicted_names = {p.name for p in (args.predictions / category).glob("*.zip")}
        reference_names = {p.name for p in (args.references / category).glob("*.zip")}
        if predicted_names != reference_names:
            raise ValueError(f"Prediction/reference archive sets differ for {category}")
        for predicted_zip in sorted((args.predictions / category).glob("*.zip")):
            reference_zip = args.references / category / predicted_zip.name
            with (
                zipfile.ZipFile(predicted_zip) as prediction,
                zipfile.ZipFile(reference_zip) as reference,
            ):
                names = set(prediction.namelist())
                if names != set(reference.namelist()):
                    raise ValueError(f"Archive members differ: {predicted_zip}")
                json_names = sorted(n for n in names if n.endswith(".json"))
                image_names = sorted(
                    n for n in names if Path(n).suffix.lower() in {".png", ".jpg", ".jpeg"}
                )
                for name in image_names:
                    if prediction.read(name) != reference.read(name):
                        raise ValueError(f"Images differ: {predicted_zip}/{name}")
                for name in json_names:
                    truth = json.loads(reference.read(name))
                    estimate = json.loads(prediction.read(name))
                    rows.append(
                        dict(
                            category=category,
                            archive=predicted_zip.name,
                            member=name,
                            recovered_reference_seed=truth.get("seed"),
                            prediction_json_sha256=sha256(prediction.read(name)),
                            reference_json_sha256=sha256(reference.read(name)),
                            **evaluate_pair(truth, estimate),
                        )
                    )
            archives.append(
                dict(
                    category=category,
                    archive=predicted_zip.name,
                    identical_images=len(image_names),
                    prediction_sha256=sha256(predicted_zip.read_bytes()),
                    reference_sha256=sha256(reference_zip.read_bytes()),
                )
            )
    if not rows:
        raise ValueError("No paired prediction archives found")
    groups = defaultdict(list)
    for row in rows:
        groups[f"{row['category']}/{row['archive']}"].append(row)
    result = dict(
        reference_description=args.reference_description,
        reference_root=str(args.references.resolve()),
        prediction_root=str(args.predictions.resolve()),
        inference_unchanged=True,
        lower_score_is_better=True,
        official_default="holds only",
        geometric_statistics=(
            "Official position-matched pairs only; wrong types remain included. "
            "SO(3) angles are additional diagnostics, not the scorer's wrapped Euler metric."
        ),
        archives=archives,
        overall=summarize(rows),
        groups={name: summarize(group) for name, group in groups.items()},
        by_category={
            category: summarize([r for r in rows if r["category"] == category])
            for category in ("real", "sim")
        },
        worst_scenes=[
            dict(
                category=r["category"],
                archive=r["archive"],
                member=r["member"],
                hold_score=r["official"]["holds"]["final_score"],
            )
            for r in sorted(
                rows, key=lambda r: r["official"]["holds"]["final_score"], reverse=True
            )[:10]
        ],
        cases=rows,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "scores.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with (args.out / "per_scene.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "category",
                "archive",
                "json",
                "reference_seed",
                "holds_score",
                "with_volumes_score",
                "matched_holds",
                "missing_holds",
                "extra_holds",
                "type_errors",
                "mean_position_mm",
                "mean_rotation_deg",
            ]
        )
        for row in rows:
            score = row["official"]
            writer.writerow(
                [
                    row["category"],
                    row["archive"],
                    row["member"],
                    row["recovered_reference_seed"],
                    score["holds"]["final_score"],
                    score["final_score"],
                    score["holds"]["matches"],
                    len(score["holds"]["missing"]),
                    len(score["holds"]["extra"]),
                    score["holds"]["matches"] - row["correctly_typed_matches"],
                    distribution(row["hold_translation_mm"])["mean"],
                    distribution(row["hold_rotation_deg"])["mean"],
                ]
            )
    print(json.dumps({k: result[k] for k in ("overall", "groups", "worst_scenes")}, indent=2))


if __name__ == "__main__":
    main()
