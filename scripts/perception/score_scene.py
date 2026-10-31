#!/usr/bin/env python3
"""Score a saved prediction with the pinned upstream comparison formula."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from rcr.perception.challenge.official_score import score_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--reference", type=Path)
    source.add_argument("--dataset", type=Path)
    parser.add_argument("--prediction", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument(
        "--volumes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include volumes; the current upstream default scores holds only",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.dataset:
        if not args.predictions or args.prediction:
            parser.error("--dataset requires --predictions DIRECTORY")
        cases = json.loads((args.dataset / "manifest.json").read_text())["cases"]
        if not cases:
            parser.error("The dataset is empty")
        rows = []
        for case in cases:
            name = case["id"]
            folder = args.dataset / name
            reference = folder / "expected_scene.json"
            if not reference.exists() and not args.volumes:
                reference = folder / "expected_holds.json"
            row = score_scene(
                json.loads(reference.read_text()),
                json.loads((args.predictions / name / "prediction.json").read_text()),
                volumes=args.volumes,
            )
            rows.append(dict(case=name, **row))
        total = sum(row["final_score"] for row in rows)
        result = dict(
            source_commit=rows[0]["source_commit"],
            volume_scoring=args.volumes,
            cases=len(rows),
            final_score=total,
            mean_score=total / len(rows),
            rows=rows,
        )
    else:
        if not args.prediction or args.predictions:
            parser.error("--reference requires --prediction FILE")
        result = score_scene(
            json.loads(args.reference.read_text()),
            json.loads(args.prediction.read_text()),
            volumes=args.volumes,
        )
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
