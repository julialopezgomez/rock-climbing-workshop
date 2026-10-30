"""Optional isolated SAM 3, landmark and rigid-trunk stages for saved real RGB."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ._paths import WEIGHTS


def load_inputs(root):
    names = ["l_with_rob", "r_with_rob"]
    images = [np.array(Image.open(root / "decoded" / name / "rgb.png")) for name in names]
    cameras = [
        json.loads((root / "decoded" / name / "calibration.json").read_text()) for name in names
    ]
    boxes = json.loads((root / "robot_crops.json").read_text())
    return names, images, cameras, [boxes[name] for name in names]


def sam_masks(root, checkpoint):
    import torch
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    names, images, _, boxes = load_inputs(root)
    model = build_sam3_image_model(checkpoint_path=str(checkpoint), load_from_HF=False)
    processor = Sam3Processor(model, confidence_threshold=0.15)
    masks, records = {}, []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for name, image, (x0, y0, x1, y1) in zip(names, images, boxes, strict=True):
            patch = image[y0:y1, x0:x1]
            state = processor.set_image(Image.fromarray(patch))
            result = processor.set_text_prompt(state=state, prompt="robot")
            scores = result["scores"].detach().cpu().float().numpy()
            values = result["masks"].detach().cpu().numpy().reshape(-1, *patch.shape[:2])
            if not len(scores) or scores.max() < 0.5:
                raise ValueError(f"SAM 3 supplies no robot mask above 0.5 for {name}")
            mask = np.zeros(image.shape[:2], bool)
            mask[y0:y1, x0:x1] = values[np.argmax(scores)]
            masks[name] = mask
            records.append(dict(recording=name, score=float(scores.max()), pixels=int(mask.sum())))
            canvas = image.copy()
            canvas[mask] = (0.55 * canvas[mask] + 0.45 * np.array([0, 230, 90])).astype(np.uint8)
            Image.fromarray(canvas).save(root / f"{name}_segmentation.jpg")
    np.savez_compressed(root / "robot_masks.npz", **masks)
    return dict(
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        prompt="robot",
        records=records,
        source="RGB only; no sensor depth, encoder states or labels",
    )


def keypoint_probe(root):
    from ultralytics import YOLO

    names, images, _, boxes = load_inputs(root)
    weights = WEIGHTS / "body/best.pt"
    model = YOLO(str(weights))
    rows = []
    for name, image, box in zip(names, images, boxes, strict=True):
        x0, y0, x1, y1 = box
        variants = [("full", image, np.zeros(2), 1280)] + [
            ("foreground", image[y0:y1, x0:x1], np.array([x0, y0]), size)
            for size in [384, 512, 640]
        ]
        for label, patch, origin, size in variants:
            result = model.predict(
                [np.ascontiguousarray(patch[..., ::-1])],
                imgsz=size,
                conf=0.01,
                device=0,
                verbose=False,
            )[0]
            observed = []
            for index, confidence in enumerate(result.boxes.conf.cpu().numpy()):
                pixels = result.keypoints.xy[index].cpu().numpy() + origin
                observed.append(
                    dict(
                        confidence=float(confidence),
                        pixels=pixels.tolist(),
                        keypoint_confidence=result.keypoints.conf[index].cpu().numpy().tolist(),
                    )
                )
            rows.append(dict(recording=name, variant=label, resolution=size, detections=observed))
    # The observation frontend must transfer to both cameras before it supplies
    # the multiview initializer. High keypoint logits alone do not establish this.
    supported = [
        name
        for name in names
        if any(
            row["recording"] == name and any(d["confidence"] >= 0.1 for d in row["detections"])
            for row in rows
        )
    ]
    return dict(
        multiview_frontend_supported=len(supported) == 2,
        supported_views=supported,
        weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
        trials=rows,
        root_pose_published=False,
        note=(
            "Detection confidence is uncalibrated on real images; "
            "geometric verification is still required"
        ),
    )


def trunk_fit(root):
    from .physical_trunk import fit_trunk

    names, images, cameras, _ = load_inputs(root)
    label = json.loads((root / "label_hypothesis.json").read_text())
    with np.load(root / "robot_masks.npz") as archive:
        masks = [archive[name] for name in names]
    result, overlays = fit_trunk(images, cameras, masks, label, root / "cache")
    for name, image in zip(names, overlays, strict=True):
        Image.fromarray(image).save(root / f"{name}_trunk.jpg")
        corners = label["outer_corners"][names.index(name)]
        x, y = np.round(np.mean(corners, axis=0)).astype(int)
        detail = Image.fromarray(image).crop((x - 80, y - 100, x + 80, y + 120))
        detail.resize((320, 440)).save(root / f"{name}_trunk_detail.jpg")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["sam", "keypoints", "trunk"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    try:
        if args.stage == "sam":
            if args.checkpoint is None:
                raise ValueError("SAM 3 requires an explicit local checkpoint")
            result = sam_masks(args.out, args.checkpoint)
        elif args.stage == "keypoints":
            result = keypoint_probe(args.out)
        else:
            result = trunk_fit(args.out)
        result["completed"] = True
    except Exception as error:
        result = dict(completed=False, error=f"{type(error).__name__}: {error}")
    (args.out / f"{args.stage}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(result)
        if not result.get("completed")
        else json.dumps(dict(stage=args.stage, completed=True))
    )
    return 0 if result["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
