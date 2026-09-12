#!/usr/bin/env python3
"""Run EXIF-normalized inference with a released RFDD-SRD checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from functools import partial
from pathlib import Path

from PIL import Image, ImageOps


CLASS_NAMES = ["Deformed", "Displaced", "Fractured", "Inverted", "Missing", "Normal"]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
TORCHVISION_MODELS = {
    "fasterrcnn_r50_fpn_v2",
    "retinanet_r50_fpn_v2",
    "fcos_r50_fpn",
}
ULTRALYTICS_MODELS = {"rtdetr_l", "yolo11m", "yolov10m"}
RELEASED_THRESHOLDS = {
    "fasterrcnn_r50_fpn_v2": 0.41,
    "retinanet_r50_fpn_v2": 0.33,
    "fcos_r50_fpn": 0.29,
    "rtdetr_l": 0.38,
    "yolo11m": 0.57,
    "yolov10m": 0.20,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB").copy()


def image_paths(root: Path) -> list[Path]:
    paths = sorted(path for path in root.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise RuntimeError(f"No supported images found in {root}")
    return paths


def serialize(image_name: str, boxes, scores, classes, score_threshold: float) -> dict:
    detections = [
        {
            "class_id": int(class_id),
            "class": CLASS_NAMES[int(class_id)],
            "score": float(score),
            "xyxy": [float(value) for value in box],
        }
        for box, score, class_id in zip(boxes, scores, classes)
        if float(score) >= score_threshold and 0 <= int(class_id) < len(CLASS_NAMES)
    ]
    return {"image": image_name, "detections": detections}


def build_torchvision_model(model_name: str, checkpoint: Path, device):
    import torch
    import torch.nn as nn
    from torchvision.models.detection import (
        fcos_resnet50_fpn,
        fasterrcnn_resnet50_fpn_v2,
        retinanet_resnet50_fpn_v2,
    )
    from torchvision.models.detection.fcos import FCOSClassificationHead
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.retinanet import RetinaNetClassificationHead

    if model_name == "fasterrcnn_r50_fpn_v2":
        model = fasterrcnn_resnet50_fpn_v2(
            weights=None,
            weights_backbone=None,
            min_size=1024,
            max_size=1333,
            box_score_thresh=0.001,
            box_nms_thresh=0.70,
            box_detections_per_img=100,
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(CLASS_NAMES) + 1)
    elif model_name == "retinanet_r50_fpn_v2":
        model = retinanet_resnet50_fpn_v2(
            weights=None,
            weights_backbone=None,
            min_size=1024,
            max_size=1333,
            score_thresh=0.001,
            nms_thresh=0.70,
            detections_per_img=100,
        )
        model.head.classification_head = RetinaNetClassificationHead(
            model.backbone.out_channels,
            model.head.classification_head.num_anchors,
            len(CLASS_NAMES) + 1,
            norm_layer=partial(nn.GroupNorm, 32),
        )
    elif model_name == "fcos_r50_fpn":
        model = fcos_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            min_size=1024,
            max_size=1333,
            score_thresh=0.001,
            nms_thresh=0.70,
            detections_per_img=100,
        )
        model.head.classification_head = FCOSClassificationHead(
            model.backbone.out_channels,
            model.head.classification_head.num_anchors,
            len(CLASS_NAMES) + 1,
            norm_layer=partial(nn.GroupNorm, 32),
        )
    else:
        raise ValueError(model_name)

    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def predict_torchvision(
    model_name: str, checkpoint: Path, paths: list[Path], device, score_threshold: float
) -> list[dict]:
    import torch
    import torchvision.transforms.functional as functional

    model = build_torchvision_model(model_name, checkpoint, device)
    rows = []
    with torch.no_grad():
        for path in paths:
            image = functional.to_tensor(load_rgb(path)).to(device)
            output = model([image])[0]
            rows.append(
                serialize(
                    path.name,
                    output["boxes"].detach().cpu().numpy(),
                    output["scores"].detach().cpu().numpy(),
                    output["labels"].detach().cpu().numpy().astype(int) - 1,
                    score_threshold,
                )
            )
    return rows


def predict_ultralytics(
    model_name: str,
    checkpoint: Path,
    paths: list[Path],
    device: str,
    score_threshold: float,
) -> list[dict]:
    from ultralytics import RTDETR, YOLO

    model_class = RTDETR if model_name == "rtdetr_l" else YOLO
    model = model_class(str(checkpoint))
    rows = []
    for path in paths:
        result = model.predict(
            source=load_rgb(path),
            imgsz=1280,
            conf=score_threshold,
            iou=0.70,
            max_det=100,
            device=device,
            verbose=False,
        )[0]
        if result.boxes is None:
            rows.append({"image": path.name, "detections": []})
            continue
        rows.append(
            serialize(
                path.name,
                result.boxes.xyxy.detach().cpu().numpy(),
                result.boxes.conf.detach().cpu().numpy(),
                result.boxes.cls.detach().cpu().numpy().astype(int),
                score_threshold,
            )
        )
    return rows


def main() -> None:
    import torch

    models = sorted(TORCHVISION_MODELS | ULTRALYTICS_MODELS)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=models)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0", help="GPU index understood by Ultralytics, or 'cpu'.")
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="Minimum confidence retained in the output. By default, use the "
        "RFDD-SRD-validation threshold associated with the released checkpoint.",
    )
    args = parser.parse_args()

    score_threshold = (
        RELEASED_THRESHOLDS[args.model]
        if args.score_threshold is None
        else args.score_threshold
    )
    if not 0.0 <= score_threshold <= 1.0:
        parser.error("--score-threshold must be in [0, 1]")

    checkpoint = args.weights.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    paths = image_paths(args.images.resolve())
    if args.model in TORCHVISION_MODELS:
        torch_device = torch.device("cpu" if args.device == "cpu" else f"cuda:{args.device}")
        rows = predict_torchvision(
            args.model, checkpoint, paths, torch_device, score_threshold
        )
    else:
        rows = predict_ultralytics(
            args.model, checkpoint, paths, args.device, score_threshold
        )

    payload = {
        "model": args.model,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha256(checkpoint),
        "class_names": CLASS_NAMES,
        "score_threshold": score_threshold,
        "orientation_policy": "PIL.ImageOps.exif_transpose before inference",
        "images": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} image predictions to {args.output}")


if __name__ == "__main__":
    main()
