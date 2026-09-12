#!/usr/bin/env python3
"""Train one RFDD-SRD group-CV detector and save validation/test predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


CV_ROOT = Path("artifacts/cv")
CLASS_NAMES = ["Deformed", "Displaced", "Fractured", "Inverted", "Missing", "Normal"]
GENERIC_WEIGHTS: dict[str, str | Path] = {
    "yolo11m": "yolo11m.pt",
    "yolov10m": "yolov10m.pt",
    "rtdetr_l": "rtdetr-l.pt",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_paths(path: Path) -> list[Path]:
    return [Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_rgb(path: Path) -> Image.Image:
    """Load pixels in the orientation used by the annotation coordinates."""
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB").copy()


def label_path_for_image(path: Path) -> Path:
    if path.parent.name != "images":
        raise ValueError(f"Expected an images/ directory, got {path}")
    return path.parent.parent / "labels" / f"{path.stem}.txt"


def serialize_ultralytics_predictions(results, source_paths: list[Path] | None = None) -> list[dict]:
    rows = []
    for index, result in enumerate(results):
        detections = []
        if result.boxes is not None:
            xyxy = result.boxes.xyxy.detach().cpu().numpy()
            scores = result.boxes.conf.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            for box, score, class_id in zip(xyxy, scores, classes):
                detections.append(
                    {
                        "class_id": int(class_id),
                        "score": float(score),
                        "xyxy": [float(value) for value in box],
                    }
                )
        image_name = source_paths[index].name if source_paths is not None else Path(result.path).name
        rows.append({"image": image_name, "detections": detections})
    if source_paths is not None and len(rows) != len(source_paths):
        raise RuntimeError(f"Prediction count mismatch: expected {len(source_paths)}, got {len(rows)}")
    return rows


def run_ultralytics(model_name: str, fold: int, seed: int, device: int, run_dir: Path) -> dict:
    import torch
    import ultralytics
    from ultralytics import RTDETR, YOLO

    weight_path = GENERIC_WEIGHTS[model_name]
    model_class = RTDETR if model_name == "rtdetr_l" else YOLO
    model = model_class(str(weight_path))
    fold_dir = CV_ROOT / "splits" / f"fold_{fold}"
    batch = 2 if model_name == "rtdetr_l" else 4
    start = time.time()
    model.train(
        data=str(fold_dir / "data.yaml"),
        epochs=100,
        patience=20,
        imgsz=1280,
        batch=batch,
        device=device,
        workers=4,
        project=str(run_dir.parent),
        name=run_dir.name,
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        weight_decay=0.0005,
        seed=seed,
        deterministic=True,
        amp=True,
        close_mosaic=10,
        plots=False,
        verbose=False,
    )
    best_path = Path(model.trainer.best).resolve()
    best_model = model_class(str(best_path))
    prediction_payloads = {}
    inference_seconds = {}
    for role in ("val", "test"):
        paths = load_paths(fold_dir / f"{role}.txt")
        infer_start = time.time()
        results = best_model.predict(
            source=[str(path) for path in paths],
            imgsz=1280,
            conf=0.001,
            iou=0.70,
            max_det=100,
            device=device,
            stream=True,
            verbose=False,
        )
        prediction_payloads[role] = serialize_ultralytics_predictions(results, paths)
        inference_seconds[role] = time.time() - infer_start

    metadata = {
        "model": model_name,
        "model_family": "RT-DETR" if model_name == "rtdetr_l" else "YOLO",
        "generic_pretraining": "COCO",
        "rfdd_srd_training_data_used": True,
        "fold": fold,
        "seed": seed,
        "device": device,
        "image_size": 1280,
        "epochs_requested": 100,
        "epochs_completed": int(model.trainer.epoch + 1),
        "best_checkpoint": str(best_path),
        "generic_weight": str(weight_path),
        "generic_weight_sha256": sha256(Path(weight_path)) if Path(weight_path).is_file() else None,
        "best_weight_sha256": sha256(best_path),
        "train_seconds": time.time() - start,
        "inference_seconds": inference_seconds,
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "cuda": torch.version.cuda,
        },
    }
    for role, images in prediction_payloads.items():
        payload = {**metadata, "split": role, "images": images}
        (run_dir / f"predictions_{role}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
    return metadata


class DetectionDataset:
    def __init__(self, list_path: Path):
        import torch

        self.torch = torch
        self.paths = load_paths(list_path)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        import torchvision.transforms.functional as functional

        path = self.paths[index]
        image = load_rgb(path)
        width, height = image.size
        boxes = []
        labels = []
        label_path = label_path_for_image(path)
        for line in label_path.read_text(encoding="utf-8").splitlines():
            class_id, cx, cy, box_width, box_height = map(float, line.split())
            boxes.append(
                [
                    (cx - box_width / 2) * width,
                    (cy - box_height / 2) * height,
                    (cx + box_width / 2) * width,
                    (cy + box_height / 2) * height,
                ]
            )
            labels.append(int(class_id) + 1)  # torchvision reserves 0 for background
        target = {
            "boxes": self.torch.as_tensor(boxes, dtype=self.torch.float32),
            "labels": self.torch.as_tensor(labels, dtype=self.torch.int64),
            "image_id": self.torch.tensor([index], dtype=self.torch.int64),
        }
        return functional.to_tensor(image), target, path.name


def collate(batch):
    return tuple(zip(*batch))


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.empty(0, dtype=float)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area_b = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return intersection / np.maximum(area_a + area_b - intersection, 1e-12)


def average_precision(scores: list[float], flags: list[int], n_gt: int) -> float:
    if n_gt == 0 or not scores:
        return 0.0
    score_array = np.asarray(scores)
    flag_array = np.asarray(flags)
    order = np.argsort(-score_array, kind="mergesort")
    true_positive = flag_array[order].astype(float)
    false_positive = 1.0 - true_positive
    recall = np.cumsum(true_positive) / n_gt
    precision = np.cumsum(true_positive) / np.maximum(
        np.cumsum(true_positive) + np.cumsum(false_positive), 1e-12
    )
    values = [precision[recall >= point].max() if np.any(recall >= point) else 0.0 for point in np.linspace(0, 1, 101)]
    return float(np.mean(values))


def ground_truth_for_dataset(dataset: DetectionDataset) -> dict[str, list[dict]]:
    result = {}
    for path in dataset.paths:
        with Image.open(path) as image:
            width, height = image.size
        label_path = label_path_for_image(path)
        records = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            class_id, cx, cy, box_width, box_height = map(float, line.split())
            records.append(
                {
                    "class_id": int(class_id),
                    "xyxy": np.asarray(
                        [
                            (cx - box_width / 2) * width,
                            (cy - box_height / 2) * height,
                            (cx + box_width / 2) * width,
                            (cy + box_height / 2) * height,
                        ]
                    ),
                }
            )
        result[path.name] = records
    return result


def predict_fasterrcnn(model, loader, device: int) -> list[dict]:
    import torch

    model.eval()
    rows = []
    with torch.no_grad():
        for images, _targets, names in loader:
            outputs = model([image.to(device) for image in images])
            for name, output in zip(names, outputs):
                boxes = output["boxes"].detach().cpu().numpy()
                scores = output["scores"].detach().cpu().numpy()
                labels = output["labels"].detach().cpu().numpy().astype(int) - 1
                detections = [
                    {
                        "class_id": int(class_id),
                        "score": float(score),
                        "xyxy": [float(value) for value in box],
                    }
                    for box, score, class_id in zip(boxes, scores, labels)
                    if 0 <= int(class_id) < len(CLASS_NAMES)
                ]
                rows.append({"image": name, "detections": detections})
    return rows


def macro_ap50(ground_truth: dict[str, list[dict]], predictions: list[dict]) -> float:
    by_image = {row["image"]: row["detections"] for row in predictions}
    class_values = []
    for class_id in range(len(CLASS_NAMES)):
        scores = []
        flags = []
        n_gt = 0
        for image_name, image_ground_truth in ground_truth.items():
            gt_boxes = np.asarray(
                [record["xyxy"] for record in image_ground_truth if record["class_id"] == class_id]
            ).reshape(-1, 4)
            n_gt += len(gt_boxes)
            matched = np.zeros(len(gt_boxes), dtype=bool)
            detections = sorted(
                [record for record in by_image.get(image_name, []) if record["class_id"] == class_id],
                key=lambda record: record["score"],
                reverse=True,
            )
            for record in detections:
                is_true_positive = 0
                ious = box_iou(np.asarray(record["xyxy"]), gt_boxes)
                for gt_index in np.argsort(-ious):
                    if ious[gt_index] < 0.5:
                        break
                    if not matched[gt_index]:
                        matched[gt_index] = True
                        is_true_positive = 1
                        break
                scores.append(record["score"])
                flags.append(is_true_positive)
        class_values.append(average_precision(scores, flags, n_gt))
    return float(np.mean(class_values))


def run_torchvision_detector(model_name: str, fold: int, seed: int, device: int, run_dir: Path) -> dict:
    from functools import partial

    import torch
    import torch.nn as nn
    import torchvision
    from torch.utils.data import DataLoader
    from torchvision.models.detection import (
        FCOS_ResNet50_FPN_Weights,
        FasterRCNN_ResNet50_FPN_V2_Weights,
        RetinaNet_ResNet50_FPN_V2_Weights,
        fcos_resnet50_fpn,
        fasterrcnn_resnet50_fpn_v2,
        retinanet_resnet50_fpn_v2,
    )
    from torchvision.models.detection.fcos import FCOSClassificationHead
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.retinanet import RetinaNetClassificationHead

    fold_dir = CV_ROOT / "splits" / f"fold_{fold}"
    train_dataset = DetectionDataset(fold_dir / "train.txt")
    val_dataset = DetectionDataset(fold_dir / "val.txt")
    test_dataset = DetectionDataset(fold_dir / "test.txt")
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        collate_fn=collate,
        generator=generator,
        persistent_workers=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=4, collate_fn=collate)

    if model_name == "fasterrcnn_r50_fpn_v2":
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn_v2(
            weights=weights,
            min_size=1024,
            max_size=1333,
            box_score_thresh=0.001,
            box_nms_thresh=0.70,
            box_detections_per_img=100,
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(CLASS_NAMES) + 1)
        model_family = "Faster R-CNN"
    elif model_name == "retinanet_r50_fpn_v2":
        weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
        model = retinanet_resnet50_fpn_v2(
            weights=weights,
            min_size=1024,
            max_size=1333,
            score_thresh=0.001,
            nms_thresh=0.70,
            detections_per_img=100,
        )
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = RetinaNetClassificationHead(
            model.backbone.out_channels,
            num_anchors,
            len(CLASS_NAMES) + 1,
            norm_layer=partial(nn.GroupNorm, 32),
        )
        model_family = "RetinaNet"
    elif model_name == "fcos_r50_fpn":
        weights = FCOS_ResNet50_FPN_Weights.DEFAULT
        model = fcos_resnet50_fpn(
            weights=weights,
            min_size=1024,
            max_size=1333,
            score_thresh=0.001,
            nms_thresh=0.70,
            detections_per_img=100,
        )
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = FCOSClassificationHead(
            model.backbone.out_channels,
            num_anchors,
            len(CLASS_NAMES) + 1,
            norm_layer=partial(nn.GroupNorm, 32),
        )
        model_family = "FCOS"
    else:
        raise ValueError(f"Unsupported Torchvision detector: {model_name}")
    model.to(device)

    learning_rate = 0.001 if model_name == "fcos_r50_fpn" else 0.005
    use_amp = model_name != "fcos_r50_fpn"
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        momentum=0.9,
        weight_decay=0.0005,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[12, 20], gamma=0.2)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_ap = -1.0
    best_epoch = -1
    stale_epochs = 0
    history = []
    start = time.time()
    val_gt = ground_truth_for_dataset(val_dataset)
    best_path = run_dir / "best.pt"

    for epoch in range(30):
        model.train()
        epoch_losses = []
        for images, targets, _names in train_loader:
            images = [image.to(device) for image in images]
            targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                losses = model(images, targets)
                loss = sum(losses.values())
            if not torch.isfinite(loss):
                details = {name: float(value.detach().cpu()) for name, value in losses.items()}
                raise RuntimeError(f"Non-finite training loss for {model_name}: {details}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(float(loss.detach().cpu()))
        scheduler.step()

        val_predictions = predict_fasterrcnn(model, val_loader, device)
        val_ap = macro_ap50(val_gt, val_predictions)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(epoch_losses)),
                "val_macro_ap50": val_ap,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            f"{model_name} fold={fold} seed={seed} epoch={epoch + 1} "
            f"loss={history[-1]['train_loss']:.4f} val_ap50={val_ap:.4f}",
            flush=True,
        )
        if val_ap > best_ap + 1e-6:
            best_ap = val_ap
            best_epoch = epoch + 1
            stale_epochs = 0
            torch.save(model.state_dict(), best_path)
        else:
            stale_epochs += 1
        if epoch + 1 >= 12 and stale_epochs >= 8:
            break

    model.load_state_dict(
        torch.load(best_path, map_location=f"cuda:{device}", weights_only=True)
    )
    prediction_payloads = {}
    inference_seconds = {}
    for role, loader in (("val", val_loader), ("test", test_loader)):
        infer_start = time.time()
        prediction_payloads[role] = predict_fasterrcnn(model, loader, device)
        inference_seconds[role] = time.time() - infer_start

    checkpoint_dir = Path(torch.hub.get_dir()) / "checkpoints"
    generic_path = checkpoint_dir / Path(weights.url).name
    if not generic_path.is_file():
        generic_path = None
    metadata = {
        "model": model_name,
        "model_family": model_family,
        "generic_pretraining": "COCO",
        "rfdd_srd_training_data_used": True,
        "fold": fold,
        "seed": seed,
        "device": device,
        "image_min_size": 1024,
        "image_max_size": 1333,
        "learning_rate": learning_rate,
        "mixed_precision": use_amp,
        "epochs_requested": 30,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_macro_ap50": best_ap,
        "best_checkpoint": str(best_path),
        "generic_weight": str(generic_path) if generic_path else weights.url,
        "generic_weight_sha256": sha256(generic_path) if generic_path else None,
        "best_weight_sha256": sha256(best_path),
        "train_seconds": time.time() - start,
        "inference_seconds": inference_seconds,
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda": torch.version.cuda,
        },
        "history": history,
    }
    for role, images in prediction_payloads.items():
        payload = {**metadata, "split": role, "images": images}
        (run_dir / f"predictions_{role}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
    return metadata


def main() -> None:
    global CV_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=CV_ROOT,
        help="Experiment root containing splits/, runs/, and dataset/.",
    )
    parser.add_argument(
        "--model",
        choices=[
            "yolo11m",
            "yolov10m",
            "rtdetr_l",
            "fasterrcnn_r50_fpn_v2",
            "retinanet_r50_fpn_v2",
            "fcos_r50_fpn",
        ],
        required=True,
    )
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument(
        "--pretrained-dir",
        type=Path,
        default=None,
        help="Optional directory containing yolo11m.pt, yolov10m.pt, and rtdetr-l.pt.",
    )
    args = parser.parse_args()

    CV_ROOT = args.root.resolve()
    if args.pretrained_dir is not None:
        pretrained_dir = args.pretrained_dir.resolve()
        for model_name, filename in {
            "yolo11m": "yolo11m.pt",
            "yolov10m": "yolov10m.pt",
            "rtdetr_l": "rtdetr-l.pt",
        }.items():
            GENERIC_WEIGHTS[model_name] = pretrained_dir / filename

    run_dir = CV_ROOT / "runs" / args.model / f"fold_{args.fold}_seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    completed_path = run_dir / "completed.json"
    if completed_path.is_file():
        print(f"Already complete: {run_dir}")
        return

    set_seed(args.seed)
    if args.model in {"yolo11m", "yolov10m", "rtdetr_l"}:
        metadata = run_ultralytics(args.model, args.fold, args.seed, args.device, run_dir)
    else:
        metadata = run_torchvision_detector(args.model, args.fold, args.seed, args.device, run_dir)
    completed_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Completed {args.model} fold={args.fold} seed={args.seed}")


if __name__ == "__main__":
    main()
