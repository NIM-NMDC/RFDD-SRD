#!/usr/bin/env python3
"""Frozen-checkpoint evaluation of RFDD-SRD detectors on an external real dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import time
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


CLASS_NAMES = ["Deformed", "Displaced", "Fractured", "Inverted", "Missing", "Normal"]
DISPLAY_MODELS = {
    "fasterrcnn_r50_fpn_v2": "Faster R-CNN R50-FPN V2",
    "retinanet_r50_fpn_v2": "RetinaNet R50-FPN V2",
    "fcos_r50_fpn": "FCOS R50-FPN",
    "rtdetr_l": "RT-DETR-L",
    "yolo11m": "YOLO11m",
    "yolov10m": "YOLOv10m",
}
SEEDS = [20260908, 20260909, 20260910]
IOU_THRESHOLDS = [round(value, 2) for value in np.arange(0.50, 0.951, 0.05)]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
METRICS = [
    "precision",
    "recall",
    "f1",
    "ap50",
    "ap50_95",
    "miss_rate",
    "false_discovery_rate",
    "fppi",
]
BOOTSTRAP_METRICS = [
    "precision",
    "recall",
    "f1",
    "ap50",
    "miss_rate",
    "false_discovery_rate",
    "fppi",
]


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_paths(real_root: Path) -> list[Path]:
    return sorted(
        path for path in (real_root / "images").iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )


def load_rgb(path: Path) -> Image.Image:
    """Load every external image in a single EXIF-normalized coordinate frame."""
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB").copy()


def label_path(real_root: Path, image_path: Path) -> Path:
    return real_root / "labels" / f"{image_path.stem}.txt"


def load_ground_truth(real_root: Path) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for path in image_paths(real_root):
        width, height = load_rgb(path).size
        records = []
        current_label_path = label_path(real_root, path)
        if not current_label_path.is_file():
            raise FileNotFoundError(current_label_path)
        for line_no, line in enumerate(current_label_path.read_text(encoding="utf-8").splitlines(), 1):
            values = line.split()
            if len(values) != 5:
                raise ValueError(f"Malformed label {current_label_path}:{line_no}")
            class_id, cx, cy, box_width, box_height = map(float, values)
            class_id = int(class_id)
            if not 0 <= class_id < len(CLASS_NAMES):
                raise ValueError(f"Invalid class {class_id} in {current_label_path}:{line_no}")
            if not all(0.0 <= value <= 1.0 for value in (cx, cy, box_width, box_height)):
                raise ValueError(f"Out-of-range box in {current_label_path}:{line_no}")
            records.append(
                {
                    "class_id": class_id,
                    "xyxy": np.asarray(
                        [
                            (cx - box_width / 2) * width,
                            (cy - box_height / 2) * height,
                            (cx + box_width / 2) * width,
                            (cy + box_height / 2) * height,
                        ],
                        dtype=float,
                    ),
                }
            )
        result[path.name] = records
    return result


def audit_real_dataset(real_root: Path, ground_truth: dict[str, list[dict]]) -> dict:
    class_counts = Counter()
    image_dimensions = Counter()
    boxes_per_image = Counter()
    for path in image_paths(real_root):
        image = load_rgb(path)
        image_dimensions[f"{image.width}x{image.height}"] += 1
        records = ground_truth[path.name]
        boxes_per_image[len(records)] += 1
        class_counts.update(record["class_id"] for record in records)
    classes_path = real_root / "classes.txt"
    if classes_path.is_file():
        listed_classes = [
            line.strip()
            for line in classes_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if listed_classes != CLASS_NAMES:
            raise ValueError(f"Class order mismatch: {listed_classes}")
    damage_containing_images = sum(
        any(int(record["class_id"]) < 5 for record in records)
        for records in ground_truth.values()
    )
    return {
        "images": len(ground_truth),
        "boxes": sum(len(records) for records in ground_truth.values()),
        "damage_containing_images": damage_containing_images,
        "normal_only_images": len(ground_truth) - damage_containing_images,
        "class_counts": {CLASS_NAMES[index]: class_counts[index] for index in range(len(CLASS_NAMES))},
        "dimensions": dict(image_dimensions),
        "boxes_per_image": dict(boxes_per_image),
    }


def build_source_groups(
    ground_truth: dict[str, list[dict]],
) -> tuple[dict[str, list[str]], dict[str, str], list[dict]]:
    """Build conservative acquisition groups from identifiers embedded in filenames."""
    descriptors = {
        image_name: Path(image_name).stem.split("-", 1)[-1] for image_name in ground_truth
    }
    descriptor_values = set(descriptors.values())
    images_by_group: dict[str, list[str]] = defaultdict(list)
    for image_name, descriptor in descriptors.items():
        normal_match = re.search(r"正常(\d+)$", descriptor)
        if normal_match:
            group = f"normal_pair_{int(normal_match.group(1)):02d}"
        else:
            normalized = descriptor
            if descriptor.endswith("2") and descriptor[:-1] in descriptor_values:
                normalized = descriptor[:-1]
            group = f"defect_view_{normalized}"
        images_by_group[group].append(image_name)

    group_stratum: dict[str, str] = {}
    manifest_rows = []
    for group, names in sorted(images_by_group.items()):
        defect_classes = sorted(
            {
                int(record["class_id"])
                for image_name in names
                for record in ground_truth[image_name]
                if int(record["class_id"]) < 5
            }
        )
        if len(defect_classes) > 1:
            raise ValueError(f"Source group contains multiple defect classes: {group} -> {defect_classes}")
        stratum = CLASS_NAMES[defect_classes[0]] if defect_classes else "Normal-only"
        group_stratum[group] = stratum
        for image_name in sorted(names):
            manifest_rows.append(
                {"image": image_name, "source_group": group, "bootstrap_stratum": stratum}
            )
    return dict(images_by_group), group_stratum, manifest_rows


class RealDetectionDataset:
    def __init__(self, real_root: Path):
        import torch

        self.torch = torch
        self.paths = image_paths(real_root)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        import torchvision.transforms.functional as functional

        path = self.paths[index]
        image = load_rgb(path)
        return functional.to_tensor(image), path.name


def collate_images(batch):
    return tuple(zip(*batch))


def build_torchvision_model(model_name: str, device: int):
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
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = RetinaNetClassificationHead(
            model.backbone.out_channels,
            num_anchors,
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
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = FCOSClassificationHead(
            model.backbone.out_channels,
            num_anchors,
            len(CLASS_NAMES) + 1,
            norm_layer=partial(nn.GroupNorm, 32),
        )
    else:
        raise ValueError(model_name)
    return model.to(device)


def predict_torchvision(model, dataset: RealDetectionDataset, device: int) -> list[dict]:
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_images,
        persistent_workers=True,
    )
    model.eval()
    rows = []
    with torch.inference_mode():
        for images, names in loader:
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


def predict_ultralytics(model_name: str, checkpoint: Path, paths: list[Path], device: int) -> list[dict]:
    from ultralytics import RTDETR, YOLO

    model_class = RTDETR if model_name == "rtdetr_l" else YOLO
    model = model_class(str(checkpoint))
    rows = []
    for path in paths:
        prediction = model.predict(
            source=load_rgb(path),
            imgsz=1280,
            batch=1,
            conf=0.001,
            iou=0.70,
            max_det=100,
            device=device,
            stream=False,
            verbose=False,
        )
        if len(prediction) != 1:
            raise RuntimeError(f"Prediction count mismatch for {path}: {len(prediction)}")
        result = prediction[0]
        detections = []
        if result.boxes is not None:
            xyxy = result.boxes.xyxy.detach().cpu().numpy()
            scores = result.boxes.conf.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            detections = [
                {
                    "class_id": int(class_id),
                    "score": float(score),
                    "xyxy": [float(value) for value in box],
                }
                for box, score, class_id in zip(xyxy, scores, classes)
                if 0 <= int(class_id) < len(CLASS_NAMES)
            ]
        rows.append({"image": path.name, "detections": detections})
    return rows


def checkpoint_path(root: Path, model: str, fold: int, seed: int) -> Path:
    run_dir = root / "runs" / model / f"fold_{fold}_seed_{seed}"
    if model in {"rtdetr_l", "yolo11m", "yolov10m"}:
        return run_dir / "weights" / "best.pt"
    return run_dir / "best.pt"


def prediction_path(root: Path, model: str, fold: int, seed: int) -> Path:
    return root / "external_real" / "predictions" / model / f"fold_{fold}_seed_{seed}.json"


def infer_runs(args: argparse.Namespace) -> None:
    import torch

    root = args.root.resolve()
    real_root = args.real_root.resolve()
    ground_truth = load_ground_truth(real_root)
    audit = audit_real_dataset(real_root, ground_truth)
    dataset = RealDetectionDataset(real_root)
    paths = image_paths(real_root)
    for model_name in args.models:
        for seed in args.seeds:
            for fold in args.folds:
                output_path = prediction_path(root, model_name, fold, seed)
                if output_path.is_file() and not args.overwrite:
                    print(f"SKIP {model_name} fold={fold} seed={seed}", flush=True)
                    continue
                current_checkpoint = checkpoint_path(root, model_name, fold, seed)
                if not current_checkpoint.is_file():
                    raise FileNotFoundError(current_checkpoint)
                start = time.time()
                if model_name in {"rtdetr_l", "yolo11m", "yolov10m"}:
                    predictions = predict_ultralytics(
                        model_name, current_checkpoint, paths, args.device
                    )
                else:
                    model = build_torchvision_model(model_name, args.device)
                    state_dict = torch.load(
                        current_checkpoint,
                        map_location=f"cuda:{args.device}",
                        weights_only=True,
                    )
                    model.load_state_dict(state_dict, strict=True)
                    predictions = predict_torchvision(model, dataset, args.device)
                    del model, state_dict
                    torch.cuda.empty_cache()
                output_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "dataset": "independent real inspection set",
                    "dataset_audit": audit,
                    "model": model_name,
                    "model_label": DISPLAY_MODELS[model_name],
                    "fold": fold,
                    "seed": seed,
                    "checkpoint": str(current_checkpoint),
                    "checkpoint_sha256": sha256(current_checkpoint),
                    "inference_seconds": time.time() - start,
                    "images": predictions,
                }
                output_path.write_text(
                    json.dumps(payload, ensure_ascii=False), encoding="utf-8"
                )
                print(
                    f"DONE {model_name} fold={fold} seed={seed} "
                    f"seconds={payload['inference_seconds']:.1f}",
                    flush=True,
                )


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


def image_match_stats(
    ground_truth: dict[str, list[dict]],
    predictions: dict[str, list[dict]],
    class_id: int,
    iou_threshold: float,
) -> dict[str, dict]:
    result = {}
    for image_name, image_ground_truth in ground_truth.items():
        gt_boxes = np.asarray(
            [record["xyxy"] for record in image_ground_truth if int(record["class_id"]) == class_id],
            dtype=float,
        ).reshape(-1, 4)
        image_predictions = sorted(
            [record for record in predictions.get(image_name, []) if int(record["class_id"]) == class_id],
            key=lambda record: float(record["score"]),
            reverse=True,
        )
        matched = np.zeros(len(gt_boxes), dtype=bool)
        scores = np.asarray([float(record["score"]) for record in image_predictions], dtype=float)
        true_positive = np.zeros(len(image_predictions), dtype=np.int8)
        for index, record in enumerate(image_predictions):
            ious = box_iou(np.asarray(record["xyxy"], dtype=float), gt_boxes)
            for gt_index in np.argsort(-ious):
                if ious[gt_index] < iou_threshold:
                    break
                if not matched[gt_index]:
                    matched[gt_index] = True
                    true_positive[index] = 1
                    break
        result[image_name] = {"scores": scores, "tp": true_positive, "n_gt": len(gt_boxes)}
    return result


def average_precision(scores: np.ndarray, true_positive: np.ndarray, n_gt: int) -> float:
    if n_gt == 0 or scores.size == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    tp = true_positive[order].astype(float)
    fp = 1.0 - tp
    recall = np.cumsum(tp) / n_gt
    precision = np.cumsum(tp) / np.maximum(np.cumsum(tp) + np.cumsum(fp), 1e-12)
    precision_envelope = np.maximum.accumulate(precision[::-1])[::-1]
    positions = np.searchsorted(recall, np.linspace(0.0, 1.0, 101), side="left")
    valid = positions < len(precision_envelope)
    interpolated = np.zeros(101, dtype=float)
    interpolated[valid] = precision_envelope[positions[valid]]
    return float(interpolated.mean())


def aggregate(stats: dict[str, dict], image_keys: list[str], threshold: float | None) -> dict:
    score_parts = []
    tp_parts = []
    selected_tp = 0
    selected_count = 0
    n_gt = 0
    for image_name in image_keys:
        item = stats[image_name]
        scores = item["scores"]
        true_positive = item["tp"]
        score_parts.append(scores)
        tp_parts.append(true_positive)
        n_gt += int(item["n_gt"])
        if threshold is not None:
            selected = scores >= threshold
            selected_tp += int(true_positive[selected].sum())
            selected_count += int(selected.sum())
    scores = np.concatenate(score_parts) if score_parts else np.empty(0, dtype=float)
    true_positive = np.concatenate(tp_parts) if tp_parts else np.empty(0, dtype=np.int8)
    tp = selected_tp
    fp = selected_count - tp
    fn = n_gt - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n_gt": n_gt,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ap": average_precision(scores, true_positive, n_gt),
        "miss_rate": 1.0 - recall,
        "false_discovery_rate": 1.0 - precision if tp + fp else float("nan"),
        "fppi": fp / len(image_keys) if image_keys else 0.0,
    }


def class_metrics(
    stats_by_iou: dict[float, dict[str, dict]], image_keys: list[str], threshold: float
) -> dict[str, float]:
    at_50 = aggregate(stats_by_iou[0.50], image_keys, threshold)
    aps = [aggregate(stats_by_iou[iou], image_keys, None)["ap"] for iou in IOU_THRESHOLDS]
    return {
        "n_gt": at_50["n_gt"],
        "tp": at_50["tp"],
        "fp": at_50["fp"],
        "fn": at_50["fn"],
        "precision": at_50["precision"],
        "recall": at_50["recall"],
        "f1": at_50["f1"],
        "ap50": aps[0],
        "ap50_95": float(np.mean(aps)),
        "miss_rate": at_50["miss_rate"],
        "false_discovery_rate": at_50["false_discovery_rate"],
        "fppi": at_50["fppi"],
    }


def collapse_binary(records: dict[str, list[dict]]) -> dict[str, list[dict]]:
    result = {}
    for image_name, image_records in records.items():
        result[image_name] = [
            {**record, "class_id": 0 if int(record["class_id"]) < 5 else 1}
            for record in image_records
        ]
    return result


def load_predictions(path: Path) -> tuple[dict[str, list[dict]], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {row["image"]: row["detections"] for row in payload["images"]}, payload


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if np.all(np.isnan(array)):
        return float("nan"), float("nan")
    return float(np.nanmean(array)), float(np.nanstd(array, ddof=1))


def percentile_ci(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if np.all(np.isnan(array)):
        return float("nan"), float("nan")
    low, high = np.nanpercentile(array, [2.5, 97.5])
    return float(low), float(high)


_BOOTSTRAP_COUNTS: np.ndarray | None = None
_STATS: dict = {}
_THRESHOLDS: dict = {}
_PREPARED: dict = {}


def prepare_weighted_bootstrap(
    stats_at_50: dict[str, dict], image_order: list[str], threshold: float
) -> dict[str, np.ndarray]:
    n_gt = np.asarray([stats_at_50[name]["n_gt"] for name in image_order], dtype=float)
    selected_tp = np.zeros(len(image_order), dtype=float)
    selected_fp = np.zeros(len(image_order), dtype=float)
    score_parts = []
    tp_parts = []
    image_index_parts = []
    for image_index, name in enumerate(image_order):
        scores = stats_at_50[name]["scores"]
        true_positive = stats_at_50[name]["tp"].astype(float)
        selected = scores >= threshold
        selected_tp[image_index] = true_positive[selected].sum()
        selected_fp[image_index] = selected.sum() - selected_tp[image_index]
        score_parts.append(scores)
        tp_parts.append(true_positive)
        image_index_parts.append(np.full(len(scores), image_index, dtype=np.int32))
    scores = np.concatenate(score_parts) if score_parts else np.empty(0, dtype=float)
    true_positive = np.concatenate(tp_parts) if tp_parts else np.empty(0, dtype=float)
    detection_image_index = (
        np.concatenate(image_index_parts) if image_index_parts else np.empty(0, dtype=np.int32)
    )
    order = np.argsort(-scores, kind="mergesort")
    return {
        "n_gt": n_gt,
        "selected_tp": selected_tp,
        "selected_fp": selected_fp,
        "true_positive": true_positive[order],
        "detection_image_index": detection_image_index[order],
    }


def weighted_bootstrap_metrics(prepared: dict[str, np.ndarray], counts: np.ndarray) -> dict[str, float]:
    n_gt = float(np.dot(prepared["n_gt"], counts))
    tp = float(np.dot(prepared["selected_tp"], counts))
    fp = float(np.dot(prepared["selected_fp"], counts))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    detection_weights = counts[prepared["detection_image_index"]]
    keep = detection_weights > 0
    if n_gt and np.any(keep):
        weights = detection_weights[keep].astype(float)
        true_positive = prepared["true_positive"][keep]
        cumulative_tp = np.cumsum(true_positive * weights)
        cumulative_fp = np.cumsum((1.0 - true_positive) * weights)
        ap_recall = cumulative_tp / n_gt
        ap_precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-12)
        envelope = np.maximum.accumulate(ap_precision[::-1])[::-1]
        positions = np.searchsorted(ap_recall, np.linspace(0.0, 1.0, 101), side="left")
        valid = positions < len(envelope)
        interpolated = np.zeros(101, dtype=float)
        interpolated[valid] = envelope[positions[valid]]
        ap50 = float(interpolated.mean())
    else:
        ap50 = 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ap50": ap50,
        "miss_rate": 1.0 - recall,
        "false_discovery_rate": 1.0 - precision if tp + fp else float("nan"),
        "fppi": fp / counts.sum() if counts.sum() else 0.0,
    }


def bootstrap_task(task: tuple[str, str, int]) -> tuple[str, str, int, dict]:
    view, model, class_id = task
    metric_values: dict[str, list[float]] = defaultdict(list)
    assert _BOOTSTRAP_COUNTS is not None
    for counts in _BOOTSTRAP_COUNTS:
        seed_rows = []
        for seed in SEEDS:
            fold_rows = []
            for fold in range(5):
                key = (view, model, seed, fold, class_id)
                fold_rows.append(weighted_bootstrap_metrics(_PREPARED[key], counts))
            seed_rows.append(
                {
                    metric: float(np.nanmean([row[metric] for row in fold_rows]))
                    for metric in BOOTSTRAP_METRICS
                }
            )
        for metric in BOOTSTRAP_METRICS:
            metric_values[metric].append(float(np.nanmean([row[metric] for row in seed_rows])))
    return view, model, class_id, {
        metric: percentile_ci(values) for metric, values in metric_values.items()
    }


def macro_bootstrap_task(model: str) -> tuple[str, str, int, dict]:
    metric_values: dict[str, list[float]] = defaultdict(list)
    assert _BOOTSTRAP_COUNTS is not None
    for counts in _BOOTSTRAP_COUNTS:
        seed_rows = []
        for seed in SEEDS:
            fold_rows = []
            for fold in range(5):
                class_rows = [
                    weighted_bootstrap_metrics(
                        _PREPARED[("six_class", model, seed, fold, class_id)], counts
                    )
                    for class_id in range(len(CLASS_NAMES))
                ]
                fold_rows.append(
                    {
                        metric: float(np.nanmean([row[metric] for row in class_rows]))
                        for metric in BOOTSTRAP_METRICS
                    }
                )
            seed_rows.append(
                {
                    metric: float(np.nanmean([row[metric] for row in fold_rows]))
                    for metric in BOOTSTRAP_METRICS
                }
            )
        for metric in BOOTSTRAP_METRICS:
            metric_values[metric].append(float(np.nanmean([row[metric] for row in seed_rows])))
    return "six_class", model, -1, {
        metric: percentile_ci(values) for metric, values in metric_values.items()
    }


def summarize(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    real_root = args.real_root.resolve()
    results_dir = root / "external_real" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    ground_truth = load_ground_truth(real_root)
    binary_ground_truth = collapse_binary(ground_truth)
    audit = audit_real_dataset(real_root, ground_truth)
    images_by_group, group_stratum, group_manifest = build_source_groups(ground_truth)
    write_csv(results_dir / "real_source_group_manifest.csv", group_manifest)

    threshold_lookup = {
        (row["model"], int(row["seed"]), int(row["outer_fold"])): float(row["operating_threshold"])
        for row in read_csv(root / "results" / "fold_thresholds.csv")
    }
    run_class_rows = []
    run_binary_rows = []
    run_macro_rows = []
    all_stats = {}
    run_metadata = []
    for model in DISPLAY_MODELS:
        for seed in SEEDS:
            for fold in range(5):
                path = prediction_path(root, model, fold, seed)
                predictions, payload = load_predictions(path)
                if set(predictions) != set(ground_truth):
                    raise RuntimeError(f"Coverage mismatch in {path}")
                run_metadata.append(
                    {
                        "model": model,
                        "seed": seed,
                        "fold": fold,
                        "checkpoint_sha256": payload["checkpoint_sha256"],
                        "inference_seconds": payload["inference_seconds"],
                    }
                )
                threshold = threshold_lookup[(model, seed, fold)]
                per_class = []
                for class_id, class_name in enumerate(CLASS_NAMES):
                    stats_by_iou = {
                        iou: image_match_stats(ground_truth, predictions, class_id, iou)
                        for iou in IOU_THRESHOLDS
                    }
                    all_stats[("six_class", model, seed, fold, class_id)] = stats_by_iou
                    metrics = class_metrics(stats_by_iou, sorted(ground_truth), threshold)
                    per_class.append(metrics)
                    run_class_rows.append(
                        {
                            "model": model,
                            "model_label": DISPLAY_MODELS[model],
                            "seed": seed,
                            "outer_fold": fold,
                            "class_id": class_id,
                            "class": class_name,
                            "operating_threshold": threshold,
                            **metrics,
                        }
                    )
                run_macro_rows.append(
                    {
                        "model": model,
                        "model_label": DISPLAY_MODELS[model],
                        "seed": seed,
                        "outer_fold": fold,
                        "operating_threshold": threshold,
                        **{metric: float(np.nanmean([row[metric] for row in per_class])) for metric in METRICS},
                    }
                )

                binary_predictions = collapse_binary(predictions)
                for binary_class_id, binary_class_name in enumerate(["Damage", "Normal"]):
                    stats_by_iou = {
                        iou: image_match_stats(
                            binary_ground_truth, binary_predictions, binary_class_id, iou
                        )
                        for iou in IOU_THRESHOLDS
                    }
                    all_stats[("binary", model, seed, fold, binary_class_id)] = stats_by_iou
                    metrics = class_metrics(stats_by_iou, sorted(ground_truth), threshold)
                    run_binary_rows.append(
                        {
                            "model": model,
                            "model_label": DISPLAY_MODELS[model],
                            "seed": seed,
                            "outer_fold": fold,
                            "class_id": binary_class_id,
                            "class": binary_class_name,
                            "operating_threshold": threshold,
                            **metrics,
                        }
                    )

    write_csv(results_dir / "per_run_per_class_metrics.csv", run_class_rows)
    write_csv(results_dir / "per_run_binary_metrics.csv", run_binary_rows)
    write_csv(results_dir / "per_run_macro_metrics.csv", run_macro_rows)

    def seed_aggregate(rows: list[dict], class_names: list[str] | None) -> list[dict]:
        output = []
        for model in DISPLAY_MODELS:
            for seed in SEEDS:
                model_seed_rows = [
                    row for row in rows if row["model"] == model and int(row["seed"]) == seed
                ]
                if class_names is None:
                    partitions = [(None, model_seed_rows)]
                else:
                    partitions = [
                        (name, [row for row in model_seed_rows if row["class"] == name])
                        for name in class_names
                    ]
                for class_name, partition in partitions:
                    current = {
                        "model": model,
                        "model_label": DISPLAY_MODELS[model],
                        "seed": seed,
                    }
                    if class_name is not None:
                        current["class"] = class_name
                        current["n_gt"] = int(partition[0]["n_gt"])
                    for metric in METRICS:
                        current[metric] = float(np.nanmean([float(row[metric]) for row in partition]))
                    output.append(current)
        return output

    seed_class_rows = seed_aggregate(run_class_rows, CLASS_NAMES)
    seed_binary_rows = seed_aggregate(run_binary_rows, ["Damage", "Normal"])
    seed_macro_rows = seed_aggregate(run_macro_rows, None)
    write_csv(results_dir / "per_seed_per_class_metrics.csv", seed_class_rows)
    write_csv(results_dir / "per_seed_binary_metrics.csv", seed_binary_rows)
    write_csv(results_dir / "per_seed_macro_metrics.csv", seed_macro_rows)

    rng = np.random.default_rng(args.bootstrap_seed)
    groups_by_stratum: dict[str, list[str]] = defaultdict(list)
    for group, stratum in group_stratum.items():
        groups_by_stratum[stratum].append(group)
    image_order = sorted(ground_truth)
    image_index_lookup = {name: index for index, name in enumerate(image_order)}
    bootstrap_counts = np.zeros((args.bootstrap, len(image_order)), dtype=np.int16)
    for _ in range(args.bootstrap):
        sampled_images = []
        for stratum in sorted(groups_by_stratum):
            groups = np.asarray(sorted(groups_by_stratum[stratum]))
            sampled_groups = rng.choice(groups, size=len(groups), replace=True)
            for group in sampled_groups:
                sampled_images.extend(images_by_group[str(group)])
        row_index = _
        for image_name, count in Counter(sampled_images).items():
            bootstrap_counts[row_index, image_index_lookup[image_name]] = count

    prepared = {}
    for key, stats_by_iou in all_stats.items():
        _view, model, seed, fold, _class_id = key
        prepared[key] = prepare_weighted_bootstrap(
            stats_by_iou[0.50], image_order, threshold_lookup[(model, seed, fold)]
        )

    global _BOOTSTRAP_COUNTS, _STATS, _THRESHOLDS, _PREPARED
    _BOOTSTRAP_COUNTS = bootstrap_counts
    _STATS = all_stats
    _THRESHOLDS = threshold_lookup
    _PREPARED = prepared
    bootstrap_results = []
    if args.workers == 1:
        for model in DISPLAY_MODELS:
            bootstrap_results.append(macro_bootstrap_task(model))
            for class_id in range(2):
                bootstrap_results.append(bootstrap_task(("binary", model, class_id)))
    else:
        # Forked workers share cached per-image matching statistics.
        context = mp.get_context("fork")
        with context.Pool(processes=min(args.workers, len(DISPLAY_MODELS))) as pool:
            bootstrap_results.extend(pool.map(macro_bootstrap_task, DISPLAY_MODELS))
        binary_tasks = [
            ("binary", model, class_id) for model in DISPLAY_MODELS for class_id in range(2)
        ]
        with context.Pool(processes=min(args.workers, len(binary_tasks))) as pool:
            bootstrap_results.extend(pool.map(bootstrap_task, binary_tasks))

    ci_lookup = {
        (view, model, class_id): intervals
        for view, model, class_id, intervals in bootstrap_results
    }

    def aggregate_over_seeds(
        seed_rows: list[dict], class_names: list[str] | None, view: str
    ) -> list[dict]:
        output = []
        for model in DISPLAY_MODELS:
            if class_names is None:
                partitions = [(-1, None)]
            else:
                partitions = list(enumerate(class_names))
            for class_id, class_name in partitions:
                rows = [row for row in seed_rows if row["model"] == model]
                if class_name is not None:
                    rows = [row for row in rows if row["class"] == class_name]
                current = {
                    "model": model,
                    "model_label": DISPLAY_MODELS[model],
                    "n_seeds": len(SEEDS),
                    "n_fold_models_per_seed": 5,
                }
                if class_name is not None:
                    current["class_id"] = class_id
                    current["class"] = class_name
                    current["n_gt"] = int(rows[0]["n_gt"])
                for metric in METRICS:
                    mean, std = mean_std([float(row[metric]) for row in rows])
                    current[f"{metric}_mean"] = mean
                    current[f"{metric}_std"] = std
                intervals = ci_lookup.get((view, model, class_id))
                if intervals:
                    for metric, (low, high) in intervals.items():
                        current[f"{metric}_ci95_low"] = low
                        current[f"{metric}_ci95_high"] = high
                output.append(current)
        return output

    aggregate_macro_rows = aggregate_over_seeds(seed_macro_rows, None, "six_class")
    aggregate_binary_rows = aggregate_over_seeds(
        seed_binary_rows, ["Damage", "Normal"], "binary"
    )

    aggregate_class_rows = []
    for model in DISPLAY_MODELS:
        for class_id, class_name in enumerate(CLASS_NAMES):
            rows = [
                row
                for row in seed_class_rows
                if row["model"] == model and row["class"] == class_name
            ]
            output = {
                "model": model,
                "model_label": DISPLAY_MODELS[model],
                "class_id": class_id,
                "class": class_name,
                "n_seeds": len(SEEDS),
                "n_fold_models_per_seed": 5,
                "n_gt": int(rows[0]["n_gt"]),
            }
            for metric in METRICS:
                mean, std = mean_std([float(row[metric]) for row in rows])
                output[f"{metric}_mean"] = mean
                output[f"{metric}_std"] = std
            aggregate_class_rows.append(output)

    write_csv(results_dir / "aggregate_macro_metrics.csv", aggregate_macro_rows)
    write_csv(results_dir / "aggregate_binary_metrics.csv", aggregate_binary_rows)
    write_csv(results_dir / "aggregate_per_class_metrics.csv", aggregate_class_rows)

    internal_macro = {row["model"]: row for row in read_csv(root / "results" / "aggregate_macro_metrics.csv")}
    comparison_rows = []
    for external in aggregate_macro_rows:
        internal = internal_macro[external["model"]]
        row = {
            "model": external["model"],
            "model_label": external["model_label"],
        }
        for metric in METRICS:
            internal_value = float(internal[f"{metric}_mean"])
            external_value = float(external[f"{metric}_mean"])
            row[f"rfdd_srd_{metric}"] = internal_value
            row[f"real_{metric}"] = external_value
            row[f"real_minus_rfdd_srd_{metric}"] = external_value - internal_value
        comparison_rows.append(row)
    write_csv(results_dir / "cross_domain_comparison.csv", comparison_rows)

    stratum_group_counts = Counter(group_stratum.values())
    metadata = {
        "external_dataset": str(real_root),
        "dataset_audit": audit,
        "models": DISPLAY_MODELS,
        "checkpoints": "five outer-fold checkpoints for each of three independent seeds per architecture",
        "training": "RFDD-SRD training partitions only",
        "model_selection": "RFDD-SRD validation partitions only",
        "external_test_use": "single frozen evaluation after all checkpoints and thresholds were fixed",
        "operating_thresholds": "fold-specific thresholds from RFDD-SRD validation macro-F1",
        "matching": "class-aware greedy one-to-one matching in descending score order at IoU 0.50",
        "ap": "101-point interpolated AP; AP50:95 averages IoU 0.50:0.05:0.95",
        "false_alarm_reporting": "FDR=FP/(TP+FP); FPPI=FP/100 external images",
        "aggregation": "fold metrics averaged within seed; mean and sample SD across three seeds",
        "bootstrap": {
            "replicates": args.bootstrap,
            "seed": args.bootstrap_seed,
            "unit": "source/acquisition group",
            "stratification": "Normal-only and each of five defect classes",
            "group_counts_by_stratum": dict(stratum_group_counts),
            "interval": "percentile 95%",
            "reported_for": "six-class macro and pooled Damage/Normal metrics",
        },
        "run_metadata": run_metadata,
    }
    (results_dir / "evaluation_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (results_dir / "dataset_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote external evaluation to {results_dir}")


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    infer_parser = subparsers.add_parser("infer")
    infer_parser.add_argument(
        "--root", type=Path, default=Path("artifacts/cv")
    )
    infer_parser.add_argument(
        "--real-root", type=Path, default=Path("data/real")
    )
    infer_parser.add_argument("--models", nargs="+", choices=list(DISPLAY_MODELS), required=True)
    infer_parser.add_argument("--folds", type=parse_int_list, default=list(range(5)))
    infer_parser.add_argument("--seeds", type=parse_int_list, default=SEEDS)
    infer_parser.add_argument("--device", type=int, required=True)
    infer_parser.add_argument("--overwrite", action="store_true")

    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument(
        "--root", type=Path, default=Path("artifacts/cv")
    )
    summary_parser.add_argument(
        "--real-root", type=Path, default=Path("data/real")
    )
    summary_parser.add_argument("--bootstrap", type=int, default=2000)
    summary_parser.add_argument("--bootstrap-seed", type=int, default=20260909)
    summary_parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))

    args = parser.parse_args()
    if args.command == "infer":
        infer_runs(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
