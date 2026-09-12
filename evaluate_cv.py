#!/usr/bin/env python3
"""Evaluate out-of-fold RFDD-SRD predictions with group-cluster bootstrap CIs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


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
OPERATING_THRESHOLDS = [round(value, 2) for value in np.arange(0.01, 1.00, 0.01)]
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

_BOOTSTRAP_SAMPLES: list[list[str]] = []
_ALL_STATS: dict = {}
_THRESHOLDS_LOOKUP: dict = {}


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_ground_truth(images_dir: Path, labels_dir: Path) -> dict[str, list[dict]]:
    result = {}
    for image_path in sorted(images_dir.glob("*")):
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        records = []
        label_path = labels_dir / f"{image_path.stem}.txt"
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
                        ],
                        dtype=float,
                    ),
                }
            )
        result[image_path.name] = records
    return result


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
    for image_name, image_gt in ground_truth.items():
        gt_boxes = np.asarray(
            [record["xyxy"] for record in image_gt if record["class_id"] == class_id], dtype=float
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
    interpolated = [
        precision[recall >= point].max() if np.any(recall >= point) else 0.0
        for point in np.linspace(0.0, 1.0, 101)
    ]
    return float(np.mean(interpolated))


def aggregate(
    stats: dict[str, dict],
    image_keys: list[str],
    threshold: float | dict[str, float] | None,
) -> dict[str, float]:
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
            image_threshold = threshold[image_name] if isinstance(threshold, dict) else threshold
            selected = scores >= image_threshold
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


def load_payload(path: Path) -> tuple[dict[str, list[dict]], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    predictions = {row["image"]: row["detections"] for row in payload["images"]}
    return predictions, payload


def choose_threshold(
    ground_truth: dict[str, list[dict]], predictions: dict[str, list[dict]], image_keys: list[str]
) -> tuple[float, float]:
    stats = {
        class_id: image_match_stats(ground_truth, predictions, class_id, 0.50)
        for class_id in range(len(CLASS_NAMES))
    }
    best_threshold = 0.50
    best_macro_f1 = -1.0
    for threshold in OPERATING_THRESHOLDS:
        macro_f1 = float(
            np.mean([aggregate(stats[class_id], image_keys, threshold)["f1"] for class_id in range(len(CLASS_NAMES))])
        )
        if macro_f1 > best_macro_f1 + 1e-12 or (
            abs(macro_f1 - best_macro_f1) <= 1e-12 and threshold > best_threshold
        ):
            best_threshold = threshold
            best_macro_f1 = macro_f1
    return best_threshold, best_macro_f1


def class_metrics(
    stats_by_iou: dict[float, dict[str, dict]],
    image_keys: list[str],
    thresholds_by_image: dict[str, float],
) -> dict[str, float]:
    at_50 = aggregate(stats_by_iou[0.50], image_keys, thresholds_by_image)
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


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(np.nanmean(array)), float(np.nanstd(array, ddof=1))


def ci(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if np.all(np.isnan(array)):
        return float("nan"), float("nan")
    low, high = np.nanpercentile(array, [2.5, 97.5])
    return float(low), float(high)


def bootstrap_ci_task(task: tuple[str, str, int]) -> tuple[str, str, int, dict[str, tuple[float, float]]]:
    """Compute one class or macro CI task in a forked worker."""
    kind, model, class_id = task
    values: dict[str, list[float]] = defaultdict(list)
    for image_sample in _BOOTSTRAP_SAMPLES:
        if kind == "class":
            seed_values = [
                class_metrics(
                    _ALL_STATS[(model, seed, class_id)],
                    image_sample,
                    _THRESHOLDS_LOOKUP[(model, seed)],
                )
                for seed in SEEDS
            ]
            for metric in METRICS:
                values[metric].append(float(np.nanmean([row[metric] for row in seed_values])))
        else:
            seed_macro_values = []
            for seed in SEEDS:
                seed_class_values = [
                    class_metrics(
                        _ALL_STATS[(model, seed, current_class_id)],
                        image_sample,
                        _THRESHOLDS_LOOKUP[(model, seed)],
                    )
                    for current_class_id in range(len(CLASS_NAMES))
                ]
                seed_macro_values.append(
                    {
                        metric: float(np.nanmean([row[metric] for row in seed_class_values]))
                        for metric in METRICS
                    }
                )
            for metric in METRICS:
                values[metric].append(
                    float(np.nanmean([row[metric] for row in seed_macro_values]))
                )
    return kind, model, class_id, {metric: ci(values[metric]) for metric in METRICS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/cv"))
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260908)
    parser.add_argument("--workers", type=int, default=min(12, max(1, (os.cpu_count() or 2) // 2)))
    args = parser.parse_args()
    root = args.root.resolve()
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    ground_truth = load_ground_truth(root / "dataset" / "images", root / "dataset" / "labels")
    manifest = read_csv(root / "source_template_manifest.csv")
    image_to_group = {
        row.get("image_name", f"{row['image_id']}.png"): row["source_template_group"]
        for row in manifest
    }
    groups_by_class: dict[int, list[str]] = defaultdict(list)
    for row in read_csv(root / "group_manifest.csv"):
        groups_by_class[int(row["defect_class_id"])].append(row["source_template_group"])
    images_by_group: dict[str, list[str]] = defaultdict(list)
    for image_name, group in image_to_group.items():
        images_by_group[group].append(image_name)

    threshold_rows = []
    run_metadata = []
    all_stats: dict[tuple[str, int, int], dict[float, dict[str, dict]]] = {}
    thresholds_lookup: dict[tuple[str, int], dict[str, float]] = {}
    seed_class_rows = []
    seed_macro_rows = []

    for model in DISPLAY_MODELS:
        for seed in SEEDS:
            combined_test_predictions = {}
            thresholds_by_image = {}
            model_seed_metadata = []
            for fold in range(5):
                run_dir = root / "runs" / model / f"fold_{fold}_seed_{seed}"
                val_predictions, val_payload = load_payload(run_dir / "predictions_val.json")
                test_predictions, test_payload = load_payload(run_dir / "predictions_test.json")
                val_keys = sorted(val_predictions)
                threshold, validation_macro_f1 = choose_threshold(ground_truth, val_predictions, val_keys)
                threshold_rows.append(
                    {
                        "model": model,
                        "model_label": DISPLAY_MODELS[model],
                        "seed": seed,
                        "outer_fold": fold,
                        "operating_threshold": threshold,
                        "validation_macro_f1": validation_macro_f1,
                        "n_validation_images": len(val_keys),
                        "n_test_images": len(test_predictions),
                    }
                )
                overlap = set(combined_test_predictions).intersection(test_predictions)
                if overlap:
                    raise RuntimeError(f"Repeated outer-test predictions: {model} {seed} {overlap}")
                combined_test_predictions.update(test_predictions)
                thresholds_by_image.update({image_name: threshold for image_name in test_predictions})
                model_seed_metadata.append(
                    {
                        "fold": fold,
                        "completed": json.loads((run_dir / "completed.json").read_text(encoding="utf-8")),
                    }
                )
            if set(combined_test_predictions) != set(ground_truth):
                missing = set(ground_truth) - set(combined_test_predictions)
                extra = set(combined_test_predictions) - set(ground_truth)
                raise RuntimeError(f"OOF coverage error {model} {seed}: missing={missing}, extra={extra}")

            thresholds_lookup[(model, seed)] = thresholds_by_image
            per_class = []
            for class_id, class_name in enumerate(CLASS_NAMES):
                stats_by_iou = {
                    iou: image_match_stats(ground_truth, combined_test_predictions, class_id, iou)
                    for iou in IOU_THRESHOLDS
                }
                all_stats[(model, seed, class_id)] = stats_by_iou
                metrics = class_metrics(stats_by_iou, sorted(ground_truth), thresholds_by_image)
                per_class.append(metrics)
                seed_class_rows.append(
                    {
                        "model": model,
                        "model_label": DISPLAY_MODELS[model],
                        "seed": seed,
                        "class_id": class_id,
                        "class": class_name,
                        **metrics,
                    }
                )
            seed_macro_rows.append(
                {
                    "model": model,
                    "model_label": DISPLAY_MODELS[model],
                    "seed": seed,
                    **{metric: float(np.nanmean([row[metric] for row in per_class])) for metric in METRICS},
                    "n_gt": int(sum(row["n_gt"] for row in per_class)),
                    "tp": int(sum(row["tp"] for row in per_class)),
                    "fp": int(sum(row["fp"] for row in per_class)),
                    "fn": int(sum(row["fn"] for row in per_class)),
                }
            )
            run_metadata.append({"model": model, "seed": seed, "folds": model_seed_metadata})

    # Defect-stratified source/template-group bootstrap. Each replicate samples
    # ten two-image groups with replacement within each of the five defect classes.
    rng = np.random.default_rng(args.bootstrap_seed)
    bootstrap_samples = []
    for _ in range(args.bootstrap):
        sampled_images = []
        for class_id in range(5):
            groups = np.asarray(sorted(groups_by_class[class_id], key=int))
            sampled_groups = rng.choice(groups, size=len(groups), replace=True)
            for group in sampled_groups:
                sampled_images.extend(sorted(images_by_group[str(group)]))
        bootstrap_samples.append(sampled_images)

    global _BOOTSTRAP_SAMPLES, _ALL_STATS, _THRESHOLDS_LOOKUP
    _BOOTSTRAP_SAMPLES = bootstrap_samples
    _ALL_STATS = all_stats
    _THRESHOLDS_LOOKUP = thresholds_lookup
    bootstrap_tasks = [
        ("class", model, class_id)
        for model in DISPLAY_MODELS
        for class_id in range(len(CLASS_NAMES))
    ] + [("macro", model, -1) for model in DISPLAY_MODELS]
    if args.workers == 1:
        bootstrap_results = [bootstrap_ci_task(task) for task in bootstrap_tasks]
    else:
        context = mp.get_context("fork")
        with context.Pool(processes=min(args.workers, len(bootstrap_tasks))) as pool:
            bootstrap_results = pool.map(bootstrap_ci_task, bootstrap_tasks)
    class_ci_lookup = {}
    macro_ci_lookup = {}
    for kind, model, class_id, intervals in bootstrap_results:
        if kind == "class":
            class_ci_lookup[(model, class_id)] = intervals
        else:
            macro_ci_lookup[model] = intervals

    aggregate_class_rows = []
    aggregate_macro_rows = []
    for model in DISPLAY_MODELS:
        model_seed_class_rows = [row for row in seed_class_rows if row["model"] == model]
        for class_id, class_name in enumerate(CLASS_NAMES):
            rows = [row for row in model_seed_class_rows if int(row["class_id"]) == class_id]
            output = {
                "model": model,
                "model_label": DISPLAY_MODELS[model],
                "class_id": class_id,
                "class": class_name,
                "n_seeds": len(SEEDS),
                "n_gt": int(rows[0]["n_gt"]),
            }
            for metric in METRICS:
                mean, std = mean_std([row[metric] for row in rows])
                output[f"{metric}_mean"] = mean
                output[f"{metric}_std"] = std

            for metric in METRICS:
                low, high = class_ci_lookup[(model, class_id)][metric]
                output[f"{metric}_ci95_low"] = low
                output[f"{metric}_ci95_high"] = high
            aggregate_class_rows.append(output)

        rows = [row for row in seed_macro_rows if row["model"] == model]
        output = {
            "model": model,
            "model_label": DISPLAY_MODELS[model],
            "n_seeds": len(SEEDS),
            "n_folds": 5,
            "n_images": len(ground_truth),
            "n_gt": int(rows[0]["n_gt"]),
        }
        for metric in METRICS:
            mean, std = mean_std([row[metric] for row in rows])
            output[f"{metric}_mean"] = mean
            output[f"{metric}_std"] = std

        for metric in METRICS:
            low, high = macro_ci_lookup[model][metric]
            output[f"{metric}_ci95_low"] = low
            output[f"{metric}_ci95_high"] = high
        aggregate_macro_rows.append(output)

    write_csv_csv(results_dir / "fold_thresholds.csv", threshold_rows)
    write_csv_csv(results_dir / "per_seed_per_class_metrics.csv", seed_class_rows)
    write_csv_csv(results_dir / "per_seed_macro_metrics.csv", seed_macro_rows)
    write_csv_csv(results_dir / "aggregate_per_class_metrics.csv", aggregate_class_rows)
    write_csv_csv(results_dir / "aggregate_macro_metrics.csv", aggregate_macro_rows)

    metadata = {
        "dataset": "RFDD-SRD reannotated release (2026-09-10)",
        "images": len(ground_truth),
        "ground_truth_boxes": sum(len(records) for records in ground_truth.values()),
        "models": DISPLAY_MODELS,
        "seeds": SEEDS,
        "outer_folds": 5,
        "outer_test_coverage_per_seed": "each of the 100 images exactly once",
        "threshold_selection": "per-fold RFDD-SRD validation subset; fixed grid 0.01:0.01:0.99; maximize macro-F1 at IoU 0.50; higher threshold breaks ties",
        "matching": "class-aware greedy one-to-one matching in descending score order",
        "ap": "101-point interpolated AP; AP50:95 averages IoU 0.50:0.05:0.95",
        "false_alarm_reporting": "FDR=FP/(TP+FP) and FPPI=FP/100 images",
        "bootstrap": {
            "replicates": args.bootstrap,
            "seed": args.bootstrap_seed,
            "unit": "two-image source/template group",
            "stratification": "10 groups sampled with replacement within each of five defect classes",
            "interval": "percentile 95%",
            "aggregation": "metric averaged over three independent training seeds within each bootstrap replicate",
        },
        "rfdd_srd_training_data_used": True,
        "run_metadata": run_metadata,
    }
    (results_dir / "evaluation_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Wrote cross-validation results to {results_dir}")


if __name__ == "__main__":
    main()
