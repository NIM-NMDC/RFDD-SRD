#!/usr/bin/env python3
"""Summarize final-model evaluation on the independent real dataset."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import evaluate_external_cv as ev


ROOT = Path("artifacts/cv")
EXP_ROOT = ROOT / "full_train_real_test"
REAL_ROOT = Path("data/real")
SEEDS = [20260908, 20260909, 20260910]
MODELS = ev.DISPLAY_MODELS


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prediction_path(model: str, seed: int) -> Path:
    return EXP_ROOT / "runs" / model / f"seed_{seed}" / "predictions_real.json"


def metric_mean_std(rows: list[dict], metric: str) -> tuple[float, float]:
    values = np.asarray([float(row[metric]) for row in rows], dtype=float)
    return float(np.nanmean(values)), float(np.nanstd(values, ddof=1))


def build_bootstrap_counts(
    ground_truth: dict, replicates: int, seed: int
) -> tuple[np.ndarray, list[str], list[dict]]:
    images_by_group, group_stratum, manifest = ev.build_source_groups(ground_truth)
    groups_by_stratum: dict[str, list[str]] = defaultdict(list)
    for group, stratum in group_stratum.items():
        groups_by_stratum[stratum].append(group)
    image_order = sorted(ground_truth)
    image_index = {name: index for index, name in enumerate(image_order)}
    counts = np.zeros((replicates, len(image_order)), dtype=np.int16)
    rng = np.random.default_rng(seed)
    for replicate in range(replicates):
        sampled_images = []
        for stratum in sorted(groups_by_stratum):
            groups = np.asarray(sorted(groups_by_stratum[stratum]))
            sampled = rng.choice(groups, size=len(groups), replace=True)
            for group in sampled:
                sampled_images.extend(images_by_group[str(group)])
        for image_name, frequency in Counter(sampled_images).items():
            counts[replicate, image_index[image_name]] = frequency
    return counts, image_order, manifest


def ci_from_prepared(
    prepared_by_seed: dict[int, dict], counts: np.ndarray
) -> dict[str, tuple[float, float]]:
    values: dict[str, list[float]] = defaultdict(list)
    for current_counts in counts:
        seed_metrics = [
            ev.weighted_bootstrap_metrics(prepared_by_seed[seed], current_counts)
            for seed in SEEDS
        ]
        for metric in ev.BOOTSTRAP_METRICS:
            values[metric].append(float(np.nanmean([row[metric] for row in seed_metrics])))
    return {metric: ev.percentile_ci(metric_values) for metric, metric_values in values.items()}


def macro_ci(
    prepared: dict[tuple[str, int, int], dict], model: str, counts: np.ndarray
) -> dict[str, tuple[float, float]]:
    values: dict[str, list[float]] = defaultdict(list)
    for current_counts in counts:
        seed_metrics = []
        for seed in SEEDS:
            class_rows = [
                ev.weighted_bootstrap_metrics(
                    prepared[(model, seed, class_id)], current_counts
                )
                for class_id in range(len(ev.CLASS_NAMES))
            ]
            seed_metrics.append(
                {
                    metric: float(np.nanmean([row[metric] for row in class_rows]))
                    for metric in ev.BOOTSTRAP_METRICS
                }
            )
        for metric in ev.BOOTSTRAP_METRICS:
            values[metric].append(float(np.nanmean([row[metric] for row in seed_metrics])))
    return {metric: ev.percentile_ci(metric_values) for metric, metric_values in values.items()}


def aggregate_rows(
    seed_rows: list[dict], class_names: list[str] | None, ci_lookup: dict
) -> list[dict]:
    output = []
    for model in MODELS:
        partitions = [(None, -1)] if class_names is None else list(
            zip(class_names, range(len(class_names)))
        )
        for class_name, class_id in partitions:
            rows = [row for row in seed_rows if row["model"] == model]
            if class_name is not None:
                rows = [row for row in rows if row["class"] == class_name]
            current = {
                "model": model,
                "model_label": MODELS[model],
                "n_seeds": len(SEEDS),
                "training_images": 100,
                "external_test_images": 100,
            }
            if class_name is not None:
                current.update(
                    {"class_id": class_id, "class": class_name, "n_gt": int(rows[0]["n_gt"])}
                )
            for metric in ev.METRICS:
                mean, std = metric_mean_std(rows, metric)
                current[f"{metric}_mean"] = mean
                current[f"{metric}_std"] = std
            intervals = ci_lookup[(model, class_id)]
            for metric, (low, high) in intervals.items():
                current[f"{metric}_ci95_low"] = low
                current[f"{metric}_ci95_high"] = high
            output.append(current)
    return output


def main() -> None:
    global ROOT, EXP_ROOT, REAL_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-root", type=Path, default=ROOT)
    parser.add_argument("--real-root", type=Path, default=REAL_ROOT)
    parser.add_argument(
        "--exp-root",
        type=Path,
        default=None,
        help="Experiment directory containing runs/ and frozen_thresholds.csv.",
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260909)
    args = parser.parse_args()
    ROOT = args.cv_root.resolve()
    EXP_ROOT = (
        args.exp_root.resolve()
        if args.exp_root is not None
        else ROOT / "full_train_real_test"
    )
    REAL_ROOT = args.real_root.resolve()

    missing = [
        str(prediction_path(model, seed))
        for model in MODELS
        for seed in SEEDS
        if not prediction_path(model, seed).is_file()
    ]
    if missing:
        raise RuntimeError("Missing prediction files:\n" + "\n".join(missing))

    results_dir = EXP_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    ground_truth = ev.load_ground_truth(REAL_ROOT)
    binary_ground_truth = ev.collapse_binary(ground_truth)
    audit = ev.audit_real_dataset(REAL_ROOT, ground_truth)
    counts, image_order, manifest = build_bootstrap_counts(
        ground_truth, args.bootstrap, args.bootstrap_seed
    )
    write_csv(results_dir / "real_source_group_manifest.csv", manifest)
    thresholds = {
        (row["model"], int(row["seed"])): float(row["operating_threshold"])
        for row in read_csv(EXP_ROOT / "frozen_thresholds.csv")
    }

    per_seed_class, per_seed_binary, per_seed_macro = [], [], []
    six_prepared: dict[tuple[str, int, int], dict] = {}
    binary_prepared: dict[tuple[str, int, int], dict] = {}
    metadata_rows = []
    for model in MODELS:
        for seed in SEEDS:
            predictions, payload = ev.load_predictions(prediction_path(model, seed))
            if set(predictions) != set(ground_truth):
                raise RuntimeError(f"Coverage mismatch: {model}/{seed}")
            threshold = thresholds[(model, seed)]
            metadata_rows.append(
                {
                    "model": model, "seed": seed, "threshold": threshold,
                    "checkpoint_sha256": payload["final_checkpoint_sha256"],
                    "epochs_completed": payload["epochs_completed"],
                    "train_seconds": payload["train_seconds"],
                    "inference_seconds": payload["inference_seconds"],
                }
            )
            class_metrics = []
            for class_id, class_name in enumerate(ev.CLASS_NAMES):
                stats_by_iou = {
                    iou: ev.image_match_stats(ground_truth, predictions, class_id, iou)
                    for iou in ev.IOU_THRESHOLDS
                }
                metrics = ev.class_metrics(stats_by_iou, image_order, threshold)
                class_metrics.append(metrics)
                per_seed_class.append(
                    {"model": model, "model_label": MODELS[model], "seed": seed,
                     "class_id": class_id, "class": class_name,
                     "operating_threshold": threshold, **metrics}
                )
                six_prepared[(model, seed, class_id)] = ev.prepare_weighted_bootstrap(
                    stats_by_iou[0.50], image_order, threshold
                )
            per_seed_macro.append(
                {"model": model, "model_label": MODELS[model], "seed": seed,
                 "operating_threshold": threshold,
                 **{metric: float(np.nanmean([row[metric] for row in class_metrics]))
                    for metric in ev.METRICS}}
            )
            binary_predictions = ev.collapse_binary(predictions)
            for class_id, class_name in enumerate(["Damage", "Normal"]):
                stats_by_iou = {
                    iou: ev.image_match_stats(binary_ground_truth, binary_predictions, class_id, iou)
                    for iou in ev.IOU_THRESHOLDS
                }
                metrics = ev.class_metrics(stats_by_iou, image_order, threshold)
                per_seed_binary.append(
                    {"model": model, "model_label": MODELS[model], "seed": seed,
                     "class_id": class_id, "class": class_name,
                     "operating_threshold": threshold, **metrics}
                )
                binary_prepared[(model, seed, class_id)] = ev.prepare_weighted_bootstrap(
                    stats_by_iou[0.50], image_order, threshold
                )

    write_csv(results_dir / "per_seed_per_class_metrics.csv", per_seed_class)
    write_csv(results_dir / "per_seed_binary_metrics.csv", per_seed_binary)
    write_csv(results_dir / "per_seed_macro_metrics.csv", per_seed_macro)
    write_csv(results_dir / "run_metadata.csv", metadata_rows)

    class_ci, binary_ci, macro_ci_lookup = {}, {}, {}
    for model in MODELS:
        print(f"Bootstrap: {MODELS[model]}", flush=True)
        macro_ci_lookup[(model, -1)] = macro_ci(six_prepared, model, counts)
        for class_id in range(len(ev.CLASS_NAMES)):
            class_ci[(model, class_id)] = ci_from_prepared(
                {seed: six_prepared[(model, seed, class_id)] for seed in SEEDS}, counts
            )
        for class_id in range(2):
            binary_ci[(model, class_id)] = ci_from_prepared(
                {seed: binary_prepared[(model, seed, class_id)] for seed in SEEDS}, counts
            )

    aggregate_class = aggregate_rows(per_seed_class, ev.CLASS_NAMES, class_ci)
    aggregate_binary = aggregate_rows(per_seed_binary, ["Damage", "Normal"], binary_ci)
    aggregate_macro = aggregate_rows(per_seed_macro, None, macro_ci_lookup)
    write_csv(results_dir / "aggregate_per_class_metrics.csv", aggregate_class)
    write_csv(results_dir / "aggregate_binary_metrics.csv", aggregate_binary)
    write_csv(results_dir / "aggregate_macro_metrics.csv", aggregate_macro)
    (results_dir / "dataset_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "dataset_audit": audit,
        "training": "all 100 RFDD-SRD images (619 boxes; reannotated 2026-09-10)",
        "testing": "100 independent real images only",
        "repeats": "three independently trained random seeds",
        "initialization": "COCO-pretrained generic weights",
        "checkpoint_selection": "fixed final epoch; epoch counts frozen from RFDD-SRD CV",
        "threshold_selection": "median RFDD-SRD validation threshold within each model/seed",
        "matching": "class-aware greedy one-to-one matching at IoU 0.50",
        "bootstrap": {"replicates": args.bootstrap, "seed": args.bootstrap_seed,
                      "unit": "source/acquisition group", "stratified": True},
    }
    (results_dir / "evaluation_metadata.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote results to {results_dir}", flush=True)


if __name__ == "__main__":
    main()
