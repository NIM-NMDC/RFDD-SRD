#!/usr/bin/env python3
"""Prepare paired, leakage-aware five-fold CV for the 2026-09-10 labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path


CLASS_NAMES = ["Deformed", "Displaced", "Fractured", "Inverted", "Missing", "Normal"]
PRIMARY_DEFECT_IDS = tuple(range(5))
SPLIT_SEED = 20260908  # identical to the earlier experiment for paired comparison
SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="RFDD-SRD root containing images/ and labels/.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/cv"), help="Cross-validation experiment root.")
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images into the experiment directory instead of creating symbolic links.",
    )
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    image_dir, label_dir = source / "images", source / "labels"
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in SUFFIXES)
    if len(paths) != 100:
        raise RuntimeError(f"Expected 100 images, found {len(paths)}")

    clean_images, clean_labels = output / "dataset" / "images", output / "dataset" / "labels"
    clean_images.mkdir(parents=True, exist_ok=True)
    clean_labels.mkdir(parents=True, exist_ok=True)
    manifest, grouped = [], defaultdict(list)
    for image_path in paths:
        match = re.match(r"^[^-]+-(\d+)-", image_path.stem)
        if not match:
            raise RuntimeError(f"Cannot parse source-template group: {image_path.name}")
        group_id = match.group(1)
        label_path = label_dir / f"{image_path.stem}.txt"
        lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        defect_ids, normal_count = [], 0
        for line in lines:
            values = line.split()
            if len(values) != 5:
                raise ValueError(f"Malformed label: {label_path}")
            class_id = int(values[0])
            if class_id in PRIMARY_DEFECT_IDS:
                defect_ids.append(class_id)
            elif class_id == 5:
                normal_count += 1
            else:
                raise ValueError(f"Invalid class in {label_path}: {class_id}")
        if len(defect_ids) != 1:
            raise RuntimeError(f"Expected one damaged fastener in {label_path}, got {defect_ids}")

        linked_image = clean_images / image_path.name
        if linked_image.exists() or linked_image.is_symlink():
            linked_image.unlink()
        if args.copy_images:
            shutil.copy2(image_path, linked_image)
        else:
            linked_image.symlink_to(image_path)
        (clean_labels / f"{image_path.stem}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        row = {
            "image_id": image_path.stem,
            "image_name": image_path.name,
            "image_sha256": sha256(image_path),
            "source_template_group": group_id,
            "defect_class_id": defect_ids[0],
            "defect_class": CLASS_NAMES[defect_ids[0]],
            "objects_benchmark": len(lines),
            "normal_objects_benchmark": normal_count,
            "benchmark_image_path": str(linked_image.absolute()),
        }
        manifest.append(row)
        grouped[group_id].append(row)

    if len(grouped) != 50:
        raise RuntimeError(f"Expected 50 source groups, found {len(grouped)}")
    groups_by_class = defaultdict(list)
    group_rows = []
    for group_id, rows in grouped.items():
        classes = {int(row["defect_class_id"]) for row in rows}
        if len(rows) != 2 or len(classes) != 1:
            raise RuntimeError(f"Invalid group {group_id}: n={len(rows)}, classes={classes}")
        class_id = next(iter(classes))
        groups_by_class[class_id].append(group_id)
        group_rows.append({
            "source_template_group": group_id,
            "defect_class_id": class_id,
            "defect_class": CLASS_NAMES[class_id],
            "n_images": 2,
            "image_ids": ";".join(sorted(row["image_id"] for row in rows)),
        })
    if any(len(groups_by_class[class_id]) != 10 for class_id in PRIMARY_DEFECT_IDS):
        raise RuntimeError("Expected ten two-image groups for every defect class")

    fold_for_group = {}
    for class_id in PRIMARY_DEFECT_IDS:
        groups = sorted(groups_by_class[class_id], key=int)
        random.Random(SPLIT_SEED + class_id).shuffle(groups)
        for index, group_id in enumerate(groups):
            fold_for_group[group_id] = index % 5
    for row in manifest:
        row["outer_fold"] = fold_for_group[row["source_template_group"]]

    split_rows, summary_rows = [], []
    all_groups = set(grouped)
    for outer_fold in range(5):
        test_groups = {group for group, fold in fold_for_group.items() if fold == outer_fold}
        development_groups = all_groups - test_groups
        validation_groups = set()
        for class_id in PRIMARY_DEFECT_IDS:
            candidates = sorted(
                group for group in development_groups
                if int(grouped[group][0]["defect_class_id"]) == class_id
            )
            random.Random(SPLIT_SEED + 100 * (outer_fold + 1) + class_id).shuffle(candidates)
            validation_groups.update(candidates[:2])
        training_groups = development_groups - validation_groups
        role_for_group = {
            **{group: "train" for group in training_groups},
            **{group: "val" for group in validation_groups},
            **{group: "test" for group in test_groups},
        }
        fold_dir = output / "splits" / f"fold_{outer_fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        for role in ("train", "val", "test"):
            rows = sorted(
                (row for row in manifest if role_for_group[row["source_template_group"]] == role),
                key=lambda row: row["image_id"],
            )
            (fold_dir / f"{role}.txt").write_text(
                "\n".join(row["benchmark_image_path"] for row in rows) + "\n", encoding="utf-8"
            )
            counts = Counter(row["defect_class"] for row in rows)
            summary_rows.append({
                "outer_fold": outer_fold, "role": role, "n_images": len(rows),
                "n_groups": len({row["source_template_group"] for row in rows}),
                **{f"images_{name}": counts[name] for name in CLASS_NAMES[:5]},
            })
            split_rows.extend({
                "outer_fold": outer_fold, "role": role, "image_id": row["image_id"],
                "source_template_group": row["source_template_group"],
                "defect_class": row["defect_class"],
            } for row in rows)
        (fold_dir / "data.yaml").write_text("\n".join([
            f"path: {output / 'dataset'}",
            f"train: {fold_dir / 'train.txt'}",
            f"val: {fold_dir / 'val.txt'}",
            f"test: {fold_dir / 'test.txt'}",
            "names:", *[f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES)], "",
        ]), encoding="utf-8")

    for outer_fold in range(5):
        role_by_group = defaultdict(set)
        for row in split_rows:
            if row["outer_fold"] == outer_fold:
                role_by_group[row["source_template_group"]].add(row["role"])
        if any(len(roles) != 1 for roles in role_by_group.values()):
            raise RuntimeError(f"Source-group leakage in fold {outer_fold}")

    write_csv(output / "source_template_manifest.csv", sorted(manifest, key=lambda row: row["image_id"]))
    write_csv(output / "group_manifest.csv", sorted(group_rows, key=lambda row: int(row["source_template_group"])))
    write_csv(output / "split_manifest.csv", split_rows)
    write_csv(output / "split_summary.csv", summary_rows)
    audit = {
        "protocol": "paired five-fold outer source-group CV with group-disjoint validation",
        "split_seed": SPLIT_SEED,
        "images": 100,
        "boxes": sum(int(row["objects_benchmark"]) for row in manifest),
        "source_template_groups": 50,
        "train_val_test_images_per_fold": [60, 20, 20],
        "groups_per_defect_class": {CLASS_NAMES[i]: len(groups_by_class[i]) for i in PRIMARY_DEFECT_IDS},
        "leakage": False,
        "annotation_source": str(source),
    }
    (output / "split_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
