#!/usr/bin/env python3
"""Audit RFDD-SRD and, optionally, an independent real-world test dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
CLASSES = ["Deformed", "Displaced", "Fractured", "Inverted", "Missing", "Normal"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(root: Path, synthetic: bool) -> dict:
    images = sorted(p for p in (root / "images").iterdir() if p.suffix.lower() in SUFFIXES)
    labels = {p.stem: p for p in (root / "labels").glob("*.txt")}
    if {p.stem for p in images} != set(labels):
        raise RuntimeError(f"Image/label stem mismatch in {root}")
    class_counts = Counter()
    box_counts = Counter()
    dimensions = Counter()
    source_groups = defaultdict(list)
    issues = []
    image_hashes = {}
    for image_path in images:
        with Image.open(image_path) as image:
            dimensions[f"{image.width}x{image.height}"] += 1
        image_hashes[image_path.name] = sha256(image_path)
        if synthetic:
            match = re.match(r"^[^-]+-(\d+)-", image_path.stem)
            if not match:
                issues.append(f"Cannot parse source template: {image_path.name}")
                group = image_path.stem
            else:
                group = match.group(1)
        else:
            descriptor = image_path.stem.split("-", 1)[-1]
            normal = re.search(r"正常(\d+)$", descriptor)
            group = f"normal_{normal.group(1)}" if normal else re.sub(r"2$", "", descriptor)
        source_groups[group].append(image_path.name)
        rows = labels[image_path.stem].read_text(encoding="utf-8").splitlines()
        box_counts[len(rows)] += 1
        for line_no, line in enumerate(rows, 1):
            values = line.split()
            if len(values) != 5:
                issues.append(f"Malformed {labels[image_path.stem]}:{line_no}")
                continue
            class_value, cx, cy, width, height = map(float, values)
            class_id = int(class_value)
            if class_value != class_id or not 0 <= class_id < 6:
                issues.append(f"Invalid class {labels[image_path.stem]}:{line_no}")
            if width <= 0 or height <= 0:
                issues.append(f"Non-positive box {labels[image_path.stem]}:{line_no}")
            if not all(0 <= value <= 1 for value in (cx, cy, width, height)):
                issues.append(f"Out-of-range value {labels[image_path.stem]}:{line_no}")
            if cx - width / 2 < -1e-6 or cy - height / 2 < -1e-6 or cx + width / 2 > 1 + 1e-6 or cy + height / 2 > 1 + 1e-6:
                issues.append(f"Box crosses image boundary {labels[image_path.stem]}:{line_no}")
            class_counts[class_id] += 1
    return {
        "root": str(root),
        "images": len(images),
        "labels": len(labels),
        "boxes": sum(class_counts.values()),
        "class_counts": {CLASSES[index]: class_counts[index] for index in range(6)},
        "dimensions": dict(dimensions),
        "boxes_per_image": dict(box_counts),
        "source_groups": len(source_groups),
        "source_group_sizes": dict(Counter(len(names) for names in source_groups.values())),
        "duplicate_image_hashes": len(image_hashes) - len(set(image_hashes.values())),
        "issues": issues,
        "image_hashes": image_hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rfdd-root", type=Path, required=True)
    parser.add_argument("--real-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    payload = {"rfdd_srd": audit(args.rfdd_root.resolve(), True)}
    if args.real_root is not None:
        payload["external_real"] = audit(args.real_root.resolve(), False)
    for result in payload.values():
        result.pop("image_hashes")
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
