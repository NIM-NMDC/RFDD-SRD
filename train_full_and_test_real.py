#!/usr/bin/env python3
"""Train final detectors on all 100 RFDD-SRD images and infer on 100 real images.

The real dataset is never used for training, validation, early stopping, checkpoint
selection, or threshold selection. Epoch counts and operating thresholds are frozen
from the completed RFDD-SRD group cross-validation experiment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import time
from functools import partial
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


CV_ROOT = Path("artifacts/cv")
EXP_ROOT = CV_ROOT / "full_train_real_test"
TRAIN_IMAGE_ROOT = CV_ROOT / "dataset" / "images"
REAL_ROOT = Path("data/real")
CLASS_NAMES = ["Deformed", "Displaced", "Fractured", "Inverted", "Missing", "Normal"]
SEEDS = [20260908, 20260909, 20260910]
EPOCHS = {
    "fasterrcnn_r50_fpn_v2": 13,
    "retinanet_r50_fpn_v2": 19,
    "fcos_r50_fpn": 20,
    "rtdetr_l": 100,
    "yolo11m": 100,
    "yolov10m": 100,
}
GENERIC_WEIGHTS: dict[str, str | Path] = {
    "yolo11m": "yolo11m.pt",
    "yolov10m": "yolov10m.pt",
    "rtdetr_l": "rtdetr-l.pt",
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def image_paths(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def load_rgb(path: Path) -> Image.Image:
    """Apply EXIF orientation before all framework-specific preprocessing."""
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB").copy()


def prepare() -> None:
    EXP_ROOT.mkdir(parents=True, exist_ok=True)
    paths = image_paths(TRAIN_IMAGE_ROOT)
    if len(paths) != 100:
        raise RuntimeError(f"Expected 100 RFDD-SRD images, found {len(paths)}")
    training_boxes = 0
    for path in paths:
        label = CV_ROOT / "dataset" / "labels" / f"{path.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(label)
        training_boxes += sum(1 for line in label.read_text(encoding="utf-8").splitlines() if line.strip())
    real_paths = image_paths(REAL_ROOT / "images")
    if len(real_paths) != 100:
        raise RuntimeError(f"Expected 100 real images, found {len(real_paths)}")
    train_list = EXP_ROOT / "all_100_train.txt"
    train_list.write_text("\n".join(str(path) for path in paths) + "\n", encoding="utf-8")
    yaml_text = (
        f"path: {CV_ROOT / 'dataset'}\n"
        f"train: {train_list}\n"
        f"val: {train_list}\n"
        "names:\n"
        + "".join(f"  {index}: {name}\n" for index, name in enumerate(CLASS_NAMES))
    )
    (EXP_ROOT / "data.yaml").write_text(yaml_text, encoding="utf-8")

    threshold_rows = []
    with (CV_ROOT / "results" / "fold_thresholds.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        source_rows = list(csv.DictReader(stream))
    for model in EPOCHS:
        for seed in SEEDS:
            values = [
                float(row["operating_threshold"])
                for row in source_rows
                if row["model"] == model and int(row["seed"]) == seed
            ]
            if len(values) != 5:
                raise RuntimeError(f"Expected five CV thresholds for {model}/{seed}, got {len(values)}")
            threshold_rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "operating_threshold": float(np.median(values)),
                    "source": "median of five RFDD-SRD validation-fold thresholds",
                }
            )
    with (EXP_ROOT / "frozen_thresholds.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(threshold_rows[0]))
        writer.writeheader()
        writer.writerows(threshold_rows)
    protocol = {
        "training_images": 100,
        "training_boxes": training_boxes,
        "external_test_images": 100,
        "seeds": SEEDS,
        "epochs": EPOCHS,
        "initialization": "generic COCO-pretrained weights",
        "checkpoint": "final epoch; no real-data model selection",
        "thresholds": "per-model/per-seed median of five validation-fold thresholds from prior RFDD-SRD group CV",
        "external_test_role": "testing only; never used for optimization or selection",
    }
    (EXP_ROOT / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(protocol, ensure_ascii=False, indent=2), flush=True)


def load_train_paths() -> list[Path]:
    return [
        Path(line.strip())
        for line in (EXP_ROOT / "all_100_train.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class TrainingDataset:
    def __init__(self, paths: list[Path]):
        import torch

        self.paths = paths
        self.torch = torch

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        import torchvision.transforms.functional as functional

        path = self.paths[index]
        image = load_rgb(path)
        width, height = image.size
        label_path = CV_ROOT / "dataset" / "labels" / f"{path.stem}.txt"
        boxes, labels = [], []
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
            labels.append(int(class_id) + 1)
        target = {
            "boxes": self.torch.as_tensor(boxes, dtype=self.torch.float32),
            "labels": self.torch.as_tensor(labels, dtype=self.torch.int64),
            "image_id": self.torch.tensor([index], dtype=self.torch.int64),
        }
        return functional.to_tensor(image), target, path.name


class RealDataset:
    def __init__(self):
        self.paths = image_paths(REAL_ROOT / "images")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        import torchvision.transforms.functional as functional

        path = self.paths[index]
        return functional.to_tensor(load_rgb(path)), path.name


def collate(batch):
    return tuple(zip(*batch))


def predict_torchvision(model, loader, device: int) -> list[dict]:
    import torch

    model.eval()
    rows = []
    with torch.no_grad():
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


def run_torchvision(model_name: str, seed: int, device: int, run_dir: Path) -> dict:
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

    generator = torch.Generator().manual_seed(seed)
    train_batch_size = 1 if model_name == "retinanet_r50_fpn_v2" else 2
    train_loader = DataLoader(
        TrainingDataset(load_train_paths()),
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate,
        generator=generator,
        persistent_workers=True,
    )
    real_loader = DataLoader(
        RealDataset(), batch_size=1, shuffle=False, num_workers=4, collate_fn=collate
    )
    if model_name == "fasterrcnn_r50_fpn_v2":
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn_v2(
            weights=weights, min_size=1024, max_size=1333,
            box_score_thresh=0.001, box_nms_thresh=0.70, box_detections_per_img=100,
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(CLASS_NAMES) + 1)
        family, learning_rate, use_amp = "Faster R-CNN", 0.005, True
    elif model_name == "retinanet_r50_fpn_v2":
        weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
        model = retinanet_resnet50_fpn_v2(
            weights=weights, min_size=1024, max_size=1333,
            score_thresh=0.001, nms_thresh=0.70, detections_per_img=100,
        )
        model.head.classification_head = RetinaNetClassificationHead(
            model.backbone.out_channels,
            model.head.classification_head.num_anchors,
            len(CLASS_NAMES) + 1,
            norm_layer=partial(nn.GroupNorm, 32),
        )
        # Full precision avoids a rare CUDA illegal-memory-access failure observed
        # with RetinaNet's focal-loss kernel on this V100/Torchvision combination.
        family, learning_rate, use_amp = "RetinaNet", 0.005, False
    elif model_name == "fcos_r50_fpn":
        weights = FCOS_ResNet50_FPN_Weights.DEFAULT
        model = fcos_resnet50_fpn(
            weights=weights, min_size=1024, max_size=1333,
            score_thresh=0.001, nms_thresh=0.70, detections_per_img=100,
        )
        model.head.classification_head = FCOSClassificationHead(
            model.backbone.out_channels,
            model.head.classification_head.num_anchors,
            len(CLASS_NAMES) + 1,
            norm_layer=partial(nn.GroupNorm, 32),
        )
        family, learning_rate, use_amp = "FCOS", 0.001, False
    else:
        raise ValueError(model_name)
    model.to(device)
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate, momentum=0.9, weight_decay=0.0005,
    )
    milestones = [value for value in (12, 20) if value < EPOCHS[model_name]]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.2)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    history, start = [], time.time()
    for epoch in range(EPOCHS[model_name]):
        model.train()
        losses = []
        for images, targets, _names in train_loader:
            images = [image.to(device) for image in images]
            targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss_items = model(images, targets)
                loss = sum(loss_items.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss: {model_name}/{seed}/{epoch + 1}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                        "learning_rate": optimizer.param_groups[0]["lr"]})
        print(f"{model_name} seed={seed} epoch={epoch + 1}/{EPOCHS[model_name]} "
              f"loss={history[-1]['train_loss']:.5f}", flush=True)

    final_path = run_dir / "final.pt"
    torch.save(model.state_dict(), final_path)
    infer_start = time.time()
    predictions = predict_torchvision(model, real_loader, device)
    infer_seconds = time.time() - infer_start
    checkpoint_dir = Path(torch.hub.get_dir()) / "checkpoints"
    generic_path = checkpoint_dir / Path(weights.url).name
    metadata = {
        "model": model_name, "model_family": family, "seed": seed, "device": device,
        "training_images": 100, "external_test_images": 100,
        "rfdd_srd_training_data_used": True,
        "generic_pretraining": "COCO", "generic_weight": str(generic_path),
        "generic_weight_sha256": sha256(generic_path),
        "epochs_requested": EPOCHS[model_name], "epochs_completed": EPOCHS[model_name],
        "checkpoint_selection": "final epoch; no validation or external-test selection",
        "final_checkpoint": str(final_path), "final_checkpoint_sha256": sha256(final_path),
        "train_seconds": time.time() - start, "inference_seconds": infer_seconds,
        "learning_rate": learning_rate, "batch_size": train_batch_size,
        "mixed_precision": use_amp, "history": history,
        "software": {"python": platform.python_version(), "torch": torch.__version__,
                     "torchvision": torchvision.__version__, "cuda": torch.version.cuda},
    }
    payload = {**metadata, "split": "external_real_test", "images": predictions}
    (run_dir / "predictions_real.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def serialize_ultralytics(result, image_name: str) -> dict:
    detections = []
    if result.boxes is not None:
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        detections = [
            {"class_id": int(class_id), "score": float(score),
             "xyxy": [float(value) for value in box]}
            for box, score, class_id in zip(boxes, scores, classes)
        ]
    return {"image": image_name, "detections": detections}


def run_ultralytics(model_name: str, seed: int, device: int, run_dir: Path) -> dict:
    import torch
    import ultralytics
    from ultralytics import RTDETR, YOLO

    model_class = RTDETR if model_name == "rtdetr_l" else YOLO
    generic_path = GENERIC_WEIGHTS[model_name]
    model = model_class(str(generic_path))
    batch = 2 if model_name == "rtdetr_l" else 4
    start = time.time()
    model.train(
        data=str(EXP_ROOT / "data.yaml"), epochs=EPOCHS[model_name], patience=0,
        imgsz=1280, batch=batch, device=device, workers=4,
        project=str(run_dir.parent), name=run_dir.name, exist_ok=True,
        pretrained=True, optimizer="AdamW", lr0=0.001, weight_decay=0.0005,
        # Disable Ultralytics' external AMP compatibility probe. This also keeps
        # training self-contained on the offline experiment server.
        seed=seed, deterministic=True, amp=False, close_mosaic=10,
        plots=False, verbose=False, val=False, save=True,
    )
    final_path = (run_dir / "weights" / "last.pt").resolve()
    if not final_path.is_file():
        raise FileNotFoundError(final_path)
    trained = model_class(str(final_path))
    predictions = []
    infer_start = time.time()
    for path in image_paths(REAL_ROOT / "images"):
        result = trained.predict(
            source=load_rgb(path), imgsz=1280, conf=0.001, iou=0.70,
            max_det=100, device=device, verbose=False,
        )[0]
        predictions.append(serialize_ultralytics(result, path.name))
    infer_seconds = time.time() - infer_start
    metadata = {
        "model": model_name,
        "model_family": "RT-DETR" if model_name == "rtdetr_l" else "YOLO",
        "seed": seed, "device": device, "training_images": 100,
        "rfdd_srd_training_data_used": True,
        "external_test_images": 100, "generic_pretraining": "COCO",
        "generic_weight": str(generic_path),
        "generic_weight_sha256": sha256(Path(generic_path)) if Path(generic_path).is_file() else None,
        "epochs_requested": EPOCHS[model_name],
        "epochs_completed": int(model.trainer.epoch + 1),
        "checkpoint_selection": "final epoch; no validation or external-test selection",
        "final_checkpoint": str(final_path), "final_checkpoint_sha256": sha256(final_path),
        "train_seconds": time.time() - start, "inference_seconds": infer_seconds,
        "software": {"python": platform.python_version(), "torch": torch.__version__,
                     "ultralytics": ultralytics.__version__, "cuda": torch.version.cuda},
    }
    payload = {**metadata, "split": "external_real_test", "images": predictions}
    (run_dir / "predictions_real.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def main() -> None:
    global CV_ROOT, EXP_ROOT, TRAIN_IMAGE_ROOT, REAL_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cv-root",
        type=Path,
        default=CV_ROOT,
        help="Cross-validation experiment root containing dataset/ and results/.",
    )
    parser.add_argument("--real-root", type=Path, default=REAL_ROOT)
    parser.add_argument(
        "--pretrained-dir",
        type=Path,
        default=None,
        help="Optional directory containing yolo11m.pt, yolov10m.pt, and rtdetr-l.pt.",
    )
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--model", choices=list(EPOCHS))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--device", type=int)
    args = parser.parse_args()
    CV_ROOT = args.cv_root.resolve()
    EXP_ROOT = CV_ROOT / "full_train_real_test"
    TRAIN_IMAGE_ROOT = CV_ROOT / "dataset" / "images"
    REAL_ROOT = args.real_root.resolve()
    if args.pretrained_dir is not None:
        pretrained_dir = args.pretrained_dir.resolve()
        for model_name, filename in {
            "yolo11m": "yolo11m.pt",
            "yolov10m": "yolov10m.pt",
            "rtdetr_l": "rtdetr-l.pt",
        }.items():
            GENERIC_WEIGHTS[model_name] = pretrained_dir / filename
    if args.prepare:
        prepare()
        return
    if args.model is None or args.seed is None or args.device is None:
        parser.error("--model, --seed, and --device are required for training")
    if not (EXP_ROOT / "protocol.json").is_file():
        prepare()
    run_dir = EXP_ROOT / "runs" / args.model / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    completed = run_dir / "completed.json"
    if completed.is_file():
        print(f"Already complete: {run_dir}")
        return
    set_seed(args.seed)
    if args.model in GENERIC_WEIGHTS:
        metadata = run_ultralytics(args.model, args.seed, args.device, run_dir)
    else:
        metadata = run_torchvision(args.model, args.seed, args.device, run_dir)
    completed.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Completed {args.model} seed={args.seed}", flush=True)


if __name__ == "__main__":
    main()
