# RFDD-SRD benchmark code

This repository contains the final code used for the RFDD-SRD benchmark experiments. It reproduces the leakage-aware internal five-fold cross-validation and the independent external real-world evaluation reported in the manuscript. Plotting and manuscript-generation scripts are intentionally excluded.

## Evaluated detectors

The same six-class label space is used by all models:

| Class ID | Fastener state |
|---:|---|
| 0 | Deformed |
| 1 | Displaced |
| 2 | Fractured |
| 3 | Inverted |
| 4 | Missing |
| 5 | Normal |

The benchmark includes six representative detectors:

- Faster R-CNN with ResNet-50 FPN V2
- RetinaNet with ResNet-50 FPN V2
- FCOS with ResNet-50 FPN
- RT-DETR-L
- YOLO11m
- YOLOv10m

Torchvision provides the first three implementations. Ultralytics provides RT-DETR-L, YOLO11m, and YOLOv10m. Every detector is initialized from generic COCO-pretrained weights and then trained on RFDD-SRD.

## Repository contents

| Path | Purpose |
|---|---|
| `audit_dataset.py` | Checks image-label pairing, class IDs, box ranges, counts, dimensions, source groups, and duplicate image hashes. |
| `prepare_group_cv.py` | Builds the source/template-group-disjoint five-fold split and its audit manifests. |
| `train_cv.py` | Trains one model, fold, and seed and saves validation/test predictions. |
| `run_cv.py` | Schedules the complete cross-validation grid over one or more GPUs. |
| `evaluate_cv.py` | Selects thresholds on validation folds and computes out-of-fold metrics and group-bootstrap confidence intervals. |
| `train_full_and_test_real.py` | Trains on all RFDD-SRD images using CV-frozen epochs and thresholds, then evaluates the independent real set. |
| `evaluate_external.py` | Aggregates repeated external-test results and computes source-group bootstrap confidence intervals. |
| `evaluate_external_cv.py` | Shared detection metrics plus optional frozen-CV-checkpoint external evaluation. |
| `predict.py` | Runs inference with any released checkpoint using a unified EXIF-normalized image loader. |
| `results/internal_cv/` | Final internal five-fold metrics. |
| `results/external_real/` | Final EXIF-normalized external real-world metrics. |
| `weights/` | One released full-data checkpoint per architecture and its checksum manifest. |

## Environment

The reported experiments used:

- Python 3.9.23
- PyTorch 2.1.0
- Torchvision 0.16.0
- Ultralytics 8.4.51
- CUDA 12.1

Create a clean environment and install a CUDA-compatible PyTorch build before installing the remaining requirements. For the original environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

GPU training is expected. The code was validated on Linux with NVIDIA GPUs. The bootstrap evaluators use multiprocessing and are most reliable on Linux.

## Dataset layout

RFDD-SRD and the external real dataset use YOLO-format text labels:

```text
RFDD_SRD_ROOT/
├── images/
│   └── <image>.(png|jpg|jpeg|bmp|tif|tiff)
└── labels/
    └── <image>.txt

REAL_ROOT/
├── images/
├── labels/
└── classes.txt        # optional; one class name per line in the order above
```

Each annotation row is `class_id center_x center_y width height`, with coordinates normalized to `[0, 1]`.

The image filename must retain the RFDD-SRD source/template identifier used by `prepare_group_cv.py`. The final release contains 100 images arranged into 50 two-image source/template groups, with ten groups for each damage category.

## 1. Audit the datasets

```bash
python audit_dataset.py \
  --rfdd-root /path/to/RFDD_SRD_ROOT \
  --real-root /path/to/REAL_ROOT \
  --output artifacts/dataset_audit.json
```

The command fails on missing labels and reports malformed or out-of-range boxes. Review the `issues` arrays before training.

## 2. Prepare leakage-aware five-fold splits

```bash
python prepare_group_cv.py \
  --source /path/to/RFDD_SRD_ROOT \
  --output artifacts/cv
```

The default mode creates symbolic links to avoid duplicating the images. Add `--copy-images` on systems where symbolic links are unavailable. Each fold contains 60 training, 20 validation, and 20 test images. All images from the same source/template group remain in one partition within a fold.

## 3. Run internal cross-validation

Run one job:

```bash
python train_cv.py \
  --root artifacts/cv \
  --model yolo11m \
  --fold 0 \
  --seed 20260908 \
  --device 0
```

Run the complete grid across available GPUs:

```bash
python run_cv.py \
  --root artifacts/cv \
  --gpus 0,1,2,3 \
  --models yolo11m,yolov10m,rtdetr_l,fasterrcnn_r50_fpn_v2,retinanet_r50_fpn_v2,fcos_r50_fpn
```

Ultralytics can download its generic pretrained checkpoints automatically. For an offline system, place `yolo11m.pt`, `yolov10m.pt`, and `rtdetr-l.pt` in one directory and pass `--pretrained-dir /path/to/pretrained` to either training command.

After all jobs finish, calculate the reported out-of-fold results:

```bash
python evaluate_cv.py \
  --root artifacts/cv \
  --bootstrap 2000 \
  --bootstrap-seed 20260908
```

The evaluator chooses an operating threshold independently for each model, seed, and outer fold by maximizing macro-F1 on that fold's validation subset. Test-fold predictions remain untouched until final scoring.

## 4. Train on all RFDD-SRD images and test the real set

Prepare the full-data protocol and freeze the median CV thresholds:

```bash
python train_full_and_test_real.py \
  --cv-root artifacts/cv \
  --real-root /path/to/REAL_ROOT \
  --prepare
```

Run every model with each of the three reported seeds. Example:

```bash
python train_full_and_test_real.py \
  --cv-root artifacts/cv \
  --real-root /path/to/REAL_ROOT \
  --model yolo11m \
  --seed 20260908 \
  --device 0
```

The external images are used only after training. They are not used for optimization, epoch selection, checkpoint selection, or threshold selection. All frameworks receive pixels after `PIL.ImageOps.exif_transpose`, ensuring that predictions and annotations share the same coordinate frame.

Aggregate the three-seed external results:

```bash
python evaluate_external.py \
  --cv-root artifacts/cv \
  --real-root /path/to/REAL_ROOT \
  --exp-root artifacts/cv/full_train_real_test \
  --bootstrap 2000 \
  --bootstrap-seed 20260909
```

## 5. Use the released checkpoints

The included files are the full-data checkpoints trained with seed `20260908`. For example:

```bash
python predict.py \
  --model yolo11m \
  --weights weights/yolo11m_seed20260908.pt \
  --images /path/to/images \
  --output artifacts/yolo11m_predictions.json \
  --device 0
```

Replace `--model` and `--weights` with the corresponding entries in `weights/checkpoint_manifest.csv` for the other architectures.
By default, `predict.py` uses the RFDD-SRD-validation operating threshold recorded for that released checkpoint. Pass `--score-threshold` only when a different deployment operating point is required.

## Metrics and statistical protocol

- Predictions are matched to ground truth class-wise using greedy one-to-one matching in descending confidence order.
- A detection is correct when its intersection over union is at least 0.50 and the class label is correct.
- Reported metrics include Precision, Recall, F1, AP50, AP50–95, missed-detection rate, false-discovery rate, and false positives per image.
- AP follows 101-point interpolation. AP50–95 averages IoU thresholds from 0.50 to 0.95 in increments of 0.05.
- Internal 95% confidence intervals use 2,000 defect-stratified source/template-group bootstrap replicates.
- External 95% confidence intervals use 2,000 source/acquisition-group bootstrap replicates.
- Reported point estimates are averaged across the three independent training seeds.

## Released results and weights

The CSV files under `results/` are the final machine-readable values used for the manuscript tables and class-wise analyses. The external files correspond to the corrected evaluation in which every framework uses the same EXIF-normalized input pixels.

Model checkpoints are large binary files and are tracked through Git LFS. Before publishing this directory:

```bash
git lfs install
git lfs track "weights/*.pt"
git add .gitattributes weights/
```

Verify downloaded checkpoints against `weights/checkpoint_manifest.csv` before inference.

## Reproducibility notes

- The three fixed seeds are `20260908`, `20260909`, and `20260910`.
- Deterministic PyTorch and Ultralytics execution is requested. Exact bitwise equality can still depend on GPU model, CUDA, cuDNN, and library builds.
- The released weights provide one ready-to-use checkpoint per architecture. Reproducing the repeated paper results requires running all three seeds.
- The dataset itself is not duplicated in this code package. Use the released RFDD-SRD dataset and preserve its filenames and class order.

If this code or the accompanying dataset is used in published work, please cite the RFDD-SRD data description paper.
