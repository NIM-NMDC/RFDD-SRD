# RFDD-SRD
## Rail Fastener Damage Detection Standard Reference Dataset

<p align="center">
  <img src="docs/cover.png" width="100%">
</p>

<p align="center">
  <b>A standardized benchmark for rail fastener damage detection under unified inspection perspective.</b>
</p>

---

## 📌 Overview

RFDD-SRD is a curated standard reference dataset designed for the evaluation of rail fastener damage detection systems. The dataset is constructed under a unified inspection perspective using industrial railway inspection imagery and focuses on providing reliable and reproducible evaluation samples for intelligent inspection research.

Unlike conventional task-oriented datasets, RFDD-SRD emphasizes:

- standardized data organization,
- controlled category coverage,
- physical and visual consistency,
- high-precision annotation reliability.

The dataset is intended to support objective comparison across different detection algorithms and imaging systems.

---

## ✨ Dataset Characteristics

- Unified railway inspection perspective
- 100 carefully curated samples
- 6 fasteners per image
- 5 representative damage categories:
  - Missing
  - Inverted
  - Displaced
  - Deformed
  - Fractured
- Pixel-accurate annotations
- Multi-annotator cross-validation
- Multi-dimensional consistency evaluation

---

## 📂 Dataset Structure

```text
RFDD-SRD.zip
│
├── images/
│   ├── 000001.png
│   ├── 000002.png
│   └── ...
│
├── labels/
│   ├── 000001.txt
│   ├── 000002.txt
│   └── ...
│
├── classes.txt
└── notes.json
