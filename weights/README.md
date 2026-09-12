# Released checkpoints

This directory contains one full-RFDD-SRD checkpoint for each evaluated architecture, all trained with seed `20260908`. The paper's reported repeated results use three independently trained seeds (`20260908`, `20260909`, and `20260910`); the remaining checkpoints can be regenerated with the supplied training code.

The checkpoint files are tracked with Git LFS because several exceed GitHub's regular per-file limit. Install Git LFS and run `git lfs install` before adding or cloning these files.

See `checkpoint_manifest.csv` for model names, validation-selected operating thresholds, file sizes, and SHA-256 checksums. `predict.py` applies the recorded threshold by default.

 [Download the released model checkpoints](https://aistudio.baidu.com/dataset/detail/397725/file).
