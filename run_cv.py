#!/usr/bin/env python3
"""Run all group-CV training jobs with one process per available GPU."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import deque
from pathlib import Path


ROOT = Path("artifacts/cv")
MODELS = [
    "yolo11m",
    "rtdetr_l",
    "fasterrcnn_r50_fpn_v2",
    "yolov10m",
    "retinanet_r50_fpn_v2",
    "fcos_r50_fpn",
]
SEEDS = [20260908, 20260909, 20260910]


def main() -> None:
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Experiment root containing splits/, runs/, and prepared dataset links/copies.",
    )
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--pretrained-dir", type=Path, default=None)
    args = parser.parse_args()
    ROOT = args.root.resolve()
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    models = [value for value in args.models.split(",") if value.strip()]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    worker_script = Path(__file__).resolve().with_name("train_cv.py")

    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    tasks = deque(
        (model, fold, seed)
        for model in models
        for seed in seeds
        for fold in range(5)
        if not (ROOT / "runs" / model / f"fold_{fold}_seed_{seed}" / "completed.json").is_file()
    )
    running: dict[int, tuple[subprocess.Popen, object, tuple[str, int, int], Path]] = {}
    failures = []
    expected = len(models) * len(seeds) * 5
    completed = expected - len(tasks)
    started_at = time.time()

    while tasks or running:
        for gpu in gpus:
            if gpu in running or not tasks:
                continue
            model, fold, seed = tasks.popleft()
            log_path = logs / f"{model}_fold{fold}_seed{seed}.log"
            log_stream = log_path.open("a", encoding="utf-8")
            command = [
                sys.executable,
                str(worker_script),
                "--root",
                str(ROOT),
                "--model",
                model,
                "--fold",
                str(fold),
                "--seed",
                str(seed),
                "--device",
                str(gpu),
            ]
            if args.pretrained_dir is not None:
                command.extend(["--pretrained-dir", str(args.pretrained_dir.resolve())])
            process = subprocess.Popen(
                command,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                cwd=str(Path(__file__).resolve().parent),
            )
            running[gpu] = (process, log_stream, (model, fold, seed), log_path)
            print(f"START gpu={gpu} model={model} fold={fold} seed={seed}", flush=True)

        time.sleep(10)
        for gpu in list(running):
            process, log_stream, task, log_path = running[gpu]
            return_code = process.poll()
            if return_code is None:
                continue
            log_stream.close()
            model, fold, seed = task
            if return_code == 0:
                completed += 1
                print(
                    f"DONE {completed} gpu={gpu} model={model} fold={fold} seed={seed}",
                    flush=True,
                )
            else:
                failures.append(
                    {
                        "gpu": gpu,
                        "model": model,
                        "fold": fold,
                        "seed": seed,
                        "return_code": return_code,
                        "log": str(log_path),
                    }
                )
                print(
                    f"FAIL gpu={gpu} model={model} fold={fold} seed={seed} rc={return_code}",
                    flush=True,
                )
            del running[gpu]

        status = {
            "completed": completed,
            "pending": len(tasks),
            "running": [
                {"gpu": gpu, "model": task[0], "fold": task[1], "seed": task[2]}
                for gpu, (_process, _stream, task, _path) in running.items()
            ],
            "failures": failures,
            "elapsed_seconds": time.time() - started_at,
        }
        (ROOT / "queue_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")

    if failures:
        raise RuntimeError(f"{len(failures)} jobs failed; see queue_status.json")
    print("ALL JOBS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
