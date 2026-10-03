#!/usr/bin/env python3
"""Run LOSO folds in parallel, one train.py process per GPU.

Each GPU pulls the next unfinished speaker from a shared queue. With
--wandb_artifacts every finished fold is uploaded as a W&B model artifact
(`<run_name>-fold_<spk>`), and folds that already have one are skipped, so a
lost machine only costs the folds in progress. Arguments this script doesn't
recognize are passed through to train.py.

Usage:
  python scripts/run_loso_parallel.py --gpus 0,1 \
    --data_path datasets/Scripts/scripts_fridriksson.json \
    --output_dir /root/checkpoints/scripts-small-taginit \
    --wandb_artifacts --run_name scripts-small-taginit \
    -- --fp16 --batch_size 4 --grad_accum 4 --num_workers 2 --wandb

  # Later, fetch every fold for evaluate_model.py --loso
  python scripts/run_loso_parallel.py --download ... (same --data_path/--output_dir/--run_name)
"""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aphasia_modeling.data.dataset import AphasiaBankDataset

TRAIN_SCRIPT = Path(__file__).resolve().parent / "train.py"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(description="Parallel LOSO training, one fold per GPU")
    p.add_argument("--gpus", type=str, default="0", help="Comma-separated GPU indices")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True,
                    help="Fold checkpoints go to output_dir/fold_<spk>")
    p.add_argument("--speakers", type=str, default=None,
                    help="Comma-separated subset of test speakers (default: all)")
    p.add_argument("--run_name", type=str, default=None,
                    help="Prefix for W&B run and artifact names (default: output_dir name)")
    p.add_argument("--wandb_artifacts", action="store_true", default=False,
                    help="Upload finished folds as W&B artifacts and skip folds already uploaded")
    p.add_argument("--wandb_project", type=str, default="talking-points")
    p.add_argument("--download", action="store_true", default=False,
                    help="Download every fold's artifact into output_dir and exit")
    args, train_args = p.parse_known_args()
    if train_args[:1] == ["--"]:
        train_args = train_args[1:]
    args.run_name = args.run_name or Path(args.output_dir).name
    return args, train_args


def artifact_name(args, spk: str) -> str:
    return f"{args.run_name}-fold_{spk}"


def artifact_path(args, api, spk: str) -> str:
    return f"{api.default_entity}/{args.wandb_project}/{artifact_name(args, spk)}:latest"


def has_artifact(args, api, spk: str) -> bool:
    import wandb
    try:
        api.artifact(artifact_path(args, api, spk))
        return True
    except wandb.errors.CommError:
        return False


def upload_fold(args, fold_dir: Path, spk: str) -> bool:
    cmd = ["wandb", "artifact", "put", str(fold_dir),
           "--name", f"{args.wandb_project}/{artifact_name(args, spk)}", "--type", "model"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[{spk}] artifact upload FAILED:\n{result.stderr[-2000:]}", flush=True)
    return result.returncode == 0


def train_one(args, train_args: list[str], spk: str, gpu: str) -> bool:
    fold_dir = Path(args.output_dir) / f"fold_{spk}"
    log_path = Path(args.output_dir) / "logs" / f"fold_{spk}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if not (fold_dir / "train_metrics.json").exists():
        cmd = [sys.executable, str(TRAIN_SCRIPT), "--data_path", args.data_path,
               "--output_dir", str(fold_dir), "--test_speaker", spk, *train_args]
        if "--wandb" in train_args and "--wandb_run_name" not in train_args:
            cmd += ["--wandb_run_name", f"{args.run_name}-{spk}"]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
        print(f"[{spk}] training on GPU {gpu}, log: {log_path}", flush=True)
        with open(log_path, "a") as log:
            returncode = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        if returncode != 0 or not (fold_dir / "train_metrics.json").exists():
            print(f"[{spk}] FAILED (exit {returncode}), see {log_path}", flush=True)
            return False

    if args.wandb_artifacts and not upload_fold(args, fold_dir, spk):
        return False
    print(f"[{spk}] done", flush=True)
    return True


def main():
    args, train_args = parse_args()
    speakers = AphasiaBankDataset.load(args.data_path).speakers
    if args.speakers:
        speakers = [s for s in speakers if s in args.speakers.split(",")]

    api = None
    if args.wandb_artifacts or args.download:
        import wandb
        api = wandb.Api()

    if args.download:
        for spk in speakers:
            fold_dir = Path(args.output_dir) / f"fold_{spk}"
            if (fold_dir / "train_metrics.json").exists():
                continue
            print(f"[{spk}] downloading {artifact_path(args, api, spk)}", flush=True)
            api.artifact(artifact_path(args, api, spk)).download(root=str(fold_dir))
        return

    todo = queue.Queue()
    for spk in speakers:
        if api is not None and has_artifact(args, api, spk):
            print(f"[{spk}] already uploaded, skipping", flush=True)
        else:
            todo.put(spk)

    failed = []

    def worker(gpu: str):
        while True:
            try:
                spk = todo.get_nowait()
            except queue.Empty:
                return
            if not train_one(args, train_args, spk, gpu):
                failed.append(spk)

    threads = [threading.Thread(target=worker, args=(g,)) for g in args.gpus.split(",")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if failed:
        print(f"Failed folds: {failed}", flush=True)
        sys.exit(1)
    print("All folds complete", flush=True)


if __name__ == "__main__":
    main()
