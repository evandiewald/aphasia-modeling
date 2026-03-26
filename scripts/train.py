#!/usr/bin/env python3
"""Training script for Whisper paraphasia detection.

Two training phases:
  Phase 1 (optional): ASR adaptation on AphasiaBank Protocol data
  Phase 2: Paraphasia-aware fine-tuning on Fridriksson subset

Usage:
  # Phase 1: ASR adaptation
  python scripts/train.py \
    --phase 1 \
    --data_path data/protocol.json \
    --output_dir checkpoints/phase1 \
    --model_name openai/whisper-small

  # Phase 2: Paraphasia fine-tuning
  python scripts/train.py \
    --phase 2 \
    --data_path data/fridriksson.json \
    --output_dir checkpoints/phase2 \
    --model_name checkpoints/phase1  # or openai/whisper-small if skipping Phase 1 \
    --test_speaker speaker_01 \
    --class_weights

  # Full LOSO cross-validation
  python scripts/train.py \
    --phase 2 \
    --data_path data/fridriksson.json \
    --output_dir checkpoints/loso \
    --model_name checkpoints/phase1 \
    --loso \
    --class_weights

Designed to run on Lambda Labs GPU instances (A100 40GB+).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Load .env file if present (for WANDB_API_KEY, etc.)
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())
from transformers import (
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    EarlyStoppingCallback,
)

# Add project root to path when running as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aphasia_modeling.data.dataset import AphasiaBankDataset
from aphasia_modeling.model.tokenizer import build_tokenizer, get_paraphasia_token_ids
from aphasia_modeling.model.whisper import (
    build_model,
    WhisperParaphasiaConfig,
    get_class_weight_tensor,
    freeze_encoder,
)
from aphasia_modeling.model.collator import ParaphasiaDataCollator
from aphasia_modeling.model.trainer import ParaphasiaTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Whisper paraphasia model")

    # Data
    p.add_argument("--data_path", type=str, required=True,
                    help="Path to preprocessed dataset JSON")
    p.add_argument("--audio_dir", type=str, default=None,
                    help="Directory containing audio .wav files")

    # Model
    p.add_argument("--model_name", type=str, default="openai/whisper-small",
                    help="HuggingFace model ID or local checkpoint path")
    p.add_argument("--phase", type=int, choices=[1, 2], default=2,
                    help="Training phase: 1=ASR adaptation, 2=paraphasia fine-tuning")

    # Training
    p.add_argument("--output_dir", type=str, default="checkpoints/run",
                    help="Output directory for checkpoints")
    p.add_argument("--epochs", type=int, default=20,
                    help="Number of training epochs")
    p.add_argument("--max_steps", type=int, default=-1,
                    help="Max training steps (overrides epochs if > 0)")
    p.add_argument("--lr", type=float, default=1e-5,
                    help="Learning rate")
    p.add_argument("--batch_size", type=int, default=4,
                    help="Per-device batch size")
    p.add_argument("--grad_accum", type=int, default=4,
                    help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    p.add_argument("--warmup_steps", type=int, default=500,
                    help="Number of warmup steps")
    p.add_argument("--fp16", action="store_true", default=False,
                    help="Use FP16 mixed precision")
    p.add_argument("--bf16", action="store_true", default=False,
                    help="Use BF16 mixed precision (A100+)")

    # Paraphasia-specific
    p.add_argument("--class_weights", action="store_true", default=False,
                    help="Apply class weighting ([p]=10, [n]=20)")
    p.add_argument("--oversample", type=int, default=1,
                    help="Oversample paraphasia utterances N times (e.g., 4 = 4x)")
    p.add_argument("--logit_bias", type=float, default=0.0,
                    help="Additive bias on paraphasia token logits (e.g., 5.0)")
    p.add_argument("--freeze_encoder", action="store_true", default=False,
                    help="Freeze encoder during training")
    p.add_argument("--time_perturbation", action="store_true", default=False,
                    help="Apply SpecAugment time perturbation")

    # LOSO cross-validation
    p.add_argument("--loso", action="store_true", default=False,
                    help="Run full LOSO cross-validation (all folds)")
    p.add_argument("--test_speaker", type=str, default=None,
                    help="Single test speaker for one fold (used if --loso is not set)")

    # Early stopping
    p.add_argument("--early_stopping", type=int, default=5,
                    help="Early stopping patience (0 to disable)")

    # Performance
    p.add_argument("--num_workers", type=int, default=0,
                    help="Dataloader workers (0 for macOS, 4+ for Linux/GPU)")

    # Logging
    p.add_argument("--wandb", action="store_true", default=False,
                    help="Log to Weights & Biases")
    p.add_argument("--wandb_project", type=str, default="talking-points",
                    help="W&B project name")
    p.add_argument("--wandb_run_name", type=str, default=None,
                    help="W&B run name (auto-generated if not set)")

    return p.parse_args()


def train_fold(
    args: argparse.Namespace,
    dataset: AphasiaBankDataset,
    test_speaker: str,
    output_dir: str,
) -> dict:
    """Train a single LOSO fold."""
    print(f"\n{'='*60}")
    print(f"Training fold: test_speaker={test_speaker}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    # Split data
    train_utts, dev_utts, test_utts = dataset.loso_split(test_speaker)
    print(f"Train: {len(train_utts)} utterances")
    print(f"Dev:   {len(dev_utts)} utterances")
    print(f"Test:  {len(test_utts)} utterances")

    # Build tokenizer and model
    tokenizer = build_tokenizer(model_name=args.model_name)

    config = WhisperParaphasiaConfig(
        model_name=args.model_name,
        freeze_encoder=args.freeze_encoder,
    )
    model, tokenizer = build_model(config, tokenizer)

    # Feature extractor
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_name)

    # Data collator
    collator = ParaphasiaDataCollator(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        apply_time_perturbation=args.time_perturbation,
    )

    # Convert to HF datasets (oversample paraphasia utterances in train only)
    train_ds = dataset.to_hf_dataset(train_utts, oversample_paraphasia=args.oversample)
    dev_ds = dataset.to_hf_dataset(dev_utts)

    # Class weights
    class_weights = None
    if args.class_weights:
        class_weights = get_class_weight_tensor(tokenizer)

    # Logit bias for paraphasia tokens
    logit_bias = None
    if args.logit_bias > 0:
        para_ids = get_paraphasia_token_ids(tokenizer)
        logit_bias = {tid: args.logit_bias for tid in para_ids.values()}

    # Wandb setup
    report_to = "none"
    run_name = None
    if args.wandb:
        import wandb
        report_to = "wandb"
        from datetime import datetime
        ts = datetime.now().strftime("%m%d-%H%M")
        run_name = args.wandb_run_name or f"fold-{test_speaker}-{ts}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "model_name": args.model_name,
                "phase": args.phase,
                "test_speaker": test_speaker,
                "train_size": len(train_utts),
                "dev_size": len(dev_utts),
                "test_size": len(test_utts),
                "lr": args.lr,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "class_weights": args.class_weights,
                "oversample": args.oversample,
                "logit_bias": args.logit_bias,
                "freeze_encoder": args.freeze_encoder,
                "time_perturbation": args.time_perturbation,
            },
            reinit=True,
        )

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        fp16=args.fp16,
        bf16=args.bf16,
        eval_strategy="steps" if args.max_steps > 0 else "epoch",
        eval_steps=args.max_steps if args.max_steps > 0 else None,
        save_strategy="steps" if args.max_steps > 0 else "epoch",
        save_steps=args.max_steps if args.max_steps > 0 else None,
        save_total_limit=3,
        load_best_model_at_end=False if args.max_steps > 0 else True,
        metric_for_best_model="eval_loss" if args.max_steps <= 0 else None,
        greater_is_better=False if args.max_steps <= 0 else None,
        predict_with_generate=True,
        generation_max_length=448,
        logging_steps=10 if args.max_steps > 0 else 50,
        report_to=report_to,
        run_name=run_name,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
    )

    # Callbacks
    callbacks = []
    if args.early_stopping > 0:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping,
        ))

    # Trainer
    trainer = ParaphasiaTrainer(
        class_weights=class_weights,
        paraphasia_logit_bias=logit_bias,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    # Train
    train_result = trainer.train()

    # Save
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    feature_extractor.save_pretrained(output_dir)

    metrics = train_result.metrics
    metrics["test_speaker"] = test_speaker
    metrics["train_size"] = len(train_utts)
    metrics["dev_size"] = len(dev_utts)
    metrics["test_size"] = len(test_utts)

    # Save metrics
    metrics_path = Path(output_dir) / "train_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"\nFold complete. Loss: {metrics.get('train_loss', 'N/A'):.4f}")

    # Finish wandb run so each fold gets its own run
    if args.wandb:
        import wandb
        wandb.finish()

    return metrics


def main():
    args = parse_args()

    # Load preprocessed dataset
    print(f"Loading dataset from {args.data_path}...")
    dataset = AphasiaBankDataset.load(args.data_path)
    print(f"Loaded {len(dataset.utterances)} utterances, "
          f"{dataset.num_speakers} speakers: {dataset.speakers}")

    if args.phase == 1:
        # Phase 1: ASR adaptation (no LOSO, use all data with a held-out dev set)
        print("\n--- Phase 1: ASR Adaptation ---")
        print("Using all data with 10% dev hold-out from each speaker")

        # For Phase 1, we don't do LOSO — just hold out a small dev set
        # Use first speaker as a dummy test speaker, but we only care about train/dev
        if not dataset.speakers:
            print("Error: no speakers found in dataset")
            sys.exit(1)

        train_fold(args, dataset, dataset.speakers[0], args.output_dir)

    elif args.phase == 2:
        # Phase 2: Paraphasia fine-tuning
        print("\n--- Phase 2: Paraphasia Fine-tuning ---")

        if args.loso:
            # Full LOSO cross-validation
            all_metrics = []
            for spk in dataset.speakers:
                fold_dir = str(Path(args.output_dir) / f"fold_{spk}")
                metrics = train_fold(args, dataset, spk, fold_dir)
                all_metrics.append(metrics)

            # Save aggregate metrics
            agg_path = Path(args.output_dir) / "all_fold_metrics.json"
            agg_path.write_text(json.dumps(all_metrics, indent=2))
            print(f"\nAll {len(all_metrics)} folds complete.")
            print(f"Aggregate metrics saved to {agg_path}")

        elif args.test_speaker:
            train_fold(args, dataset, args.test_speaker, args.output_dir)

        else:
            print("Error: Phase 2 requires --loso or --test_speaker")
            sys.exit(1)


if __name__ == "__main__":
    main()
