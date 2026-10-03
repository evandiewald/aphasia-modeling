#!/usr/bin/env python3
"""Train Whisper single-seq paraphasia models with LOSO cross-validation.

The decoder learns to emit [p]/[n]/[s] inline after paraphasic words,
matching CHAI's single-seq setup. Each fold holds out one speaker as test
and 10% of every other speaker's utterances as dev (CHAI's partitioning).

Usage:
  # One fold
  python scripts/train.py \
    --data_path datasets/Scripts/scripts_fridriksson.json \
    --output_dir checkpoints/scripts-small \
    --test_speaker P1

  # All folds (one checkpoint per fold under output_dir/fold_<spk>)
  python scripts/train.py \
    --data_path datasets/Scripts/scripts_fridriksson.json \
    --output_dir checkpoints/scripts-small \
    --loso

  # Train on [p]/[n] only, leaving [s] to a separate stage
  python scripts/train.py ... --tag_classes pn

Interrupted runs pick up where they left off when re-run with the same
arguments: --loso skips folds that already have train_metrics.json, and a
fold in progress resumes from its latest epoch checkpoint (--no-resume to
start the fold over).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
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
    EarlyStoppingCallback,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
)
from transformers.trainer_utils import get_last_checkpoint

# Add project root to path when running as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aphasia_modeling.data.dataset import AphasiaBankDataset
from aphasia_modeling.model.collator import ParaphasiaDataCollator
from aphasia_modeling.model.tokenizer import get_paraphasia_token_ids
from aphasia_modeling.model.trainer import (
    ParaphasiaTrainer,
    make_tag_logits_preprocessor,
    make_tag_metrics,
)
from aphasia_modeling.model.whisper import (
    WhisperParaphasiaConfig,
    build_model,
    get_class_weight_tensor,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Whisper paraphasia model")

    # Data
    p.add_argument("--data_path", type=str, required=True,
                    help="Path to preprocessed dataset JSON")
    p.add_argument("--max_duration", type=float, default=30.0,
                    help="Drop train/dev utterances longer than this many seconds (0 = keep all)")

    # Model
    p.add_argument("--model_name", type=str, default="openai/whisper-small",
                    help="HuggingFace model ID or local checkpoint path")
    p.add_argument("--freeze_encoder", action="store_true", default=False,
                    help="Freeze encoder during training")
    p.add_argument("--gradient_checkpointing", action="store_true", default=False,
                    help="Trade compute for memory (useful for whisper-large)")

    # Training
    p.add_argument("--output_dir", type=str, default="checkpoints/run",
                    help="Output directory for checkpoints")
    p.add_argument("--epochs", type=int, default=20,
                    help="Number of training epochs")
    p.add_argument("--max_steps", type=int, default=-1,
                    help="Stop after this many steps, evaluating/saving once at the end (smoke tests)")
    p.add_argument("--lr", type=float, default=1e-5,
                    help="Learning rate")
    p.add_argument("--batch_size", type=int, default=8,
                    help="Per-device batch size")
    p.add_argument("--grad_accum", type=int, default=2,
                    help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    p.add_argument("--warmup_ratio", type=float, default=0.1,
                    help="Fraction of steps used for LR warmup")
    p.add_argument("--fp16", action="store_true", default=False,
                    help="Use FP16 mixed precision")
    p.add_argument("--bf16", action="store_true", default=False,
                    help="Use BF16 mixed precision (A100+)")
    p.add_argument("--seed", type=int, default=42,
                    help="Training seed")

    # Paraphasia-specific
    p.add_argument("--tag_classes", type=str, default="pns",
                    help="Tag classes in the targets, e.g. 'pns' or 'pn' (others become plain words)")
    p.add_argument("--tag_weight", type=float, default=None,
                    help="Cross-entropy weight on tag tokens (default: unweighted)")
    p.add_argument("--tag_init", choices=["words", "mean"], default="words",
                    help="Init tag embeddings from descriptor words, or all from the vocab mean")
    p.add_argument("--tag_lr_scale", type=float, default=10.0,
                    help="Tag embedding rows train at this multiple of --lr (1 = no scaling)")
    p.add_argument("--oversample", type=int, default=1,
                    help="Repeat utterances containing paraphasias N times in train")
    p.add_argument("--time_perturbation", action=argparse.BooleanOptionalAction, default=True,
                    help="Speed perturbation at CHAI's rates")

    # LOSO cross-validation
    p.add_argument("--loso", action="store_true", default=False,
                    help="Run all LOSO folds")
    p.add_argument("--test_speaker", type=str, default=None,
                    help="Single test speaker for one fold (used if --loso is not set)")

    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                    help="Resume an interrupted fold from its latest checkpoint in output_dir")

    # Early stopping
    p.add_argument("--early_stopping", type=int, default=5,
                    help="Early stopping patience in epochs (0 to disable)")

    # Performance
    p.add_argument("--num_workers", type=int, default=0,
                    help="Dataloader workers (0 for macOS, 4+ for Linux/GPU)")
    p.add_argument("--empty_cache_steps", type=int, default=None,
                    help="Free the accelerator cache every N steps (keeps Apple MPS memory from creeping)")

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

    train_utts, dev_utts, test_utts = dataset.loso_split(test_speaker)

    # Whisper sees at most 30s; longer clips get truncated against a full
    # transcript, which teaches hallucination. Test set is left untouched.
    if args.max_duration > 0:
        n_before = len(train_utts) + len(dev_utts)
        train_utts = [u for u in train_utts if u.end_time - u.start_time <= args.max_duration]
        dev_utts = [u for u in dev_utts if u.end_time - u.start_time <= args.max_duration]
        print(f"Dropped {n_before - len(train_utts) - len(dev_utts)} train/dev "
              f"utterances longer than {args.max_duration}s")

    print(f"Train: {len(train_utts)} utterances")
    print(f"Dev:   {len(dev_utts)} utterances")
    print(f"Test:  {len(test_utts)} utterances")

    config = WhisperParaphasiaConfig(
        model_name=args.model_name,
        freeze_encoder=args.freeze_encoder,
        tag_init=args.tag_init,
    )
    model, tokenizer = build_model(config)
    tag_ids = get_paraphasia_token_ids(tokenizer)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_name)

    collator = ParaphasiaDataCollator(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        apply_time_perturbation=args.time_perturbation,
        tag_classes=tuple(args.tag_classes),
    )

    train_ds = dataset.to_hf_dataset(train_utts, oversample_paraphasia=args.oversample)
    dev_ds = dataset.to_hf_dataset(dev_utts)

    report_to = "none"
    run_name = None
    if args.wandb:
        import wandb
        from datetime import datetime
        report_to = "wandb"
        ts = datetime.now().strftime("%m%d-%H%M")
        run_name = args.wandb_run_name or f"fold-{test_speaker}-{ts}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                **vars(args),
                "test_speaker": test_speaker,
                "train_size": len(train_utts),
                "dev_size": len(dev_utts),
                "test_size": len(test_utts),
            },
            reinit="finish_previous",
        )

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_ratio,  # float < 1 is a ratio in transformers v5
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        eval_strategy="steps" if args.max_steps > 0 else "epoch",
        save_strategy="steps" if args.max_steps > 0 else "epoch",
        eval_steps=args.max_steps if args.max_steps > 0 else None,
        save_steps=args.max_steps if args.max_steps > 0 else None,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=False,
        logging_steps=10,
        report_to=report_to,
        run_name=run_name,
        dataloader_num_workers=args.num_workers,
        torch_empty_cache_steps=args.empty_cache_steps,
        remove_unused_columns=False,
        seed=args.seed,
    )

    callbacks = []
    if args.early_stopping > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping))

    class_weights = None
    if args.tag_weight is not None:
        class_weights = get_class_weight_tensor(tokenizer, args.tag_weight)

    trainer = ParaphasiaTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=callbacks,
        class_weights=class_weights,
        tag_token_ids=list(tag_ids.values()),
        tag_lr_scale=args.tag_lr_scale,
        compute_metrics=make_tag_metrics(tag_ids),
        preprocess_logits_for_metrics=make_tag_logits_preprocessor(list(tag_ids.values())),
    )

    resume_from = None
    if args.resume and Path(output_dir).is_dir():
        resume_from = get_last_checkpoint(output_dir)
        if resume_from:
            print(f"Resuming from {resume_from}")
    train_result = trainer.train(resume_from_checkpoint=resume_from)

    # Best checkpoint (by dev loss) is loaded at end; save it as the fold model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    feature_extractor.save_pretrained(output_dir)
    # The intermediate checkpoint duplicates the saved model
    for ckpt in Path(output_dir).glob("checkpoint-*"):
        shutil.rmtree(ckpt)

    metrics = train_result.metrics
    metrics["test_speaker"] = test_speaker
    metrics["train_size"] = len(train_utts)
    metrics["dev_size"] = len(dev_utts)
    metrics["test_size"] = len(test_utts)
    metrics["best_eval_loss"] = trainer.state.best_metric
    (Path(output_dir) / "train_metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\nFold complete. Best dev loss: {trainer.state.best_metric:.4f}")

    if args.wandb:
        import wandb
        wandb.finish()

    return metrics


def main():
    args = parse_args()

    print(f"Loading dataset from {args.data_path}...")
    dataset = AphasiaBankDataset.load(args.data_path)
    print(f"Loaded {len(dataset.utterances)} utterances, "
          f"{dataset.num_speakers} speakers: {dataset.speakers}")

    if args.loso:
        all_metrics = []
        for spk in dataset.speakers:
            fold_dir = str(Path(args.output_dir) / f"fold_{spk}")
            done = Path(fold_dir) / "train_metrics.json"
            if done.exists():
                print(f"Fold {spk}: already complete, skipping")
                all_metrics.append(json.loads(done.read_text()))
                continue
            all_metrics.append(train_fold(args, dataset, spk, fold_dir))

        agg_path = Path(args.output_dir) / "all_fold_metrics.json"
        agg_path.write_text(json.dumps(all_metrics, indent=2))
        print(f"\nAll {len(all_metrics)} folds complete. Metrics: {agg_path}")
    elif args.test_speaker:
        train_fold(args, dataset, args.test_speaker, args.output_dir)
    else:
        print("Error: pass --loso or --test_speaker")
        sys.exit(1)


if __name__ == "__main__":
    main()
