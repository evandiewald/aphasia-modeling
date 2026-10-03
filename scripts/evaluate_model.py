#!/usr/bin/env python3
"""Evaluate Whisper single-seq paraphasia models with the CHAI metric suite.

Runs inference on each fold's held-out speaker and computes WER, AWER,
AWER-PD, TD-binary, TD-[p]/[n]/[s]/all, and utterance-level F1. With
--loso, predictions from all folds are pooled before scoring, as CHAI
reports results "aggregated over all folds".

Usage:
  # One fold
  python scripts/evaluate_model.py \
    --model_dir checkpoints/scripts-small \
    --data_path datasets/Scripts/scripts_fridriksson.json \
    --test_speaker P1

  # All folds (expects model_dir/fold_<spk> for every speaker)
  python scripts/evaluate_model.py \
    --model_dir checkpoints/scripts-small \
    --data_path datasets/Scripts/scripts_fridriksson.json \
    --loso
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import librosa
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aphasia_modeling.data.dataset import AphasiaBankDataset
from aphasia_modeling.data.preprocess import to_single_seq
from aphasia_modeling.evaluation.metrics import MetricResult, compute_all_metrics
from aphasia_modeling.model.inference import SAMPLING_RATE, ParaphasiaPredictor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate paraphasia model")
    p.add_argument("--model_dir", type=str, required=True,
                    help="Fold checkpoint dir, or LOSO root containing fold_<spk> dirs")
    p.add_argument("--data_path", type=str, required=True,
                    help="Path to preprocessed dataset JSON")
    p.add_argument("--loso", action="store_true", default=False,
                    help="Evaluate every fold and pool predictions")
    p.add_argument("--speakers", type=str, default=None,
                    help="With --loso: only evaluate these comma-separated folds")
    p.add_argument("--test_speaker", type=str, default=None,
                    help="Speaker held out by the checkpoint in --model_dir")
    p.add_argument("--output_dir", type=str, default=None,
                    help="Where to write metrics/predictions (default: model_dir/eval)")
    p.add_argument("--batch_size", type=int, default=16,
                    help="Utterances per inference batch")
    p.add_argument("--num_beams", type=int, default=1,
                    help="Beam size for decoding")
    p.add_argument("--device", type=str, default=None,
                    help="Device (cuda/cpu), auto-detects if not set")
    p.add_argument("--quick", type=int, default=0,
                    help="Only run on the first N test utterances per fold (0 = all)")
    return p.parse_args()


def predict_fold(args, dataset, test_speaker: str, model_dir: Path) -> list[dict]:
    """Run inference on one fold's test speaker."""
    _, _, test_utts = dataset.loso_split(test_speaker)
    if args.quick > 0:
        test_utts = test_utts[:args.quick]
    print(f"\n[{test_speaker}] {len(test_utts)} test utterances, model {model_dir}")

    predictor = ParaphasiaPredictor(model_dir, device=args.device, num_beams=args.num_beams)

    rows = []
    for i in tqdm(range(0, len(test_utts), args.batch_size), desc=f"Inference {test_speaker}"):
        batch = test_utts[i : i + args.batch_size]
        audios = [
            librosa.load(
                u.audio_path,
                sr=SAMPLING_RATE,
                offset=u.start_time,
                duration=u.end_time - u.start_time,
            )[0]
            for u in batch
        ]
        for utt, hyp in zip(batch, predictor.predict_batch(audios)):
            rows.append({
                "utterance_id": utt.utterance_id,
                "speaker_id": utt.speaker_id,
                "reference": to_single_seq(utt.words, utt.labels),
                "hypothesis": hyp,
            })

    del predictor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def score(rows: list[dict]) -> MetricResult:
    refs = [r["reference"].split() for r in rows]
    hyps = [r["hypothesis"].split() for r in rows]
    return compute_all_metrics(refs, hyps)


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir) if args.output_dir else model_dir / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = AphasiaBankDataset.load(args.data_path)

    if args.loso:
        speakers = dataset.speakers
        if args.speakers:
            speakers = [s for s in speakers if s in args.speakers.split(",")]
        folds = [(spk, model_dir / f"fold_{spk}") for spk in speakers]
        missing = [str(d) for _, d in folds if not d.exists()]
        if missing:
            print(f"Error: missing fold checkpoints: {missing}")
            sys.exit(1)
    elif args.test_speaker:
        folds = [(args.test_speaker, model_dir)]
    else:
        print("Error: pass --loso or --test_speaker")
        sys.exit(1)

    rows = []
    per_fold = {}
    for spk, fold_dir in folds:
        fold_rows = predict_fold(args, dataset, spk, fold_dir)
        per_fold[spk] = asdict(score(fold_rows))
        rows.extend(fold_rows)

    pooled = score(rows)
    result = {"pooled": asdict(pooled), "per_fold": per_fold}
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    (output_dir / "predictions.json").write_text(json.dumps(rows, indent=2))

    print(f"\n{'='*60}\nPooled over {len(folds)} fold(s), {len(rows)} utterances\n{'='*60}")
    print(pooled)

    print("\nSample predictions:")
    for r in rows[:8]:
        print(f"  [{r['utterance_id']}]")
        print(f"    REF: {r['reference']}")
        print(f"    HYP: {r['hypothesis']}")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
