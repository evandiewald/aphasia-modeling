#!/usr/bin/env python3
"""Evaluate a trained Whisper paraphasia model on a test set.

Runs inference on all test utterances and computes the full CHAI metric suite.

Usage:
  python scripts/evaluate_model.py \
    --model_path checkpoints/phase2/fold_speaker_01 \
    --data_path data/fridriksson.json \
    --test_speaker speaker_01 \
    --output_dir results/fold_speaker_01
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aphasia_modeling.data.dataset import AphasiaBankDataset
from aphasia_modeling.data.preprocess import parse_single_seq, to_single_seq
from aphasia_modeling.evaluation.metrics import compute_all_metrics
from aphasia_modeling.model.inference import ParaphasiaPredictor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate paraphasia model")
    p.add_argument("--model_path", type=str, required=True,
                    help="Path to trained model checkpoint")
    p.add_argument("--data_path", type=str, required=True,
                    help="Path to preprocessed dataset JSON")
    p.add_argument("--test_speaker", type=str, required=True,
                    help="Speaker ID to use as test set")
    p.add_argument("--output_dir", type=str, default="results",
                    help="Directory for evaluation output")
    p.add_argument("--device", type=str, default=None,
                    help="Device (cuda/cpu), auto-detects if not set")
    p.add_argument("--batch_size", type=int, default=8,
                    help="Batch size for inference")
    p.add_argument("--quick", type=int, default=0,
                    help="Only run on N utterances (0 = all)")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading dataset from {args.data_path}...")
    dataset = AphasiaBankDataset.load(args.data_path)
    _, _, test_utts = dataset.loso_split(args.test_speaker)
    if args.quick > 0:
        test_utts = test_utts[:args.quick]
    print(f"Test set: {len(test_utts)} utterances (speaker: {args.test_speaker})")

    # Load model
    print(f"Loading model from {args.model_path}...")
    predictor = ParaphasiaPredictor(args.model_path, device=args.device)

    # Use half precision for inference if on CUDA
    if predictor.device.type == "cuda":
        predictor.model = predictor.model.half()

    # Build references and load audio in batches
    refs = []
    hyps = []
    results = []
    batch_size = args.batch_size

    # Pre-build all references
    for utt in test_utts:
        ref_seq = to_single_seq(utt.words, utt.labels)
        refs.append(ref_seq.split())
        results.append({
            "utterance_id": utt.utterance_id,
            "reference": ref_seq,
            "hypothesis": "",
        })

    # Batched inference
    for i in tqdm(range(0, len(test_utts), batch_size), desc="Inference"):
        batch_utts = test_utts[i : i + batch_size]

        # Load audio for the batch (using segment timestamps)
        audios = []
        for utt in batch_utts:
            if utt.audio_path:
                kwargs = {"sr": 16000, "mono": True}
                if utt.start_time > 0 or utt.end_time > 0:
                    kwargs["offset"] = utt.start_time
                    if utt.end_time > utt.start_time:
                        kwargs["duration"] = utt.end_time - utt.start_time
                audio, _ = librosa.load(utt.audio_path, **kwargs)
                audios.append(audio)
            else:
                audios.append(np.zeros(16000, dtype=np.float32))

        # Batch predict
        hyp_texts = predictor.predict_batch(audios)

        for j, hyp_text in enumerate(hyp_texts):
            idx = i + j
            hyps.append(hyp_text.split() if hyp_text else [])
            results[idx]["hypothesis"] = hyp_text

    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_all_metrics(refs, hyps)
    print(f"\n{metrics}")

    # Save outputs
    metrics_dict = {
        "test_speaker": args.test_speaker,
        "num_utterances": len(test_utts),
        "wer": metrics.wer,
        "awer": metrics.awer,
        "awer_pd": metrics.awer_pd,
        "td_binary": metrics.td_binary,
        "td_p": metrics.td_p,
        "td_n": metrics.td_n,
        "td_all": metrics.td_all,
        "f1_p": metrics.f1_p,
        "f1_n": metrics.f1_n,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics_dict, indent=2))
    (output_dir / "predictions.json").write_text(json.dumps(results, indent=2))

    # Print sample predictions
    n_samples = min(10, len(results))
    print(f"\nSample predictions ({n_samples}):")
    for r in results[:n_samples]:
        print(f"  [{r['utterance_id']}]")
        print(f"    REF: {r['reference']}")
        print(f"    HYP: {r['hypothesis']}")
        print()

    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
