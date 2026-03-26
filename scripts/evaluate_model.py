#!/usr/bin/env python3
"""Evaluate a trained Whisper paraphasia model on a test set.

Runs inference on all test utterances and computes WER + utterance-level F1.

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
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aphasia_modeling.data.dataset import AphasiaBankDataset
from aphasia_modeling.data.preprocess import to_single_seq
from aphasia_modeling.evaluation.metrics import compute_wer
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
    p.add_argument("--threshold", type=float, default=0.5,
                    help="Classification threshold (0-1). Higher = fewer tags.")
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
    predictor = ParaphasiaPredictor(
        args.model_path, device=args.device, threshold=args.threshold
    )

    # Use half precision for inference if on CUDA
    if predictor.device.type == "cuda":
        predictor.model = predictor.model.half()

    # Build ground truth
    ref_texts = []  # Plain text (for WER)
    ref_has_p = []  # Binary: does utterance have [p]?
    ref_has_n = []  # Binary: does utterance have [n]?

    for utt in test_utts:
        ref_texts.append(" ".join(utt.words))
        ref_has_p.append(1 if "p" in utt.labels else 0)
        ref_has_n.append(1 if "n" in utt.labels else 0)

    # Run inference in batches
    hyp_texts = []
    hyp_has_p = []
    hyp_has_n = []
    results = []
    batch_size = args.batch_size

    for i in tqdm(range(0, len(test_utts), batch_size), desc="Inference"):
        batch_utts = test_utts[i : i + batch_size]

        # Load audio
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

        preds = predictor.predict_batch(audios)

        for j, pred in enumerate(preds):
            idx = i + j
            hyp_texts.append(pred.text)
            hyp_has_p.append(1 if pred.has_p else 0)
            hyp_has_n.append(1 if pred.has_n else 0)

            ref_seq = to_single_seq(test_utts[idx].words, test_utts[idx].labels)
            results.append({
                "utterance_id": test_utts[idx].utterance_id,
                "reference": ref_seq,
                "hypothesis": pred.to_single_seq(),
                "prob_p": round(pred.prob_p, 4),
                "prob_n": round(pred.prob_n, 4),
            })

    # Compute WER
    print("\nComputing metrics...")
    refs_for_wer = [r.split() for r in ref_texts]
    hyps_for_wer = [h.split() for h in hyp_texts]
    wer = compute_wer(refs_for_wer, hyps_for_wer)

    # Compute utterance-level F1
    f1_p = f1_score(ref_has_p, hyp_has_p, zero_division=0.0)
    f1_n = f1_score(ref_has_n, hyp_has_n, zero_division=0.0)
    prec_p = precision_score(ref_has_p, hyp_has_p, zero_division=0.0)
    prec_n = precision_score(ref_has_n, hyp_has_n, zero_division=0.0)
    rec_p = recall_score(ref_has_p, hyp_has_p, zero_division=0.0)
    rec_n = recall_score(ref_has_n, hyp_has_n, zero_division=0.0)

    print(f"\nWER:          {wer:.4f}")
    print(f"F1-[p]:       {f1_p:.4f}  (P={prec_p:.4f}, R={rec_p:.4f})")
    print(f"F1-[n]:       {f1_n:.4f}  (P={prec_n:.4f}, R={rec_n:.4f})")
    print(f"Threshold:    {args.threshold}")
    print(f"Utterances:   {len(test_utts)}")
    print(f"  Ref has [p]: {sum(ref_has_p)}, Hyp has [p]: {sum(hyp_has_p)}")
    print(f"  Ref has [n]: {sum(ref_has_n)}, Hyp has [n]: {sum(hyp_has_n)}")

    # Save outputs
    metrics_dict = {
        "test_speaker": args.test_speaker,
        "num_utterances": len(test_utts),
        "threshold": args.threshold,
        "wer": wer,
        "f1_p": f1_p,
        "f1_n": f1_n,
        "precision_p": prec_p,
        "precision_n": prec_n,
        "recall_p": rec_p,
        "recall_n": rec_n,
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
        print(f"    P(p)={r['prob_p']:.3f}  P(n)={r['prob_n']:.3f}")
        print()

    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
