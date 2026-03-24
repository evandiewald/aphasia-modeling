"""CLI entry point for aphasia-modeling pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def cmd_preprocess(args: argparse.Namespace) -> None:
    """Parse and preprocess AphasiaBank CHAT files."""
    from aphasia_modeling.data import parse_cha_directory
    from aphasia_modeling.data.preprocess import preprocess_dataset, to_single_seq

    print(f"Parsing .cha files from {args.cha_dir} ...")
    utterances = parse_cha_directory(
        args.cha_dir,
        audio_dir=args.audio_dir,
    )
    print(f"  Found {len(utterances)} raw PAR utterances")

    valid = preprocess_dataset(utterances)
    print(f"  {len(valid)} valid after preprocessing ({len(utterances) - len(valid)} skipped)")

    # Print label distribution
    label_counts = {"c": 0, "p": 0, "n": 0, "s": 0}
    for utt in valid:
        for label in utt.labels:
            label_counts[label] = label_counts.get(label, 0) + 1
    total_words = sum(label_counts.values())
    print(f"  Word-level label distribution ({total_words} total words):")
    for label in ("c", "p", "n", "s"):
        count = label_counts[label]
        pct = 100 * count / total_words if total_words else 0
        print(f"    [{label}]: {count} ({pct:.1f}%)")

    # Save
    if args.output:
        from aphasia_modeling.data.dataset import AphasiaBankDataset

        dataset = AphasiaBankDataset(valid)
        dataset.save(args.output)
        print(f"  Saved to {args.output}")

    # Show samples
    if args.show_samples:
        n = min(args.show_samples, len(valid))
        print(f"\n  Sample utterances ({n}):")
        for utt in valid[:n]:
            seq = to_single_seq(utt.words, utt.labels)
            print(f"    [{utt.utterance_id}] {seq}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Run evaluation metrics on prediction files."""
    from aphasia_modeling.evaluation import compute_all_metrics

    ref_lines = Path(args.ref_file).read_text().strip().splitlines()
    hyp_lines = Path(args.hyp_file).read_text().strip().splitlines()

    if len(ref_lines) != len(hyp_lines):
        print(f"Error: ref has {len(ref_lines)} lines, hyp has {len(hyp_lines)}")
        sys.exit(1)

    refs = [line.strip().split() for line in ref_lines]
    hyps = [line.strip().split() for line in hyp_lines]

    result = compute_all_metrics(refs, hyps)
    print(result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aphasia speech modeling pipeline"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Preprocess command
    prep = subparsers.add_parser("preprocess", help="Parse and preprocess CHAT files")
    prep.add_argument("cha_dir", help="Directory containing .cha files")
    prep.add_argument("--audio-dir", help="Directory containing .wav files")
    prep.add_argument("--output", "-o", help="Output JSON path for preprocessed data")
    prep.add_argument("--show-samples", type=int, default=5, help="Number of sample utterances to print")

    # Evaluate command
    evl = subparsers.add_parser("evaluate", help="Run evaluation metrics")
    evl.add_argument("ref_file", help="Reference file (one utterance per line, tokens space-separated)")
    evl.add_argument("hyp_file", help="Hypothesis file (same format)")

    args = parser.parse_args()
    if args.command == "preprocess":
        cmd_preprocess(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
