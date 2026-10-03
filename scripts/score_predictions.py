#!/usr/bin/env python3
"""Score saved predictions.json files with the CHAI metric suite.

Re-scores without re-running inference, e.g. after a metric fix, or to
compare runs on the same subset of speakers / tag classes.

Usage:
  # Pooled + per-speaker metrics over every predictions.json under a results dir
  python scripts/score_predictions.py results/scripts-small-taginit

  # Two runs on the same folds, scoring only [p]/[n] (other tags dropped from
  # both reference and hypothesis, so a pn model isn't charged for [s])
  python scripts/score_predictions.py results/scripts-small-taginit results/scripts-small-pn \\
    --speakers P4,P10 --classes pn
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aphasia_modeling.evaluation.metrics import compute_all_metrics

TAGS = ("[p]", "[n]", "[s]")


def load_rows(path: Path) -> list[dict]:
    """Rows from a predictions.json, or from every fold_*/predictions.json under a dir."""
    if path.is_file():
        return json.loads(path.read_text())
    files = sorted(path.glob("fold_*/predictions.json")) or [path / "predictions.json"]
    return [row for f in files for row in json.loads(f.read_text())]


def keep_classes(text: str, classes: str) -> str:
    return " ".join(t for t in text.split() if t not in TAGS or t[1] in classes)


def score(rows: list[dict]) -> dict:
    m = asdict(compute_all_metrics(
        [r["reference"].split() for r in rows], [r["hypothesis"].split() for r in rows]
    ))
    counts = Counter()
    for r in rows:
        for t in TAGS:
            counts[f"ref_{t[1]}"] += r["reference"].split().count(t)
            counts[f"hyp_{t[1]}"] += r["hypothesis"].split().count(t)
    return {**m, **counts}


def speaker(row: dict) -> str:
    return row.get("speaker_id") or row["utterance_id"].split("_")[0]


def main():
    p = argparse.ArgumentParser(description="Score saved predictions")
    p.add_argument("paths", nargs="+", help="predictions.json files or results dirs")
    p.add_argument("--speakers", type=str, default=None, help="Comma-separated subset")
    p.add_argument("--classes", type=str, default="pns", help="Tag classes to score")
    p.add_argument("--per_speaker", action="store_true", help="Also print each speaker")
    p.add_argument("--output", type=str, default=None, help="Write metrics JSON here (one path only)")
    args = p.parse_args()

    for path in map(Path, args.paths):
        rows = load_rows(path)
        if args.speakers:
            rows = [r for r in rows if speaker(r) in args.speakers.split(",")]
        rows = [{**r, "reference": keep_classes(r["reference"], args.classes),
                 "hypothesis": keep_classes(r["hypothesis"], args.classes)} for r in rows]
        pooled = score(rows)
        per = {spk: score([r for r in rows if speaker(r) == spk])
               for spk in sorted({speaker(r) for r in rows}, key=lambda s: (len(s), s))}

        print(f"\n== {path} ({len(per)} speakers, {len(rows)} utterances, classes {args.classes})")
        print(_fmt("pooled", pooled))
        if args.per_speaker:
            for spk, m in per.items():
                print(_fmt(spk, m))
        if args.output:
            Path(args.output).write_text(json.dumps({"pooled": pooled, "per_speaker": per}, indent=2))


def _fmt(name: str, m: dict) -> str:
    return (f"{name:7s} n={m['num_utterances']:4d} WER={m['wer']*100:5.1f} AWER={m['awer']*100:5.1f} "
            f"TD bin={m['td_binary']:.2f} p={m['td_p']:.2f} n={m['td_n']:.2f} s={m['td_s']:.2f} all={m['td_all']:.2f} "
            f"F1 p/n/s={m['f1_p']:.2f}/{m['f1_n']:.2f}/{m['f1_s']:.2f} "
            f"tags hyp/ref p={m['hyp_p']}/{m['ref_p']} n={m['hyp_n']}/{m['ref_n']} s={m['hyp_s']}/{m['ref_s']}")


if __name__ == "__main__":
    main()
