"""Evaluation metrics matching CHAI's evaluation.py.

All metrics:
- WER: Standard word error rate (paraphasia tags stripped)
- AWER: Augmented WER on word/label compound tokens
- TD-binary: Temporal distance for binary paraphasia detection
- TD-multiclass: Per-class temporal distance (TD-[p], TD-[n], TD-all)
- Utterance-level F1: Per-class binary classification at utterance level
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from jiwer import wer as jiwer_wer
from sklearn.metrics import f1_score, precision_score, recall_score

from .alignment import (
    EPS,
    align_sequences,
    strip_paraphasia_tags,
)


@dataclass
class MetricResult:
    """Container for all evaluation metrics."""

    wer: float = 0.0
    awer: float = 0.0
    awer_pd: float = 0.0  # Paraphasia-detection variant

    td_binary: float = 0.0
    td_p: float = 0.0
    td_n: float = 0.0
    td_all: float = 0.0

    f1_p: float = 0.0
    f1_n: float = 0.0
    recall_p: float = 0.0
    recall_n: float = 0.0

    num_utterances: int = 0

    def __str__(self) -> str:
        lines = [
            f"WER:       {self.wer:.4f}",
            f"AWER:      {self.awer:.4f}",
            f"AWER-PD:   {self.awer_pd:.4f}",
            f"TD-binary: {self.td_binary:.4f}",
            f"TD-[p]:    {self.td_p:.4f}",
            f"TD-[n]:    {self.td_n:.4f}",
            f"TD-all:    {self.td_all:.4f}",
            f"F1-[p]:    {self.f1_p:.4f}",
            f"F1-[n]:    {self.f1_n:.4f}",
            f"Recall-[p]: {self.recall_p:.4f}",
            f"Recall-[n]: {self.recall_n:.4f}",
            f"Utterances: {self.num_utterances}",
        ]
        return "\n".join(lines)


def compute_wer(
    refs: list[list[str]], hyps: list[list[str]]
) -> float:
    """Standard WER with paraphasia tags stripped.

    Args:
        refs: List of reference token sequences (may include [p]/[n]/[s]).
        hyps: List of hypothesis token sequences.
    """
    ref_strs = []
    hyp_strs = []
    for ref, hyp in zip(refs, hyps):
        ref_words, _ = strip_paraphasia_tags(ref)
        hyp_words, _ = strip_paraphasia_tags(hyp)
        ref_strs.append(" ".join(ref_words) if ref_words else "<empty>")
        hyp_strs.append(" ".join(hyp_words) if hyp_words else "<empty>")
    return jiwer_wer(ref_strs, hyp_strs)


def compute_awer(
    refs: list[list[str]], hyps: list[list[str]]
) -> float:
    """Augmented WER: WER on word/label compound tokens.

    Each word becomes "word/label" (e.g., "cat/p", "the/c").
    """
    ref_strs = []
    hyp_strs = []
    for ref, hyp in zip(refs, hyps):
        ref_strs.append(" ".join(_to_compound_tokens(ref)) or "<empty>")
        hyp_strs.append(" ".join(_to_compound_tokens(hyp)) or "<empty>")
    return jiwer_wer(ref_strs, hyp_strs)


def compute_awer_pd(
    refs: list[list[str]], hyps: list[list[str]]
) -> float:
    """AWER-PD: Paraphasia detection variant.

    For phonemic/neologistic paraphasias, remove the word and keep only
    the label. For correct/semantic, keep "word label".
    """
    ref_strs = []
    hyp_strs = []
    for ref, hyp in zip(refs, hyps):
        ref_strs.append(" ".join(_to_pd_tokens(ref)) or "<empty>")
        hyp_strs.append(" ".join(_to_pd_tokens(hyp)) or "<empty>")
    return jiwer_wer(ref_strs, hyp_strs)


def _to_compound_tokens(tokens: list[str]) -> list[str]:
    """Convert token list to word/label compounds."""
    words, tag_map = strip_paraphasia_tags(tokens)
    result = []
    for i, word in enumerate(words):
        label = tag_map.get(i, "c")
        result.append(f"{word}/{label}")
    return result


def _to_pd_tokens(tokens: list[str]) -> list[str]:
    """Convert to paraphasia-detection tokens (AWER-PD format)."""
    words, tag_map = strip_paraphasia_tags(tokens)
    result = []
    for i, word in enumerate(words):
        label = tag_map.get(i, "c")
        if label in ("p", "n"):
            # For p/n paraphasias, just use the label (word is corrupted anyway)
            result.append(label)
        else:
            result.append(f"{word}/{label}")
    return result


def compute_td_binary(
    refs: list[list[str]], hyps: list[list[str]]
) -> float:
    """Temporal distance — binary paraphasia detection.

    For each true paraphasia position, find nearest predicted paraphasia.
    Average across utterances. Lower is better.
    """
    td_values = []
    for ref, hyp in zip(refs, hyps):
        td = _td_for_utterance(ref, hyp, class_specific=False)
        if td is not None:
            td_values.append(td)
    return float(np.mean(td_values)) if td_values else 0.0


def compute_td_multiclass(
    refs: list[list[str]], hyps: list[list[str]]
) -> dict[str, float]:
    """Per-class temporal distance.

    Returns dict with keys "p", "n", "all".
    """
    td_per_class = {"p": [], "n": []}

    for ref, hyp in zip(refs, hyps):
        for cls in ("p", "n"):
            td = _td_for_utterance(ref, hyp, target_class=cls)
            if td is not None:
                td_per_class[cls].append(td)

    result = {}
    for cls in ("p", "n"):
        result[cls] = float(np.mean(td_per_class[cls])) if td_per_class[cls] else 0.0
    result["all"] = sum(result.values())
    return result


def _td_for_utterance(
    ref_tokens: list[str],
    hyp_tokens: list[str],
    class_specific: bool = True,
    target_class: str | None = None,
) -> float | None:
    """Compute temporal distance for a single utterance.

    Args:
        ref_tokens: Reference token sequence with paraphasia tags.
        hyp_tokens: Hypothesis token sequence with paraphasia tags.
        class_specific: If False, treat all paraphasias as one class (binary).
        target_class: If set, only consider this paraphasia class.
    """
    # Align sequences
    ref_words, ref_labels, hyp_words, hyp_labels = align_sequences(
        ref_tokens, hyp_tokens
    )

    seq_len = len(ref_words)
    if seq_len == 0:
        return None

    # Get positions of paraphasias in ref and hyp
    if target_class:
        ref_positions = [i for i, l in enumerate(ref_labels) if l == target_class]
        hyp_positions = [i for i, l in enumerate(hyp_labels) if l == target_class]
    elif not class_specific:
        ref_positions = [i for i, l in enumerate(ref_labels) if l in ("p", "n")]
        hyp_positions = [i for i, l in enumerate(hyp_labels) if l in ("p", "n")]
    else:
        return None

    if not ref_positions and not hyp_positions:
        return None  # No paraphasias to evaluate
    if not ref_positions or not hyp_positions:
        # One side has paraphasias, the other doesn't
        # Default distance: max(position, seq_len) per CHAI
        total = 0.0
        for pos in ref_positions:
            total += max(pos, seq_len)
        for pos in hyp_positions:
            total += max(pos, seq_len)
        return total / seq_len

    # TTC: true-to-closest (for each ref paraphasia, find nearest hyp)
    ttc = 0.0
    for r_pos in ref_positions:
        min_dist = min(abs(r_pos - h_pos) for h_pos in hyp_positions)
        ttc += min_dist

    # CTT: closest-to-true (for each hyp paraphasia, find nearest ref)
    ctt = 0.0
    for h_pos in hyp_positions:
        min_dist = min(abs(h_pos - r_pos) for r_pos in ref_positions)
        ctt += min_dist

    return (ttc + ctt) / seq_len


def compute_utterance_f1(
    refs: list[list[str]], hyps: list[list[str]]
) -> dict[str, dict[str, float]]:
    """Utterance-level binary F1 per paraphasia type.

    For each type (p, n, s): does the utterance contain at least one
    instance of that type?

    Returns dict with keys "p", "n", "s", each containing "f1", "precision",
    "recall".
    """
    result = {}
    for cls in ("p", "n"):
        ref_binary = []
        hyp_binary = []
        for ref, hyp in zip(refs, hyps):
            _, ref_tags = strip_paraphasia_tags(ref)
            _, hyp_tags = strip_paraphasia_tags(hyp)
            ref_binary.append(1 if cls in ref_tags.values() else 0)
            hyp_binary.append(1 if cls in hyp_tags.values() else 0)

        if sum(ref_binary) == 0:
            result[cls] = {"f1": 0.0, "precision": 0.0, "recall": 0.0}
        else:
            result[cls] = {
                "f1": f1_score(ref_binary, hyp_binary, zero_division=0.0),
                "precision": precision_score(ref_binary, hyp_binary, zero_division=0.0),
                "recall": recall_score(ref_binary, hyp_binary, zero_division=0.0),
            }
    return result


def compute_all_metrics(
    refs: list[list[str]], hyps: list[list[str]]
) -> MetricResult:
    """Compute the full CHAI evaluation metric suite.

    Args:
        refs: List of reference token sequences (e.g., ["the", "cat", "[p]", "sat"]).
        hyps: List of hypothesis token sequences.

    Returns:
        MetricResult with all metrics populated.
    """
    result = MetricResult(num_utterances=len(refs))

    result.wer = compute_wer(refs, hyps)
    result.awer = compute_awer(refs, hyps)
    result.awer_pd = compute_awer_pd(refs, hyps)

    result.td_binary = compute_td_binary(refs, hyps)

    td_mc = compute_td_multiclass(refs, hyps)
    result.td_p = td_mc["p"]
    result.td_n = td_mc["n"]
    result.td_all = td_mc["all"]

    utt_f1 = compute_utterance_f1(refs, hyps)
    result.f1_p = utt_f1["p"]["f1"]
    result.f1_n = utt_f1["n"]["f1"]
    result.recall_p = utt_f1["p"]["recall"]
    result.recall_n = utt_f1["n"]["recall"]

    return result
