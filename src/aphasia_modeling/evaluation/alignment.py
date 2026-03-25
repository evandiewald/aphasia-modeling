"""Sequence alignment for evaluation.

Implements the alignment pipeline from CHAI's evaluation.py:
1. Strip paraphasia tags to get word-only sequences
2. Align predicted/true word sequences via minimum edit distance
3. Reinsert paraphasia tags onto aligned sequences
"""

from __future__ import annotations

EPS = "<eps>"
PARAPHASIA_TAGS = {"[p]", "[n]", "[c]"}


def strip_paraphasia_tags(tokens: list[str]) -> tuple[list[str], dict[int, str]]:
    """Remove paraphasia tags, returning words and a map of word_index -> tag.

    Args:
        tokens: Token list, e.g. ["the", "cat", "[p]", "sat"]

    Returns:
        (words, tag_map) where words = ["the", "cat", "sat"]
        and tag_map = {1: "p"}
    """
    words = []
    tag_map: dict[int, str] = {}
    for token in tokens:
        if token in PARAPHASIA_TAGS:
            if words:
                tag_map[len(words) - 1] = token[1]  # "[p]" -> "p"
        elif token != EPS:
            words.append(token)
    return words, tag_map


def levenshtein_alignment(
    ref: list[str], hyp: list[str]
) -> list[tuple[str, str]]:
    """Compute minimum edit distance alignment between ref and hyp.

    Returns a list of (ref_token, hyp_token) pairs where <eps> is used
    for insertions and deletions.
    """
    n, m = len(ref), len(hyp)

    # DP table
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                    dp[i - 1][j - 1],  # substitution
                )

    # Backtrace
    alignment = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            alignment.append((ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            # Substitution
            alignment.append((ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            # Deletion (ref word missing from hyp)
            alignment.append((ref[i - 1], EPS))
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            # Insertion (extra word in hyp)
            alignment.append((EPS, hyp[j - 1]))
            j -= 1
        else:
            # Fallback: shouldn't happen but handle gracefully
            if i > 0:
                alignment.append((ref[i - 1], EPS))
                i -= 1
            elif j > 0:
                alignment.append((EPS, hyp[j - 1]))
                j -= 1

    alignment.reverse()
    return alignment


def align_sequences(
    ref_tokens: list[str], hyp_tokens: list[str]
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Align reference and hypothesis sequences with paraphasia tags.

    1. Strip paraphasia tags from both sequences
    2. Align word-only sequences
    3. Reinsert paraphasia tags

    Returns:
        (ref_words_aligned, ref_labels_aligned,
         hyp_words_aligned, hyp_labels_aligned)

        where *_words_aligned may contain <eps> for gaps and
        *_labels_aligned contain "c", "p", "n", or "s" per position.
    """
    ref_words, ref_tags = strip_paraphasia_tags(ref_tokens)
    hyp_words, hyp_tags = strip_paraphasia_tags(hyp_tokens)

    alignment = levenshtein_alignment(ref_words, hyp_words)

    ref_aligned_words = []
    ref_aligned_labels = []
    hyp_aligned_words = []
    hyp_aligned_labels = []

    ref_idx = 0
    hyp_idx = 0

    for ref_word, hyp_word in alignment:
        ref_aligned_words.append(ref_word)
        hyp_aligned_words.append(hyp_word)

        # Get labels
        if ref_word != EPS:
            ref_aligned_labels.append(ref_tags.get(ref_idx, "c"))
            ref_idx += 1
        else:
            ref_aligned_labels.append("c")

        if hyp_word != EPS:
            hyp_aligned_labels.append(hyp_tags.get(hyp_idx, "c"))
            hyp_idx += 1
        else:
            hyp_aligned_labels.append("c")

    return ref_aligned_words, ref_aligned_labels, hyp_aligned_words, hyp_aligned_labels


def reinsert_paraphasia_tags(
    words: list[str], labels: list[str]
) -> list[str]:
    """Convert aligned words + labels back to token sequence with tags.

    Example: words=["the", "cat", "sat"], labels=["c", "p", "c"]
    Returns: ["the", "cat", "[p]", "sat"]
    """
    result = []
    for word, label in zip(words, labels):
        result.append(word)
        if label in ("p", "n"):
            result.append(f"[{label}]")
    return result
