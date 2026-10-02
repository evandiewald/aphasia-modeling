"""Extend Whisper's tokenizer with inline paraphasia tags.

Adds [p] (phonemic), [n] (neologistic), and [s] (semantic) as single tokens
the decoder emits right after the paraphasic word (CHAI's single-seq format).

The tags are added as regular (non-special) tokens with lstrip=True, so the
space before a tag is absorbed into it rather than becoming a standalone
"Ġ" token, and decode(skip_special_tokens=True) keeps them.
"""

from __future__ import annotations

import re

from transformers import AddedToken, WhisperTokenizerFast

PARAPHASIA_TOKENS = ["[p]", "[n]", "[s]"]

_TAG_SPACING = re.compile(r"\s*(\[[pns]\])")


def build_tokenizer(
    model_name: str = "openai/whisper-small",
    language: str = "en",
    task: str = "transcribe",
) -> WhisperTokenizerFast:
    """Load Whisper tokenizer and add the paraphasia tag tokens.

    Idempotent: loading a fine-tuned checkpoint that already has the tags
    adds nothing.
    """
    tokenizer = WhisperTokenizerFast.from_pretrained(
        model_name, language=language, task=task
    )
    tokenizer.add_tokens([
        AddedToken(t, lstrip=True, rstrip=False, normalized=False, special=False)
        for t in PARAPHASIA_TOKENS
    ])
    return tokenizer


def get_paraphasia_token_ids(tokenizer: WhisperTokenizerFast) -> dict[str, int]:
    """Map each tag string to its token ID, e.g. {"[p]": 51865, ...}."""
    ids = {}
    for token in PARAPHASIA_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        assert token_id is not None and token_id != tokenizer.unk_token_id, (
            f"Token {token!r} not in vocabulary — was build_tokenizer() used?"
        )
        ids[token] = token_id
    return ids


def format_target(single_seq: str) -> str:
    """Format a single-seq string as a Whisper target.

    Whisper's pretrained outputs start with a space-prefixed word
    ("Ġaphasia"), so targets get a leading space to match.
    """
    return " " + single_seq.strip()


def normalize_output(text: str) -> str:
    """Normalize decoded text back to CHAI single-seq format.

    Lowercases, strips punctuation (but not tag brackets), and restores the
    space before each tag that lstrip absorbed: "fekts[p] my" -> "fekts [p] my".
    """
    text = text.lower()
    text = re.sub(r"[^\w\s'\[\]]", "", text)
    text = text.replace("'", "")
    text = _TAG_SPACING.sub(r" \1", text)
    return re.sub(r"\s+", " ", text).strip()
