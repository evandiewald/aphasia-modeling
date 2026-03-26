"""Whisper tokenizer setup.

The tokenizer is unmodified from stock Whisper — no special paraphasia tokens.
Paraphasia detection is handled by a separate classification head on the
decoder hidden states, not by emitting special tokens from the vocabulary.
"""

from __future__ import annotations

from transformers import WhisperTokenizerFast


def build_tokenizer(
    model_name: str = "openai/whisper-small",
    language: str = "en",
    task: str = "transcribe",
) -> WhisperTokenizerFast:
    """Load stock Whisper tokenizer.

    Args:
        model_name: HuggingFace model ID or local path.
        language: Language for Whisper's forced decoder prefix.
        task: Whisper task ("transcribe" or "translate").

    Returns:
        Unmodified Whisper tokenizer.
    """
    return WhisperTokenizerFast.from_pretrained(
        model_name,
        language=language,
        task=task,
    )
