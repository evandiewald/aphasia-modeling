"""Extend Whisper's tokenizer with paraphasia special tokens.

Adds [p] (phonemic) and [n] (neologistic) as single tokens that the
decoder can emit inline with word tokens. Semantic paraphasias ([s]) are
handled by a separate Stage 2 LLM pass.
"""

from __future__ import annotations

from transformers import WhisperTokenizerFast, WhisperTokenizer


PARAPHASIA_TOKENS = ["[p]", "[n]"]

# Class weights from CHAI paper (correct=1 is implicit via CE loss)
PARAPHASIA_CLASS_WEIGHTS = {
    "[p]": 10.0,
    "[n]": 20.0,
}


def build_tokenizer(
    model_name: str = "openai/whisper-small",
    language: str = "en",
    task: str = "transcribe",
) -> WhisperTokenizerFast:
    """Load Whisper tokenizer and add paraphasia special tokens.

    Args:
        model_name: HuggingFace model ID or local path.
        language: Language for Whisper's forced decoder prefix.
        task: Whisper task ("transcribe" or "translate").

    Returns:
        Tokenizer with [p], [n] added as special tokens.
        Use tokenizer.additional_special_tokens_ids to get their IDs.
    """
    tokenizer = WhisperTokenizerFast.from_pretrained(
        model_name,
        language=language,
        task=task,
    )

    # Add paraphasia tokens as additional special tokens
    # This ensures they're treated as single tokens, not split by BPE
    num_added = tokenizer.add_special_tokens({
        "additional_special_tokens": PARAPHASIA_TOKENS,
    })

    assert num_added == len(PARAPHASIA_TOKENS), (
        f"Expected to add {len(PARAPHASIA_TOKENS)} tokens, but added {num_added}. "
        f"Some tokens may already exist in the vocabulary."
    )

    return tokenizer


def get_paraphasia_token_ids(tokenizer: WhisperTokenizerFast) -> dict[str, int]:
    """Get the token IDs for the paraphasia special tokens.

    Returns:
        Dict mapping token string to token ID, e.g. {"[p]": 51866, ...}
    """
    ids = {}
    for token in PARAPHASIA_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        assert token_id != tokenizer.unk_token_id, (
            f"Token {token!r} resolved to UNK — was build_tokenizer() called?"
        )
        ids[token] = token_id
    return ids
