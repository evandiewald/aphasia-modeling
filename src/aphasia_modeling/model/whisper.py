"""Whisper model setup for single-seq paraphasia detection.

Handles:
- Loading pretrained Whisper and resizing embeddings for the tag tokens
- Initializing each tag embedding from a distinct descriptor word
- Configuring generation so the tags aren't suppressed
- Optional encoder freezing and per-token loss weights for the tags
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import (
    GenerationConfig,
    WhisperForConditionalGeneration,
    WhisperTokenizerFast,
)

from .tokenizer import build_tokenizer, get_paraphasia_token_ids

TAG_INIT_WORDS = {"[p]": " phonemic", "[n]": " neologism", "[s]": " semantic"}


@dataclass
class WhisperParaphasiaConfig:
    """Configuration for Whisper paraphasia model."""

    model_name: str = "openai/whisper-small"
    language: str = "en"
    task: str = "transcribe"
    freeze_encoder: bool = False
    # "words": each tag starts from its descriptor word (TAG_INIT_WORDS);
    # "mean": all tags start from the vocab mean. Whisper ties the output
    # projection to these embeddings, so mean-init tags score identically and
    # at a low LR never separate (the model collapses onto [p]).
    tag_init: str = "words"


def build_model(
    config: WhisperParaphasiaConfig | None = None,
    tokenizer: WhisperTokenizerFast | None = None,
) -> tuple[WhisperForConditionalGeneration, WhisperTokenizerFast]:
    """Build Whisper with the paraphasia tag tokens.

    Works for both a stock checkpoint (tags get added and initialized) and
    a fine-tuned one (tags already present; nothing is resized).
    """
    if config is None:
        config = WhisperParaphasiaConfig()

    if tokenizer is None:
        tokenizer = build_tokenizer(
            model_name=config.model_name,
            language=config.language,
            task=config.task,
        )

    try:
        model = WhisperForConditionalGeneration.from_pretrained(
            config.model_name, attn_implementation="sdpa"
        )
    except (ValueError, ImportError):
        model = WhisperForConditionalGeneration.from_pretrained(config.model_name)

    old_vocab_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > old_vocab_size:
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        _init_new_embeddings(model, tokenizer, old_vocab_size, config.tag_init)

    if config.freeze_encoder:
        freeze_encoder(model)

    model.generation_config = _build_generation_config(model, config, tokenizer)

    return model, tokenizer


def freeze_encoder(model: WhisperForConditionalGeneration) -> None:
    """Freeze all encoder parameters."""
    for param in model.model.encoder.parameters():
        param.requires_grad = False


def unfreeze_encoder(model: WhisperForConditionalGeneration) -> None:
    """Unfreeze all encoder parameters."""
    for param in model.model.encoder.parameters():
        param.requires_grad = True


def get_class_weight_tensor(
    tokenizer: WhisperTokenizerFast,
    tag_weight: float,
) -> torch.Tensor:
    """Per-token cross-entropy weights: `tag_weight` for tags, 1 elsewhere."""
    weights = torch.ones(len(tokenizer))
    for token_id in get_paraphasia_token_ids(tokenizer).values():
        weights[token_id] = tag_weight
    return weights


def _init_new_embeddings(
    model: WhisperForConditionalGeneration,
    tokenizer: WhisperTokenizerFast,
    old_vocab_size: int,
    tag_init: str,
) -> None:
    """Initialize the new tag embeddings.

    "words" sets each tag to the mean embedding of its descriptor word's
    subword tokens, so the tags start apart; "mean" uses the vocab mean for
    all of them. Whisper ties proj_out to embed_tokens, so this covers the
    output projection too; untied heads are handled separately just in case.
    """
    if tag_init not in ("words", "mean"):
        raise ValueError(f"tag_init must be 'words' or 'mean', got {tag_init!r}")
    tag_ids = get_paraphasia_token_ids(tokenizer)

    def init(weight: torch.Tensor) -> None:
        weight[old_vocab_size:] = weight[:old_vocab_size].mean(dim=0)
        if tag_init == "words":
            for tag, word in TAG_INIT_WORDS.items():
                word_ids = tokenizer(word, add_special_tokens=False).input_ids
                weight[tag_ids[tag]] = weight[word_ids].mean(dim=0)

    with torch.no_grad():
        embed = model.get_input_embeddings().weight
        init(embed)

        proj = model.get_output_embeddings().weight
        if proj.data_ptr() != embed.data_ptr():
            init(proj)


def _build_generation_config(
    model: WhisperForConditionalGeneration,
    config: WhisperParaphasiaConfig,
    tokenizer: WhisperTokenizerFast,
) -> GenerationConfig:
    """Whisper generation config with the tag tokens unsuppressed."""
    try:
        gen = GenerationConfig.from_pretrained(config.model_name)
    except OSError:
        gen = model.generation_config
    gen.language = config.language
    gen.task = config.task
    tag_ids = set(get_paraphasia_token_ids(tokenizer).values())
    gen.suppress_tokens = [t for t in (gen.suppress_tokens or []) if t not in tag_ids]
    return gen
