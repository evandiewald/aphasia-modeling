"""Whisper model setup for single-seq paraphasia detection.

Handles:
- Loading pretrained Whisper and resizing embeddings for the tag tokens
- Mean-initializing the new tag embeddings
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


@dataclass
class WhisperParaphasiaConfig:
    """Configuration for Whisper paraphasia model."""

    model_name: str = "openai/whisper-small"
    language: str = "en"
    task: str = "transcribe"
    freeze_encoder: bool = False
    # Initialize new token embeddings as mean of existing embeddings
    init_new_embeddings_from_mean: bool = True


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
        if config.init_new_embeddings_from_mean:
            _init_new_embeddings(model, old_vocab_size)

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
    old_vocab_size: int,
) -> None:
    """Initialize new token embeddings as the mean of existing embeddings.

    Whisper ties proj_out to embed_tokens, so this covers the output
    projection too; untied heads are handled separately just in case.
    """
    with torch.no_grad():
        embed = model.get_input_embeddings().weight
        embed[old_vocab_size:] = embed[:old_vocab_size].mean(dim=0)

        proj = model.get_output_embeddings().weight
        if proj.data_ptr() != embed.data_ptr():
            proj[old_vocab_size:] = proj[:old_vocab_size].mean(dim=0)


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
