"""Whisper model setup for paraphasia detection.

Handles:
- Loading pretrained Whisper and resizing embeddings for new tokens
- Configuring generation parameters
- Optional encoder freezing for fine-tuning
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperTokenizerFast,
    GenerationConfig,
)

from .tokenizer import build_tokenizer, get_paraphasia_token_ids, PARAPHASIA_CLASS_WEIGHTS


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
    """Build Whisper model with paraphasia token support.

    Args:
        config: Model configuration. Uses defaults if None.
        tokenizer: Pre-built tokenizer. If None, builds one from config.

    Returns:
        (model, tokenizer) tuple ready for training.
    """
    if config is None:
        config = WhisperParaphasiaConfig()

    if tokenizer is None:
        tokenizer = build_tokenizer(
            model_name=config.model_name,
            language=config.language,
            task=config.task,
        )

    # Use Flash Attention 2 if available (requires flash-attn package + Ampere+ GPU)
    try:
        model = WhisperForConditionalGeneration.from_pretrained(
            config.model_name, attn_implementation="flash_attention_2"
        )
    except (ValueError, ImportError):
        model = WhisperForConditionalGeneration.from_pretrained(config.model_name)

    # Resize token embeddings to accommodate new paraphasia tokens
    old_vocab_size = model.config.vocab_size
    model.resize_token_embeddings(len(tokenizer))

    if config.init_new_embeddings_from_mean:
        _init_new_embeddings(model, old_vocab_size)

    # Freeze encoder if requested (for Phase 2 fine-tuning experiments)
    if config.freeze_encoder:
        freeze_encoder(model)

    # Configure generation
    model.generation_config = GenerationConfig.from_pretrained(
        config.model_name,
    )
    model.generation_config.language = config.language
    model.generation_config.task = config.task
    # Allow model to generate the new paraphasia tokens
    # Suppress tokens list may block new token IDs — clear it
    model.generation_config.suppress_tokens = _get_suppress_tokens(
        model.generation_config, tokenizer
    )

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
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build a per-token class weight tensor for cross-entropy loss.

    Assigns higher weights to paraphasia tokens to handle class imbalance
    (from CHAI: [p]=2, [n]=4, [s]=10, everything else=1).

    Returns:
        Tensor of shape (vocab_size,) with per-token weights.
    """
    vocab_size = len(tokenizer)
    weights = torch.ones(vocab_size, device=device)

    para_ids = get_paraphasia_token_ids(tokenizer)
    for token, token_id in para_ids.items():
        weights[token_id] = PARAPHASIA_CLASS_WEIGHTS[token]

    return weights


def _init_new_embeddings(
    model: WhisperForConditionalGeneration,
    old_vocab_size: int,
) -> None:
    """Initialize new token embeddings as the mean of existing embeddings.

    This gives new tokens a reasonable starting point rather than random
    initialization, which can help with fine-tuning stability.
    """
    with torch.no_grad():
        # Decoder input embeddings
        embed = model.model.decoder.embed_tokens.weight
        mean_embed = embed[:old_vocab_size].mean(dim=0)
        embed[old_vocab_size:] = mean_embed

        # Output projection (lm_head) shares weights with embed_tokens
        # in Whisper, so resizing handles both. But if they're separate:
        if model.proj_out is not None and model.proj_out.weight is not embed:
            proj = model.proj_out.weight
            mean_proj = proj[:old_vocab_size].mean(dim=0)
            proj[old_vocab_size:] = mean_proj


def _get_suppress_tokens(
    gen_config: GenerationConfig,
    tokenizer: WhisperTokenizerFast,
) -> list[int]:
    """Remove paraphasia token IDs from the suppress list.

    Whisper's default generation config suppresses many token IDs.
    We need to ensure our new paraphasia tokens aren't suppressed.
    """
    suppress = gen_config.suppress_tokens or []
    para_ids = set(get_paraphasia_token_ids(tokenizer).values())
    return [t for t in suppress if t not in para_ids]
