"""Whisper model setup for paraphasia detection.

Handles:
- Loading pretrained Whisper (unmodified vocabulary)
- Wrapping with WhisperWithParaphasiaHead for utterance classification
- Configuring generation parameters
- Optional encoder/decoder freezing for head-only training
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperTokenizerFast,
    GenerationConfig,
)

from .classifier import WhisperWithParaphasiaHead


@dataclass
class WhisperParaphasiaConfig:
    """Configuration for Whisper paraphasia model."""

    model_name: str = "openai/whisper-small"
    language: str = "en"
    task: str = "transcribe"
    freeze_encoder: bool = False
    freeze_decoder: bool = False
    # Classification head weight relative to ASR loss
    cls_alpha: float = 1.0
    # Positive class weights for BCE loss: [pw_phonemic, pw_neologistic]
    cls_pos_weights: list[float] | None = None


def build_model(
    config: WhisperParaphasiaConfig | None = None,
    tokenizer: WhisperTokenizerFast | None = None,
) -> tuple[WhisperWithParaphasiaHead, WhisperTokenizerFast]:
    """Build Whisper model with utterance-level paraphasia classification head.

    Args:
        config: Model configuration. Uses defaults if None.
        tokenizer: Pre-built tokenizer. If None, loads stock Whisper tokenizer.

    Returns:
        (model, tokenizer) tuple ready for training.
    """
    if config is None:
        config = WhisperParaphasiaConfig()

    if tokenizer is None:
        tokenizer = WhisperTokenizerFast.from_pretrained(
            config.model_name,
            language=config.language,
            task=config.task,
        )

    # Load Whisper with SDPA if available
    try:
        whisper = WhisperForConditionalGeneration.from_pretrained(
            config.model_name, attn_implementation="sdpa"
        )
    except (ValueError, ImportError):
        whisper = WhisperForConditionalGeneration.from_pretrained(
            config.model_name
        )

    # Freeze encoder/decoder if requested
    if config.freeze_encoder:
        freeze_encoder(whisper)
    if config.freeze_decoder:
        freeze_decoder(whisper)

    # Configure generation
    whisper.generation_config = GenerationConfig.from_pretrained(
        config.model_name,
    )
    whisper.generation_config.language = config.language
    whisper.generation_config.task = config.task

    # Wrap with classification head
    model = WhisperWithParaphasiaHead(
        whisper,
        alpha=config.cls_alpha,
        cls_pos_weights=config.cls_pos_weights,
    )

    # If both encoder and decoder are frozen, only train the head
    if config.freeze_encoder and config.freeze_decoder:
        model.cls_only = True

    return model, tokenizer


def freeze_encoder(model: WhisperForConditionalGeneration) -> None:
    """Freeze all encoder parameters."""
    for param in model.model.encoder.parameters():
        param.requires_grad = False


def unfreeze_encoder(model: WhisperForConditionalGeneration) -> None:
    """Unfreeze all encoder parameters."""
    for param in model.model.encoder.parameters():
        param.requires_grad = True


def freeze_decoder(model: WhisperForConditionalGeneration) -> None:
    """Freeze all decoder and output projection parameters."""
    for param in model.model.decoder.parameters():
        param.requires_grad = False
    if model.proj_out is not None:
        for param in model.proj_out.parameters():
            param.requires_grad = False
