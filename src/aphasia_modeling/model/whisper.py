"""Whisper model setup for paraphasia detection.

Handles:
- Loading pretrained Whisper (unmodified vocabulary)
- Wrapping with WhisperWithParaphasiaHead for classification
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
    # Class weights for the 3-class classification head [correct, p, n]
    cls_class_weights: list[float] | None = None


def build_model(
    config: WhisperParaphasiaConfig | None = None,
    tokenizer: WhisperTokenizerFast | None = None,
) -> tuple[WhisperWithParaphasiaHead, WhisperTokenizerFast]:
    """Build Whisper model with paraphasia classification head.

    The decoder vocabulary is unmodified — no special tokens added.
    Paraphasia detection is handled by a separate classification head
    on the decoder hidden states.

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
        cls_class_weights=config.cls_class_weights,
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


def freeze_decoder(model: WhisperForConditionalGeneration) -> None:
    """Freeze all decoder and output projection parameters."""
    for param in model.model.decoder.parameters():
        param.requires_grad = False
    if model.proj_out is not None:
        for param in model.proj_out.parameters():
            param.requires_grad = False
