"""Utterance-level paraphasia classification head on Whisper decoder hidden states.

The decoder does pure ASR (unmodified Whisper vocab). This module adds a
multilabel classification head that pools decoder hidden states and predicts
whether the utterance contains phonemic and/or neologistic paraphasias.

This is a 2-output binary (multilabel) problem:
  - has_phonemic (0/1)
  - has_neologistic (0/1)
An utterance can have both, either, or neither.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from transformers import WhisperForConditionalGeneration

# Indices into the 2-output classification vector
UTT_PHONEMIC = 0
UTT_NEOLOGISTIC = 1


class UtteranceClassifierHead(nn.Module):
    """Multilabel classification head for utterance-level paraphasia detection.

    Takes a pooled decoder hidden state and predicts two independent binary
    labels: has_phonemic and has_neologistic.
    """

    NUM_LABELS = 2  # [has_p, has_n]

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, self.NUM_LABELS),
        )

    def forward(self, pooled_hidden: torch.Tensor) -> torch.Tensor:
        """Classify utterance.

        Args:
            pooled_hidden: (batch, hidden_size) — mean-pooled decoder states.

        Returns:
            Logits of shape (batch, 2) for [has_p, has_n].
        """
        return self.head(pooled_hidden)


class WhisperWithParaphasiaHead(nn.Module):
    """Whisper model with utterance-level paraphasia classification head.

    The decoder does pure ASR. The classification head pools decoder hidden
    states and predicts whether the utterance contains [p] and/or [n].

    Training loss: asr_loss + alpha * bce_classification_loss
    (or cls_only mode: just bce_classification_loss)
    """

    def __init__(
        self,
        whisper_model: WhisperForConditionalGeneration,
        alpha: float = 1.0,
        cls_pos_weights: list[float] | None = None,
    ):
        """
        Args:
            whisper_model: The base Whisper model (unmodified vocab).
            alpha: Weight for classification loss relative to ASR loss.
            cls_pos_weights: Positive class weights for BCEWithLogitsLoss,
                as [pw_phonemic, pw_neologistic]. Upweights positive examples.
        """
        super().__init__()
        self.whisper = whisper_model
        self.alpha = alpha
        self.cls_only = False  # Set True to skip ASR loss (head-only training)

        hidden_size = whisper_model.config.d_model
        self.classifier = UtteranceClassifierHead(hidden_size)

        if cls_pos_weights:
            self.register_buffer(
                "_pos_weights",
                torch.tensor(cls_pos_weights, dtype=torch.float32),
            )
        else:
            self._pos_weights = None

    def forward(
        self,
        input_features,
        labels=None,
        cls_labels=None,
        **kwargs,
    ):
        """Forward pass with ASR and/or classification loss.

        Args:
            input_features: Mel spectrogram features.
            labels: Token IDs for ASR loss.
            cls_labels: Utterance-level binary labels, shape (batch, 2),
                values 0.0 or 1.0 for [has_p, has_n].
        """
        kwargs.pop("output_hidden_states", None)
        outputs = self.whisper(
            input_features=input_features,
            labels=labels,
            output_hidden_states=True,
            **kwargs,
        )

        if cls_labels is not None:
            # Mean-pool decoder hidden states over non-padding positions
            decoder_hidden = outputs.decoder_hidden_states[-1]  # (batch, seq, hidden)
            mask = (labels != -100).float()  # (batch, seq)
            mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=1)  # avoid div by 0
            pooled = (decoder_hidden * mask.unsqueeze(-1)).sum(dim=1) / mask_sum  # (batch, hidden)

            cls_logits = self.classifier(pooled)  # (batch, 2)

            loss_fct = nn.BCEWithLogitsLoss(
                pos_weight=self._pos_weights.to(cls_logits.device) if self._pos_weights is not None else None,
            )
            cls_loss = loss_fct(cls_logits, cls_labels.float())

            if self.cls_only:
                outputs.loss = cls_loss
            else:
                outputs.loss = outputs.loss + self.alpha * cls_loss

            self._last_cls_loss = cls_loss.item()
        else:
            self._last_cls_loss = None

        return outputs

    def classify(
        self,
        input_features: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run classification only (for inference).

        Args:
            input_features: Mel features (batch, n_mels, frames).
            decoder_input_ids: Token IDs from generate() output.

        Returns:
            Sigmoid probabilities, shape (batch, 2) for [has_p, has_n].
        """
        with torch.no_grad():
            outputs = self.whisper(
                input_features=input_features,
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=True,
            )
            decoder_hidden = outputs.decoder_hidden_states[-1]

            # Pool over all positions (no label masking at inference)
            pooled = decoder_hidden.mean(dim=1)

            cls_logits = self.classifier(pooled)
            return torch.sigmoid(cls_logits)

    def generate(self, *args, **kwargs):
        """Delegate to Whisper's generate for ASR inference."""
        return self.whisper.generate(*args, **kwargs)

    def save_pretrained(self, path: str | Path) -> None:
        """Save Whisper (HF format) and classifier head (safetensors)."""
        from safetensors.torch import save_file

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.whisper.save_pretrained(path)
        save_file(self.classifier.state_dict(), path / "classifier_head.safetensors")

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        alpha: float = 1.0,
        cls_pos_weights: list[float] | None = None,
        device: str | None = None,
    ) -> WhisperWithParaphasiaHead:
        """Load Whisper and classifier head from a checkpoint."""
        from safetensors.torch import load_file

        path = Path(path)
        whisper = WhisperForConditionalGeneration.from_pretrained(path)
        model = cls(whisper, alpha=alpha, cls_pos_weights=cls_pos_weights)

        head_path = path / "classifier_head.safetensors"
        if head_path.exists():
            state = load_file(head_path, device=device or "cpu")
            model.classifier.load_state_dict(state)

        return model

    # --- Properties needed by HF Trainer ---

    _keys_to_ignore_on_save = None
    _keys_to_ignore_on_load_missing = None

    @property
    def config(self):
        return self.whisper.config

    @property
    def generation_config(self):
        return self.whisper.generation_config

    @generation_config.setter
    def generation_config(self, value):
        self.whisper.generation_config = value

    @property
    def dtype(self):
        return next(self.parameters()).dtype
