"""Paraphasia classification head on top of Whisper decoder hidden states.

Instead of relying on the decoder to emit [p]/[n] tokens from the full 51k+
vocabulary, this module adds a small classification head that makes a 3-class
decision (correct / [p] / [n]) at each word position using the decoder's
hidden states. This reduces the problem from "pick 1 of 51k tokens" to
"pick 1 of 3 classes".

Can be used in two modes:
1. Joint training: train alongside the seq2seq loss (ASR + classification)
2. Post-hoc: freeze Whisper, train only the classification head on decoder states
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import WhisperForConditionalGeneration


class ParaphasiaClassifierHead(nn.Module):
    """Small classification head for paraphasia detection.

    Takes decoder hidden states and predicts {correct=0, [p]=1, [n]=2}
    at each token position.
    """

    NUM_CLASSES = 3  # correct, [p], [n]
    LABEL_MAP = {0: "c", 1: "p", 2: "n"}

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, self.NUM_CLASSES),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Classify each position.

        Args:
            hidden_states: (batch, seq_len, hidden_size) from decoder.

        Returns:
            Logits of shape (batch, seq_len, 3).
        """
        return self.head(hidden_states)


class WhisperWithParaphasiaHead(nn.Module):
    """Whisper model with an auxiliary paraphasia classification head.

    During training, computes both:
    1. Standard seq2seq cross-entropy loss (for ASR)
    2. Classification loss on decoder hidden states (for paraphasia detection)

    The final loss is: seq2seq_loss + alpha * classification_loss
    """

    def __init__(
        self,
        whisper_model: WhisperForConditionalGeneration,
        paraphasia_token_ids: dict[str, int],
        alpha: float = 1.0,
        class_weights: list[float] | None = None,
    ):
        """
        Args:
            whisper_model: The base Whisper model.
            paraphasia_token_ids: {"[p]": id, "[n]": id} mapping.
            alpha: Weight for classification loss relative to seq2seq loss.
            class_weights: Optional [w_correct, w_p, w_n] for CE loss.
        """
        super().__init__()
        self.whisper = whisper_model
        self.paraphasia_token_ids = paraphasia_token_ids
        self.alpha = alpha

        hidden_size = whisper_model.config.d_model
        self.classifier = ParaphasiaClassifierHead(hidden_size)

        # Build class weights for the 3-class problem
        if class_weights:
            self._cls_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self._cls_weights = None

        # Map from token IDs to class indices
        self._token_to_class = {}
        for token, tid in paraphasia_token_ids.items():
            if token == "[p]":
                self._token_to_class[tid] = 1
            elif token == "[n]":
                self._token_to_class[tid] = 2

    def forward(self, input_features, labels=None, **kwargs):
        """Forward pass with both seq2seq and classification losses."""
        # Run Whisper forward — get decoder hidden states
        outputs = self.whisper(
            input_features=input_features,
            labels=labels,
            output_hidden_states=True,
            **kwargs,
        )

        if labels is None:
            return outputs

        # Get last decoder hidden state
        decoder_hidden = outputs.decoder_hidden_states[-1]

        # Build classification targets from labels
        cls_targets = self._build_cls_targets(labels)

        # Classify
        cls_logits = self.classifier(decoder_hidden)

        # Classification loss (only on non-ignored positions)
        if self._cls_weights is not None:
            weights = self._cls_weights.to(cls_logits.device)
            cls_loss_fct = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
        else:
            cls_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

        cls_loss = cls_loss_fct(
            cls_logits.view(-1, ParaphasiaClassifierHead.NUM_CLASSES),
            cls_targets.view(-1),
        )

        # Combined loss
        total_loss = outputs.loss + self.alpha * cls_loss

        # Attach for logging
        outputs.loss = total_loss
        outputs.cls_loss = cls_loss

        return outputs

    def _build_cls_targets(self, labels: torch.Tensor) -> torch.Tensor:
        """Convert token-level labels to 3-class targets.

        For each position in the label sequence:
        - If the token is [p] → class 1
        - If the token is [n] → class 2
        - If the token is -100 (padding) → -100 (ignore)
        - Otherwise → class 0 (correct)

        But we actually want to label the WORD position, not the tag position.
        The tag [p] follows the word it labels. So we look ahead: if position
        i+1 is a paraphasia token, then position i gets that class.
        """
        batch_size, seq_len = labels.shape
        targets = torch.zeros_like(labels)  # default: correct (class 0)

        for i in range(batch_size):
            for j in range(seq_len):
                tok = labels[i, j].item()
                if tok == -100:
                    targets[i, j] = -100
                elif tok in self._token_to_class:
                    # This position IS a paraphasia tag — mark preceding word
                    targets[i, j] = -100  # ignore the tag position itself
                    if j > 0 and targets[i, j - 1] != -100:
                        targets[i, j - 1] = self._token_to_class[tok]

        return targets

    def generate(self, *args, **kwargs):
        """Delegate to Whisper's generate for inference."""
        return self.whisper.generate(*args, **kwargs)

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
    def device(self):
        return next(self.parameters()).device
