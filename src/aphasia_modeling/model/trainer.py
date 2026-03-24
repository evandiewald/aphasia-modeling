"""Custom trainer with class-weighted loss for paraphasia tokens."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import Seq2SeqTrainer


class ParaphasiaTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer with optional per-token class weighting.

    Overrides compute_loss to apply higher weights to paraphasia tokens
    ([p]=2, [n]=4, [s]=10) to handle class imbalance.
    """

    def __init__(self, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(**kwargs)
        self._class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if self._class_weights is None:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Move weights to same device as logits
        weights = self._class_weights.to(logits.device)

        loss_fct = nn.CrossEntropyLoss(
            weight=weights,
            ignore_index=-100,
        )

        # Reshape for cross entropy: (batch * seq_len, vocab_size)
        loss = loss_fct(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )

        return (loss, outputs) if return_outputs else loss
