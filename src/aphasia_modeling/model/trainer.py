"""Custom trainer with class-weighted loss and logit bias for paraphasia tokens."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import Seq2SeqTrainer


class ParaphasiaTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer with optional per-token class weighting and logit bias.

    Supports two mechanisms for handling class imbalance:
    - class_weights: Per-token weights in the cross-entropy loss
    - logit_bias: Additive bias on paraphasia token logits before loss,
      making it easier for the model to emit them (they compete against 51k+ tokens)
    """

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        paraphasia_logit_bias: dict[int, float] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._class_weights = class_weights
        self._logit_bias = paraphasia_logit_bias  # {token_id: bias_value}

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if self._class_weights is None and self._logit_bias is None:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits

        # Apply logit bias: boost paraphasia token logits before loss
        if self._logit_bias:
            for token_id, bias in self._logit_bias.items():
                logits[:, :, token_id] += bias

        if self._class_weights is not None:
            weights = self._class_weights.to(logits.device)
            loss_fct = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
        else:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

        loss = loss_fct(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )

        return (loss, outputs) if return_outputs else loss
