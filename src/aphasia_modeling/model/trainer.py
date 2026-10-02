"""Seq2SeqTrainer with optional per-token loss weighting for paraphasia tags."""

from __future__ import annotations

import torch
from torch import nn
from transformers import Seq2SeqTrainer


class ParaphasiaTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer whose cross-entropy can upweight the tag tokens.

    Off by default, matching CHAI's single-seq (plain CE); exposed for sweeps.
    """

    def __init__(self, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(**kwargs)
        self._class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if self._class_weights is None:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        # Labels stay in inputs so the model derives decoder_input_ids;
        # the loss is then recomputed with weights from the logits.
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = nn.CrossEntropyLoss(
            weight=self._class_weights.to(device=logits.device, dtype=logits.dtype),
            ignore_index=-100,
        )
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return (loss, outputs) if return_outputs else loss
