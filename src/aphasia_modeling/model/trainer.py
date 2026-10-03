"""Seq2SeqTrainer with paraphasia-tag specific training and eval hooks."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from transformers import EvalPrediction, Seq2SeqTrainer


class ParaphasiaTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer with optional tag-token loss weighting and tag LR scaling.

    class_weights: per-token CE weights (off by default, matching CHAI's
        single-seq plain CE).
    tag_token_ids / tag_lr_scale: the tag embedding rows take steps
        `tag_lr_scale` times larger than the rest of the model. They share one
        (tied) embedding matrix with every other token, so this can't be an
        optimizer param group, and scaling their gradient does nothing under
        Adam; instead each optimizer step's update to those rows is amplified.
    """

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        tag_token_ids: list[int] | None = None,
        tag_lr_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._class_weights = class_weights
        self._tag_token_ids = tag_token_ids or []
        self._tag_lr_scale = tag_lr_scale

    def create_optimizer(self, *args, **kwargs):
        optimizer = super().create_optimizer(*args, **kwargs)
        if self._tag_lr_scale != 1.0 and self._tag_token_ids:
            _scale_row_updates(
                optimizer,
                self.model.get_input_embeddings().weight,
                self._tag_token_ids,
                self._tag_lr_scale,
            )
        return optimizer

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


def _scale_row_updates(
    optimizer: torch.optim.Optimizer,
    weight: torch.nn.Parameter,
    rows: list[int],
    scale: float,
) -> None:
    """Make each optimizer step move weight[rows] `scale` times further."""
    idx = torch.tensor(rows, device=weight.device)
    snapshot: dict[str, torch.Tensor] = {}

    def pre_hook(opt, args, kwargs):
        snapshot["rows"] = weight.detach()[idx].clone()

    def post_hook(opt, args, kwargs):
        with torch.no_grad():
            before = snapshot.pop("rows")
            delta = weight[idx] - before
            weight[idx] = before + scale * delta

    # HF may wrap the optimizer (e.g. accelerate); hook the underlying one
    inner = getattr(optimizer, "optimizer", optimizer)
    inner.register_step_pre_hook(pre_hook)
    inner.register_step_post_hook(post_hook)


# ---- Teacher-forced tag metrics ----------------------------------------------


def make_tag_logits_preprocessor(tag_token_ids: list[int]):
    """Reduce eval logits to (argmax over vocab, argmax over the tags only).

    Keeps eval memory small: [B, T, V] logits become [B, T, 2] ids.
    """
    tag_idx = torch.tensor(tag_token_ids)

    def preprocess(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        full = logits.argmax(dim=-1)
        among_tags = tag_idx.to(logits.device)[logits[..., tag_idx.to(logits.device)].argmax(dim=-1)]
        return torch.stack([full, among_tags], dim=-1)

    return preprocess


def make_tag_metrics(tag_ids: dict[str, int]):
    """compute_metrics reporting how the model handles tags under teacher forcing.

    Per tag: how often the model's top token is that tag (`pred_*`) vs. how
    often the reference has it (`ref_*`), and `recall_*`, the fraction of
    reference positions of that tag where it is the top token. `tag_class_acc`
    asks only "which tag" at reference tag positions (argmax over the tags),
    so it tracks whether the tags have separated even before they're emitted.
    A collapse onto one tag shows up as one pred_* count and zeros elsewhere.
    """

    def compute(pred: EvalPrediction) -> dict[str, float]:
        ids = np.asarray(pred.predictions)
        labels = np.asarray(pred.label_ids)
        full, among_tags = ids[..., 0], ids[..., 1]
        valid = labels != -100
        is_tag = np.isin(labels, list(tag_ids.values()))

        metrics = {}
        for tag, tid in tag_ids.items():
            name = tag.strip("[]")
            ref = labels == tid
            metrics[f"ref_{name}"] = int(ref.sum())
            metrics[f"pred_{name}"] = int(((full == tid) & valid).sum())
            metrics[f"recall_{name}"] = float((full[ref] == tid).mean()) if ref.any() else 0.0
        metrics["tag_class_acc"] = (
            float((among_tags[is_tag] == labels[is_tag]).mean()) if is_tag.any() else 0.0
        )
        return metrics

    return compute
