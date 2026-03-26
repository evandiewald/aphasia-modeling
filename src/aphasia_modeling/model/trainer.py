"""Custom trainer for Whisper + paraphasia classification head."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import Seq2SeqTrainer

from .classifier import WhisperWithParaphasiaHead, CLS_CORRECT, CLS_PHONEMIC, CLS_NEOLOGISTIC


class ParaphasiaTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer for WhisperWithParaphasiaHead.

    - Passes cls_labels through to the model
    - Logs ASR/classification losses separately
    - Computes token-level classification accuracy and F1 during eval
    - Handles save with shared Whisper tensors
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        cls_labels = inputs.get("cls_labels")

        outputs = model(
            input_features=inputs["input_features"],
            labels=labels,
            cls_labels=cls_labels,
        )

        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def evaluation_loop(self, dataloader, description, **kwargs):
        """Override eval loop to compute classification metrics."""
        model = self.model
        model.eval()

        all_cls_preds = []
        all_cls_targets = []
        total_loss = 0.0
        total_asr_loss = 0.0
        total_cls_loss = 0.0
        n_batches = 0

        for inputs in dataloader:
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                outputs = model(
                    input_features=inputs["input_features"],
                    labels=inputs["labels"],
                    cls_labels=inputs.get("cls_labels"),
                    output_hidden_states=True,
                )

            total_loss += outputs.loss.item()
            if hasattr(outputs, "cls_loss"):
                total_cls_loss += outputs.cls_loss.item()
                total_asr_loss += (outputs.loss - model.alpha * outputs.cls_loss).item()

                # Get classification predictions
                decoder_hidden = outputs.decoder_hidden_states[-1]
                cls_logits = model.classifier(decoder_hidden)
                cls_preds = cls_logits.argmax(dim=-1)  # (batch, seq_len)
                cls_targets = inputs["cls_labels"]

                # Only keep non-ignored positions
                mask = cls_targets != -100
                all_cls_preds.append(cls_preds[mask].cpu())
                all_cls_targets.append(cls_targets[mask].cpu())

            n_batches += 1

        # Compute metrics
        metrics = {
            "eval_loss": total_loss / max(n_batches, 1),
            "eval_asr_loss": total_asr_loss / max(n_batches, 1),
            "eval_cls_loss": total_cls_loss / max(n_batches, 1),
        }

        if all_cls_preds:
            preds = torch.cat(all_cls_preds)
            targets = torch.cat(all_cls_targets)
            cls_metrics = _compute_cls_metrics(preds, targets)
            metrics.update(cls_metrics)

        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )

        return metrics

    def _save(self, output_dir=None, state_dict=None):
        """Override default save to handle shared Whisper tensors."""
        output_dir = output_dir or self.args.output_dir
        model = self.model

        if isinstance(model, WhisperWithParaphasiaHead):
            model.save_pretrained(output_dir)
        else:
            super()._save(output_dir, state_dict=state_dict)

        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)


def _compute_cls_metrics(
    preds: torch.Tensor, targets: torch.Tensor
) -> dict[str, float]:
    """Compute token-level classification metrics."""
    metrics = {}

    # Overall accuracy
    metrics["eval_cls_accuracy"] = (preds == targets).float().mean().item()

    # Per-class precision, recall, F1
    for cls_id, name in [(CLS_PHONEMIC, "p"), (CLS_NEOLOGISTIC, "n")]:
        pred_pos = preds == cls_id
        true_pos = targets == cls_id
        tp = (pred_pos & true_pos).sum().float()
        fp = (pred_pos & ~true_pos).sum().float()
        fn = (~pred_pos & true_pos).sum().float()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[f"eval_cls_f1_{name}"] = float(f1)
        metrics[f"eval_cls_precision_{name}"] = float(precision)
        metrics[f"eval_cls_recall_{name}"] = float(recall)

    return metrics
