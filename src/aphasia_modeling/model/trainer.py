"""Custom trainer for Whisper + paraphasia classification head."""

from __future__ import annotations

from pathlib import Path

from transformers import Seq2SeqTrainer

from .classifier import WhisperWithParaphasiaHead


class ParaphasiaTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer for WhisperWithParaphasiaHead.

    - Passes cls_labels through to the model via compute_loss
    - Handles save with shared Whisper tensors
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(
            input_features=inputs["input_features"],
            labels=inputs["labels"],
            cls_labels=inputs.get("cls_labels"),
        )

        return (outputs.loss, outputs) if return_outputs else outputs.loss

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
