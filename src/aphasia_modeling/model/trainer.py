"""Custom trainer for Whisper + paraphasia classification head."""

from __future__ import annotations

from transformers import Seq2SeqTrainer


class ParaphasiaTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer that passes cls_labels through to the model.

    The WhisperWithParaphasiaHead model handles the combined loss
    (ASR + classification) internally. This trainer just ensures
    cls_labels from the collator reach the model's forward method.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # The model's forward() accepts cls_labels and computes
        # combined loss internally, so we just call it directly.
        labels = inputs["labels"]
        cls_labels = inputs.get("cls_labels")

        outputs = model(
            input_features=inputs["input_features"],
            labels=labels,
            cls_labels=cls_labels,
        )

        return (outputs.loss, outputs) if return_outputs else outputs.loss
