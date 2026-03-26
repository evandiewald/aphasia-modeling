"""Paraphasia classification head on top of Whisper decoder hidden states.

Instead of relying on the decoder to emit [p]/[n] tokens from the full 51k+
vocabulary, this module adds a small classification head that makes a 3-class
decision (correct / [p] / [n]) at each word position using the decoder's
hidden states. This reduces the problem from "pick 1 of 51k tokens" to
"pick 1 of 3 classes".

The decoder does pure ASR (no paraphasia tokens in vocab). The classification
head is trained jointly via an auxiliary loss on the decoder hidden states.
At inference time, the head's predictions are interleaved into the transcript.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from transformers import WhisperForConditionalGeneration


# Classification labels
CLS_CORRECT = 0
CLS_PHONEMIC = 1
CLS_NEOLOGISTIC = 2
CLS_LABELS = {CLS_CORRECT: "c", CLS_PHONEMIC: "p", CLS_NEOLOGISTIC: "n"}


class ParaphasiaClassifierHead(nn.Module):
    """Small classification head for paraphasia detection.

    Takes decoder hidden states and predicts {correct=0, [p]=1, [n]=2}
    at each token position.
    """

    NUM_CLASSES = 3

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

    The decoder does pure ASR — no paraphasia tokens in the vocabulary.
    The classification head trains on decoder hidden states to detect
    paraphasias as a 3-class problem at each token position.

    Training loss: asr_loss + alpha * classification_loss
    """

    def __init__(
        self,
        whisper_model: WhisperForConditionalGeneration,
        alpha: float = 1.0,
        cls_class_weights: list[float] | None = None,
    ):
        """
        Args:
            whisper_model: The base Whisper model (unmodified vocab).
            alpha: Weight for classification loss relative to ASR loss.
            cls_class_weights: Optional [w_correct, w_p, w_n] for the
                classification CE loss.
        """
        super().__init__()
        self.whisper = whisper_model
        self.alpha = alpha

        hidden_size = whisper_model.config.d_model
        self.classifier = ParaphasiaClassifierHead(hidden_size)

        if cls_class_weights:
            self.register_buffer(
                "_cls_weights",
                torch.tensor(cls_class_weights, dtype=torch.float32),
            )
        else:
            self._cls_weights = None

    def forward(
        self,
        input_features,
        labels=None,
        cls_labels=None,
        **kwargs,
    ):
        """Forward pass with both ASR and classification losses.

        Args:
            input_features: Mel spectrogram features.
            labels: Token IDs for ASR loss (standard Whisper labels).
            cls_labels: Per-token classification targets (0=correct, 1=[p],
                2=[n], -100=ignore). Same shape as labels.
        """
        outputs = self.whisper(
            input_features=input_features,
            labels=labels,
            output_hidden_states=True,
            **kwargs,
        )

        if cls_labels is not None:
            decoder_hidden = outputs.decoder_hidden_states[-1]
            cls_logits = self.classifier(decoder_hidden)

            if self._cls_weights is not None:
                weights = self._cls_weights.to(cls_logits.device)
                cls_loss_fct = nn.CrossEntropyLoss(
                    weight=weights, ignore_index=-100
                )
            else:
                cls_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

            cls_loss = cls_loss_fct(
                cls_logits.view(-1, ParaphasiaClassifierHead.NUM_CLASSES),
                cls_labels.view(-1),
            )

            outputs.loss = outputs.loss + self.alpha * cls_loss

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
            Predicted class per position (batch, seq_len) — 0/1/2.
        """
        with torch.no_grad():
            outputs = self.whisper(
                input_features=input_features,
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=True,
            )
            decoder_hidden = outputs.decoder_hidden_states[-1]
            cls_logits = self.classifier(decoder_hidden)
            return cls_logits.argmax(dim=-1)

    def generate(self, *args, **kwargs):
        """Delegate to Whisper's generate for ASR inference."""
        return self.whisper.generate(*args, **kwargs)

    def save_pretrained(self, path: str | Path) -> None:
        """Save Whisper (HF format) and classifier head (safetensors)."""
        from safetensors.torch import save_file

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save Whisper in standard HF format (handles shared tensors)
        self.whisper.save_pretrained(path)

        # Save classifier head separately
        save_file(self.classifier.state_dict(), path / "classifier_head.safetensors")

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        alpha: float = 1.0,
        cls_class_weights: list[float] | None = None,
        device: str | None = None,
    ) -> WhisperWithParaphasiaHead:
        """Load Whisper and classifier head from a checkpoint."""
        from safetensors.torch import load_file

        path = Path(path)
        whisper = WhisperForConditionalGeneration.from_pretrained(path)
        model = cls(whisper, alpha=alpha, cls_class_weights=cls_class_weights)

        head_path = path / "classifier_head.safetensors"
        if head_path.exists():
            state = load_file(head_path, device=device or "cpu")
            model.classifier.load_state_dict(state)

        return model

    # --- Properties needed by HF Trainer ---

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
