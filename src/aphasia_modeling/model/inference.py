"""Inference module for trained Whisper paraphasia models.

Loads a checkpoint, decodes audio files, and outputs single-seq format
transcripts with inline paraphasia labels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperFeatureExtractor,
    WhisperTokenizerFast,
)


class ParaphasiaPredictor:
    """Run inference with a trained Whisper paraphasia model."""

    def __init__(
        self,
        model_path: str | Path,
        device: str | None = None,
    ):
        """Load model, tokenizer, and feature extractor from a checkpoint.

        Args:
            model_path: Path to saved checkpoint directory.
            device: Device to use ("cuda", "cpu", etc.). Auto-detects if None.
        """
        model_path = str(model_path)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.tokenizer = WhisperTokenizerFast.from_pretrained(model_path)
        # Feature extractor may not be saved in intermediate checkpoints —
        # fall back to base whisper config from the model's config
        try:
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)
        except OSError:
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        # Build generation kwargs — force English transcription and prevent loops
        self._gen_kwargs = {
            "max_length": 448,
            "language": "en",
            "task": "transcribe",
            "no_repeat_ngram_size": 3,
        }

    def predict(
        self,
        audio: np.ndarray,
        sampling_rate: int = 16000,
        max_length: int = 448,
    ) -> str:
        """Transcribe audio with paraphasia labels.

        Args:
            audio: Audio waveform as float32 numpy array.
            sampling_rate: Audio sampling rate.
            max_length: Maximum generation length.

        Returns:
            Single-seq format string, e.g. "the cat [p] sat on mat [s]"
        """
        features = self.feature_extractor(
            audio,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding="max_length",
        )
        input_features = features.input_features.to(
            device=self.device, dtype=self.model.dtype
        )

        # Whisper doesn't use encoder attention mask — it always pads to 30s.
        # Pass a dummy attention mask to suppress the false-positive warning.
        attention_mask = torch.ones_like(input_features[:, 0, :], dtype=torch.long)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_features,
                attention_mask=attention_mask,
                **self._gen_kwargs,
            )

        # Decode, skipping special tokens except paraphasia tags
        text = self._decode(generated_ids[0])
        return text

    def predict_batch(
        self,
        audios: list[np.ndarray],
        sampling_rate: int = 16000,
        max_length: int = 448,
    ) -> list[str]:
        """Transcribe a batch of audio files.

        Args:
            audios: List of audio waveforms.
            sampling_rate: Audio sampling rate.
            max_length: Maximum generation length.

        Returns:
            List of single-seq format strings.
        """
        features = self.feature_extractor(
            audios,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding="max_length",
        )
        input_features = features.input_features.to(
            device=self.device, dtype=self.model.dtype
        )
        attention_mask = torch.ones_like(input_features[:, 0, :], dtype=torch.long)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_features,
                attention_mask=attention_mask,
                **self._gen_kwargs,
            )

        return [self._decode(ids) for ids in generated_ids]

    def predict_file(self, audio_path: str | Path, **kwargs) -> str:
        """Transcribe an audio file.

        Args:
            audio_path: Path to .wav file.
            **kwargs: Passed to predict().

        Returns:
            Single-seq format string.
        """
        import librosa

        audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)
        return self.predict(audio, sampling_rate=sr, **kwargs)

    def _decode(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs to text, preserving paraphasia tags.

        Whisper's decode with skip_special_tokens=True would strip our
        paraphasia tokens. Instead we decode all tokens and then clean
        up the Whisper-specific special tokens manually.
        """
        # Decode without skipping special tokens
        text = self.tokenizer.decode(token_ids, skip_special_tokens=False)

        # Remove Whisper's special tokens but keep [p], [n], [s]
        # Whisper special tokens look like <|en|>, <|transcribe|>, <|startoftranscript|>, etc.
        import re
        text = re.sub(r"<\|[^|]*\|>", "", text)

        # Clean up whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text
