"""Inference module for trained Whisper paraphasia models.

Loads a checkpoint, decodes audio files, and outputs single-seq format
transcripts with inline paraphasia labels. The paraphasia labels come
from a classification head on the decoder hidden states, not from the
decoder vocabulary.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizerFast,
)

from .classifier import (
    WhisperWithParaphasiaHead,
    CLS_LABELS,
    CLS_CORRECT,
)


class ParaphasiaPredictor:
    """Run inference with a trained Whisper + classification head model."""

    def __init__(
        self,
        model_path: str | Path,
        device: str | None = None,
    ):
        model_path = str(model_path)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.tokenizer = WhisperTokenizerFast.from_pretrained(
            model_path, language="en", task="transcribe"
        )
        try:
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
                model_path
            )
        except OSError:
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
                "openai/whisper-small"
            )

        self.model = WhisperWithParaphasiaHead.from_pretrained(
            model_path, device=device
        )
        self.model.to(self.device)
        self.model.eval()

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
    ) -> str:
        """Transcribe audio with paraphasia labels.

        Returns single-seq format string, e.g. "the cat [p] sat on mat"
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

        # Generate ASR transcript
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_features, **self._gen_kwargs
            )

        # Classify each token position
        cls_preds = self.model.classify(input_features, generated_ids)

        # Merge ASR text with classification predictions
        return self._merge(generated_ids[0], cls_preds[0])

    def predict_batch(
        self,
        audios: list[np.ndarray],
        sampling_rate: int = 16000,
    ) -> list[str]:
        """Transcribe a batch of audio files."""
        features = self.feature_extractor(
            audios,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding="max_length",
        )
        input_features = features.input_features.to(
            device=self.device, dtype=self.model.dtype
        )

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_features, **self._gen_kwargs
            )

        cls_preds = self.model.classify(input_features, generated_ids)

        return [
            self._merge(generated_ids[i], cls_preds[i])
            for i in range(len(audios))
        ]

    def predict_file(self, audio_path: str | Path, **kwargs) -> str:
        """Transcribe an audio file."""
        import librosa

        audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)
        return self.predict(audio, sampling_rate=sr, **kwargs)

    def _merge(
        self, token_ids: torch.Tensor, cls_preds: torch.Tensor
    ) -> str:
        """Merge decoded text with classification predictions.

        Decodes token by token, inserting [p]/[n] after words classified
        as paraphasic. Normalizes output to lowercase, no punctuation
        (matching CHAI reference format).
        """
        # Decode full text first (for cleanup)
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)

        # Normalize: lowercase, strip punctuation (refs are lowercase no-punct)
        text = text.lower()
        text = re.sub(r"[^\w\s']", "", text)  # keep apostrophes for contractions
        text = re.sub(r"\s+", " ", text).strip()

        words = text.split()
        if not words:
            return ""

        # Get per-token predictions, skipping prefix tokens
        prefix_len = len(self.tokenizer.prefix_tokens)
        preds = cls_preds[prefix_len:].tolist()

        # Map token positions back to words by tokenizing each word
        result = []
        token_pos = 0
        for word in words:
            word_tokens = self.tokenizer.encode(word, add_special_tokens=False)
            n_tokens = len(word_tokens)

            # Take the majority vote across the word's tokens
            word_preds = preds[token_pos: token_pos + n_tokens]
            if word_preds:
                # Use max (most "paraphasic") prediction for the word
                word_cls = max(word_preds)
            else:
                word_cls = CLS_CORRECT

            result.append(word)
            if word_cls != CLS_CORRECT:
                tag = CLS_LABELS[word_cls]
                result.append(f"[{tag}]")

            token_pos += n_tokens

        return " ".join(result)
