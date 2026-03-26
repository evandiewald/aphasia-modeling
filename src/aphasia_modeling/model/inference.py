"""Inference module for trained Whisper paraphasia models.

Loads a checkpoint, decodes audio, and classifies utterances for paraphasia
presence. The ASR transcript comes from Whisper's decoder. Paraphasia labels
come from an utterance-level classification head on decoder hidden states.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizerFast,
)

from .classifier import (
    WhisperWithParaphasiaHead,
    UTT_PHONEMIC,
    UTT_NEOLOGISTIC,
)


@dataclass
class PredictionResult:
    """Result of paraphasia prediction for a single utterance."""
    text: str           # Normalized ASR transcript (lowercase, no punct)
    has_p: bool         # Utterance contains phonemic paraphasia
    has_n: bool         # Utterance contains neologistic paraphasia
    prob_p: float       # Probability of phonemic paraphasia
    prob_n: float       # Probability of neologistic paraphasia

    def to_single_seq(self) -> str:
        """Format as single-seq string with tags for eval compatibility.

        Inserts [p]/[n] after the first word if flagged.
        """
        words = self.text.split()
        if not words:
            return ""

        result = [words[0]]
        if self.has_p:
            result.append("[p]")
        if self.has_n:
            result.append("[n]")
        result.extend(words[1:])
        return " ".join(result)


class ParaphasiaPredictor:
    """Run inference with a trained Whisper + utterance classification model."""

    def __init__(
        self,
        model_path: str | Path,
        device: str | None = None,
        threshold: float = 0.5,
    ):
        """
        Args:
            model_path: Path to saved checkpoint.
            device: Device to use.
            threshold: Minimum sigmoid probability to flag a paraphasia type.
        """
        model_path = str(model_path)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.threshold = threshold

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
    ) -> PredictionResult:
        """Transcribe audio and classify for paraphasias."""
        results = self.predict_batch([audio], sampling_rate=sampling_rate)
        return results[0]

    def predict_batch(
        self,
        audios: list[np.ndarray],
        sampling_rate: int = 16000,
    ) -> list[PredictionResult]:
        """Transcribe a batch of audio and classify for paraphasias."""
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

        # Classify utterances
        cls_probs = self.model.classify(input_features, generated_ids)  # (batch, 2)

        results = []
        for i in range(len(audios)):
            # Decode and normalize
            text = self.tokenizer.decode(
                generated_ids[i], skip_special_tokens=True
            )
            text = text.lower()
            text = re.sub(r"[^\w\s']", "", text)
            text = re.sub(r"\s+", " ", text).strip()

            prob_p = cls_probs[i, UTT_PHONEMIC].item()
            prob_n = cls_probs[i, UTT_NEOLOGISTIC].item()

            results.append(PredictionResult(
                text=text,
                has_p=prob_p >= self.threshold,
                has_n=prob_n >= self.threshold,
                prob_p=prob_p,
                prob_n=prob_n,
            ))

        return results

    def predict_file(self, audio_path: str | Path, **kwargs) -> PredictionResult:
        """Transcribe an audio file."""
        import librosa

        audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)
        return self.predict(audio, sampling_rate=sr, **kwargs)
