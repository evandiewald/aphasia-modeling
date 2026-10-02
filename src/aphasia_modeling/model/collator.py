"""Data collator for Whisper single-seq paraphasia fine-tuning.

Handles:
- Audio feature extraction via WhisperFeatureExtractor
- Target tokenization in single-seq format ("word [p] word word [n]")
- Choosing which tag classes are trained on (others become plain words)
- Padding and label masking (-100 for pad tokens in labels)
- Time perturbation (speed changes at CHAI's rates)
"""

from __future__ import annotations

import random
import warnings
from dataclasses import dataclass
from typing import Any

import librosa
import numpy as np
import torch
from transformers import WhisperFeatureExtractor, WhisperTokenizerFast

from ..data.preprocess import to_single_seq
from .tokenizer import format_target

warnings.filterwarnings("ignore", message=".*audioread.*")
warnings.filterwarnings("ignore", message=".*PySoundFile failed.*")

# CHAI time perturbation rates
SPEC_AUGMENT_RATES = [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]


@dataclass
class ParaphasiaDataCollator:
    """Collator for Whisper single-seq paraphasia training.

    Expects each example to have:
    - "audio_path" (with "start_time"/"end_time" in seconds)
      OR "audio": dict with "array" and "sampling_rate"
    - "text": space-separated words
    - "labels": space-separated per-word labels ("c p c c n")

    Returns dict with:
    - "input_features": mel spectrogram tensor
    - "labels": target token IDs, -100 on padding
    """

    feature_extractor: WhisperFeatureExtractor
    tokenizer: WhisperTokenizerFast
    apply_time_perturbation: bool = False
    # Tag classes to train on; words with other labels are untagged targets
    tag_classes: tuple[str, ...] = ("p", "n", "s")

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if features[0].get("audio_path"):
            arrays = [self._load_segment(f) for f in features]
        else:
            arrays = [self._raw_audio(f) for f in features]

        if self.apply_time_perturbation:
            arrays = [self._time_perturb(a) for a in arrays]

        input_features = self.feature_extractor(
            arrays,
            sampling_rate=self.feature_extractor.sampling_rate,
            return_tensors="pt",
            padding="max_length",
        ).input_features

        return {
            "input_features": input_features,
            "labels": self._tokenize_targets(features),
        }

    def target_text(self, feature: dict[str, Any]) -> str:
        """Single-seq target string for one example, restricted to tag_classes."""
        words = feature["text"].split()
        labels = feature["labels"].split()
        labels = [l if l in self.tag_classes else "c" for l in labels]
        return format_target(to_single_seq(words, labels))

    def _raw_audio(self, feature: dict[str, Any]) -> np.ndarray:
        audio = feature["audio"]
        array = audio["array"] if isinstance(audio, dict) else audio
        return np.asarray(array, dtype=np.float32)

    def _load_segment(self, feature: dict[str, Any]) -> np.ndarray:
        start = feature["start_time"]
        end = feature["end_time"]
        array, _ = librosa.load(
            feature["audio_path"],
            sr=self.feature_extractor.sampling_rate,
            offset=start,
            duration=end - start,
        )
        return array.astype(np.float32)

    def _time_perturb(self, audio: np.ndarray) -> np.ndarray:
        """Apply random speed perturbation by resampling."""
        rate = random.choice(SPEC_AUGMENT_RATES)
        if rate == 1.0:
            return audio

        orig_len = len(audio)
        new_len = int(orig_len / rate)
        if new_len < 1:
            return audio

        indices = np.linspace(0, orig_len - 1, new_len)
        return np.interp(indices, np.arange(orig_len), audio).astype(np.float32)

    def _tokenize_targets(self, features: list[dict[str, Any]]) -> torch.Tensor:
        """Tokenize single-seq targets.

        The tokenizer prepends <|startoftranscript|>, but the model also
        inserts it when shifting labels right into decoder inputs. Left in,
        training would see it twice while generation sees it once, so it is
        dropped here.
        """
        encoded = self.tokenizer(
            [self.target_text(f) for f in features],
            padding=True,
            truncation=True,
            max_length=448,
            return_tensors="pt",
        )
        labels = encoded.input_ids

        sot = self.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        if (labels[:, 0] == sot).all():
            labels = labels[:, 1:]
            attention = encoded.attention_mask[:, 1:]
        else:
            attention = encoded.attention_mask

        # Pad and EOS share an ID in Whisper; attention_mask marks the real
        # EOS as content, so only true padding gets masked.
        return labels.masked_fill(attention.ne(1), -100)
