"""Data collator for Whisper paraphasia fine-tuning.

Handles:
- Audio feature extraction via WhisperFeatureExtractor
- Target sequence tokenization (text with inline paraphasia tokens)
- Padding and label masking (-100 for pad tokens in labels)
- SpecAugment time perturbation (speed changes per CHAI rates)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import librosa
import numpy as np
import torch
from transformers import WhisperFeatureExtractor, WhisperTokenizerFast

# CHAI SpecAugment time perturbation rates
SPEC_AUGMENT_RATES = [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]


@dataclass
class ParaphasiaDataCollator:
    """Collator for Whisper paraphasia training.

    Expects each example to have:
    - "audio": dict with "array" (numpy float32) and "sampling_rate" (int)
    - "single_seq": str in single-seq format ("word [p] word word [n]")

    Or for preprocessed features:
    - "input_features": pre-extracted mel spectrogram
    - "labels": pre-tokenized label IDs
    """

    feature_extractor: WhisperFeatureExtractor
    tokenizer: WhisperTokenizerFast
    apply_time_perturbation: bool = False

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # Determine audio source: raw audio dict, audio_path string, or pre-extracted features
        has_audio_path = "audio_path" in features[0] and features[0]["audio_path"]
        has_raw_audio = "audio" in features[0] and features[0]["audio"] is not None

        if has_audio_path:
            input_features = self._load_and_extract(features)
        elif has_raw_audio:
            input_features = self._extract_audio_features(features)
        else:
            input_features = torch.tensor(
                np.stack([f["input_features"] for f in features]),
                dtype=torch.float32,
            )

        # Tokenize target sequences
        if "labels" in features[0] and isinstance(features[0]["labels"], list):
            # Pre-tokenized
            label_ids = features[0]["labels"]
            labels = self._pad_labels([f["labels"] for f in features])
        else:
            labels = self._tokenize_targets(features)

        return {
            "input_features": input_features,
            "labels": labels,
        }

    def _extract_audio_features(
        self, features: list[dict[str, Any]]
    ) -> torch.Tensor:
        """Extract mel spectrogram features from raw audio."""
        audio_arrays = []
        for f in features:
            audio = f["audio"]
            array = audio["array"] if isinstance(audio, dict) else audio
            if isinstance(array, np.ndarray):
                array = array.astype(np.float32)

            # Apply time perturbation (speed augmentation)
            if self.apply_time_perturbation:
                array = self._time_perturb(array)

            audio_arrays.append(array)

        # Extract features
        batch = self.feature_extractor(
            audio_arrays,
            sampling_rate=self.feature_extractor.sampling_rate,
            return_tensors="pt",
            padding="max_length",
        )
        return batch.input_features

    def _load_and_extract(
        self, features: list[dict[str, Any]]
    ) -> torch.Tensor:
        """Load audio from file paths and extract features."""
        sr = self.feature_extractor.sampling_rate
        audio_arrays = []
        for f in features:
            path = f["audio_path"]
            start = f.get("start_time", 0.0)
            end = f.get("end_time", 0.0)

            # Load with optional segment slicing
            if start > 0 or end > 0:
                duration = (end - start) if end > start else None
                array, _ = librosa.load(path, sr=sr, offset=start, duration=duration)
            else:
                array, _ = librosa.load(path, sr=sr)

            array = array.astype(np.float32)

            if self.apply_time_perturbation:
                array = self._time_perturb(array)

            audio_arrays.append(array)

        batch = self.feature_extractor(
            audio_arrays,
            sampling_rate=sr,
            return_tensors="pt",
            padding="max_length",
        )
        return batch.input_features

    def _time_perturb(self, audio: np.ndarray) -> np.ndarray:
        """Apply random time perturbation (speed change).

        Resamples audio by a random rate from CHAI's set:
        [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]

        Rate > 1.0 = faster speech (shorter), rate < 1.0 = slower (longer).
        """
        rate = random.choice(SPEC_AUGMENT_RATES)
        if rate == 1.0:
            return audio

        # Simple linear interpolation for speed change
        orig_len = len(audio)
        new_len = int(orig_len / rate)
        if new_len < 1:
            return audio

        indices = np.linspace(0, orig_len - 1, new_len)
        return np.interp(indices, np.arange(orig_len), audio).astype(np.float32)

    def _tokenize_targets(
        self, features: list[dict[str, Any]]
    ) -> torch.Tensor:
        """Tokenize single-seq text targets."""
        texts = [f.get("single_seq", f.get("text", "")) for f in features]

        # Tokenize with the extended tokenizer
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=448,  # Whisper's max target length
            return_tensors="pt",
        )

        labels = encoded.input_ids

        # Replace pad tokens with -100 so they're ignored in loss.
        # IMPORTANT: pad and eos share the same token ID in Whisper (50257).
        # We must keep the FIRST occurrence (the real EOS) and only mask
        # the trailing padding. For each sequence, find where padding starts
        # (first pad token AFTER the last non-pad token).
        pad_id = self.tokenizer.pad_token_id
        for i in range(labels.size(0)):
            # Find the first pad token after content ends
            token_ids = labels[i].tolist()
            # The EOS is the first occurrence of pad_id; padding starts after
            try:
                first_eos = token_ids.index(pad_id)
                # Keep the EOS, mask everything after it
                labels[i, first_eos + 1:] = -100
            except ValueError:
                # No pad/eos token found — nothing to mask
                pass

        return labels

    def _pad_labels(self, label_lists: list[list[int]]) -> torch.Tensor:
        """Pad pre-tokenized label sequences."""
        max_len = max(len(l) for l in label_lists)
        padded = []
        for labels in label_lists:
            pad_len = max_len - len(labels)
            padded.append(labels + [-100] * pad_len)
        return torch.tensor(padded, dtype=torch.long)
