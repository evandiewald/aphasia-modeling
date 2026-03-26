"""Data collator for Whisper paraphasia fine-tuning.

Handles:
- Audio feature extraction via WhisperFeatureExtractor
- Target sequence tokenization (plain ASR text — no paraphasia tokens)
- Classification labels aligned to token positions (for paraphasia head)
- Padding and label masking (-100 for pad tokens in labels)
- SpecAugment time perturbation (speed changes per CHAI rates)
"""

from __future__ import annotations

import random
import warnings
from dataclasses import dataclass, field
from typing import Any

warnings.filterwarnings("ignore", message=".*audioread.*", category=FutureWarning)

import librosa
import numpy as np
import torch
from transformers import WhisperFeatureExtractor, WhisperTokenizerFast

from .classifier import CLS_CORRECT, CLS_PHONEMIC, CLS_NEOLOGISTIC

# CHAI SpecAugment time perturbation rates
SPEC_AUGMENT_RATES = [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]

# Map from string labels to classification indices
_LABEL_TO_CLS = {"c": CLS_CORRECT, "p": CLS_PHONEMIC, "n": CLS_NEOLOGISTIC}


@dataclass
class ParaphasiaDataCollator:
    """Collator for Whisper paraphasia training.

    Expects each example to have:
    - "audio_path": str path to WAV file (with start_time/end_time)
      OR "audio": dict with "array" and "sampling_rate"
    - "text": str plain text (ASR target, no paraphasia tokens)
    - "labels": str space-separated paraphasia labels ("c p c c n")

    Returns dict with:
    - "input_features": mel spectrogram tensor
    - "labels": token IDs for ASR loss
    - "cls_labels": per-token classification targets for the paraphasia head
    """

    feature_extractor: WhisperFeatureExtractor
    tokenizer: WhisperTokenizerFast
    apply_time_perturbation: bool = False
    use_classifier: bool = False

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # --- Audio features ---
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

        # --- ASR labels (plain text, no paraphasia tokens) ---
        labels = self._tokenize_targets(features)

        result = {
            "input_features": input_features,
            "labels": labels,
        }

        # --- Classification labels (if using classifier head) ---
        if self.use_classifier:
            cls_labels = self._build_cls_labels(features, labels)
            result["cls_labels"] = cls_labels

        return result

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

            if self.apply_time_perturbation:
                array = self._time_perturb(array)

            audio_arrays.append(array)

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
        """Apply random time perturbation (speed change)."""
        rate = random.choice(SPEC_AUGMENT_RATES)
        if rate == 1.0:
            return audio

        orig_len = len(audio)
        new_len = int(orig_len / rate)
        if new_len < 1:
            return audio

        indices = np.linspace(0, orig_len - 1, new_len)
        return np.interp(indices, np.arange(orig_len), audio).astype(np.float32)

    def _tokenize_targets(
        self, features: list[dict[str, Any]]
    ) -> torch.Tensor:
        """Tokenize plain text targets (no paraphasia tokens)."""
        texts = [f.get("text", "") for f in features]

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=448,
            return_tensors="pt",
        )

        labels = encoded.input_ids

        # Mask padding with -100 (keep first EOS)
        pad_id = self.tokenizer.pad_token_id
        for i in range(labels.size(0)):
            token_ids = labels[i].tolist()
            try:
                first_eos = token_ids.index(pad_id)
                labels[i, first_eos + 1:] = -100
            except ValueError:
                pass

        return labels

    def _build_cls_labels(
        self,
        features: list[dict[str, Any]],
        asr_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Build per-token classification labels aligned to ASR token positions.

        Each word in the text has a paraphasia label (c/p/n). We need to map
        these word-level labels to subword token positions, since BPE may split
        a word into multiple tokens.

        Strategy: tokenize each word individually to find how many tokens it
        produces, then assign that word's paraphasia label to all its tokens.
        Special tokens (prefix, EOS, padding) get -100.
        """
        batch_size, seq_len = asr_labels.shape
        cls_labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)

        for i, f in enumerate(features):
            word_labels = f.get("labels", "").split()
            text = f.get("text", "")
            words = text.split()

            if not words or len(words) != len(word_labels):
                continue

            # Find how many prefix tokens the tokenizer adds
            # Whisper prefix: <|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|>
            prefix_len = len(self.tokenizer.prefix_tokens)

            # Tokenize each word to get its token count
            token_pos = prefix_len  # start after prefix
            for word, label in zip(words, word_labels):
                word_tokens = self.tokenizer.encode(
                    word, add_special_tokens=False
                )
                n_tokens = len(word_tokens)
                cls_id = _LABEL_TO_CLS.get(label, CLS_CORRECT)

                for t in range(n_tokens):
                    pos = token_pos + t
                    if pos < seq_len:
                        cls_labels[i, pos] = cls_id

                token_pos += n_tokens

        return cls_labels
