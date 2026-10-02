"""Inference for trained Whisper single-seq paraphasia models.

Loads a checkpoint, decodes audio, and returns CHAI single-seq strings
("aphasia fekts [p] my language not my ditikalt [n]").
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from transformers import WhisperFeatureExtractor

from .tokenizer import normalize_output
from .whisper import WhisperParaphasiaConfig, build_model

SAMPLING_RATE = 16000
# Whisper's input window
MAX_CHUNK_SECONDS = 30.0
# Decoder loops ("stand stand stand ...") are cut to this many consecutive
# repeats. Real repetition in Scripts-Fridriksson tops out at 6 for a word
# and 3 for a phrase, so genuine stuttering/perseveration is untouched.
MAX_CONSECUTIVE_REPEATS = 8
MAX_LOOP_NGRAM = 4
# Output length cap per chunk: TOKENS_PER_SECOND * duration + TOKEN_SLACK.
# Scripts-Fridriksson references peak at 3.1 tokens/s (p99) and none exceed
# 4/s + 10; 6/s also covers fluent speech (~4 words/s). Bounds loops that
# vary slightly ("and it is meed and it is fool ...") and so evade
# collapse_loops.
TOKENS_PER_SECOND = 6.0
TOKEN_SLACK = 10


class ParaphasiaPredictor:
    """Run inference with a trained Whisper paraphasia model."""

    def __init__(
        self,
        model_path: str | Path,
        device: str | None = None,
        num_beams: int = 1,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model, self.tokenizer = build_model(
            WhisperParaphasiaConfig(model_name=str(model_path))
        )
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(str(model_path))
        self.model.to(self.device)
        if self.device.type == "cuda":
            self.model.half()
        self.model.eval()

        # No n-gram blocking: aphasic speech has real repetitions
        # ("and and then", "the broken the broken glass").
        self._gen_kwargs = {
            "language": "en",
            "task": "transcribe",
            "max_new_tokens": 440,
            "num_beams": num_beams,
        }

    def predict(self, audio: np.ndarray) -> str:
        """Transcribe one 16 kHz waveform to a single-seq string."""
        return self.predict_batch([audio])[0]

    def predict_batch(self, audios: list[np.ndarray]) -> list[str]:
        """Transcribe 16 kHz waveforms to single-seq strings.

        Audio longer than 30s is split into consecutive chunks whose
        transcripts are joined.
        """
        chunks: list[np.ndarray] = []
        owners: list[int] = []
        for i, audio in enumerate(audios):
            for chunk in _split(audio):
                chunks.append(chunk)
                owners.append(i)

        features = self.feature_extractor(
            chunks,
            sampling_rate=SAMPLING_RATE,
            return_tensors="pt",
            padding="max_length",
        ).input_features.to(device=self.device, dtype=self.model.dtype)
        attention_mask = torch.ones(features.shape[0], features.shape[-1], dtype=torch.long, device=self.device)

        caps = [_token_cap(len(c) / SAMPLING_RATE) for c in chunks]
        gen_kwargs = dict(self._gen_kwargs)
        gen_kwargs["max_new_tokens"] = min(gen_kwargs["max_new_tokens"], max(caps))

        with torch.no_grad():
            generated = self.model.generate(
                features, attention_mask=attention_mask, **gen_kwargs
            )

        # Trim each chunk to its own cap (the batch ran to the longest one's).
        # Whether generate() returns the <|startoftranscript|>... prefix varies
        # across transformers versions, so leading special tokens are skipped.
        special = set(self.tokenizer.all_special_ids)
        texts = []
        for ids, cap in zip(generated.tolist(), caps):
            start = 0
            while start < len(ids) and ids[start] in special:
                start += 1
            texts.append(self.tokenizer.decode(ids[start : start + cap], skip_special_tokens=True))
        joined = [[] for _ in audios]
        for owner, text in zip(owners, texts):
            joined[owner].append(text)
        return [
            " ".join(collapse_loops(normalize_output(" ".join(parts)).split()))
            for parts in joined
        ]

    def predict_file(self, audio_path: str | Path) -> str:
        """Transcribe an audio file."""
        import librosa

        audio, _ = librosa.load(str(audio_path), sr=SAMPLING_RATE, mono=True)
        return self.predict(audio)


def collapse_loops(
    tokens: list[str],
    max_repeats: int = MAX_CONSECUTIVE_REPEATS,
    max_n: int = MAX_LOOP_NGRAM,
) -> list[str]:
    """Truncate any n-gram (n ≤ max_n) repeated more than max_repeats times in a row."""
    out: list[str] = []
    i = 0
    while i < len(tokens):
        for n in range(1, max_n + 1):
            gram = tokens[i : i + n]
            reps = 1
            while tokens[i + reps * n : i + (reps + 1) * n] == gram:
                reps += 1
            if reps > max_repeats:
                out.extend(gram * max_repeats)
                i += reps * n
                break
        else:
            out.append(tokens[i])
            i += 1
    return out


def _token_cap(seconds: float) -> int:
    """Maximum output tokens for a chunk of this duration."""
    return int(TOKENS_PER_SECOND * seconds) + TOKEN_SLACK


def _split(audio: np.ndarray) -> list[np.ndarray]:
    """Split audio into ≤30s chunks of near-equal length."""
    max_len = int(MAX_CHUNK_SECONDS * SAMPLING_RATE)
    n = max(1, -(-len(audio) // max_len))
    return list(np.array_split(audio, n))
