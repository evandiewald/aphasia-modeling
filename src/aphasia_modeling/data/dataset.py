"""HuggingFace Dataset loader for preprocessed AphasiaBank data.

Supports CHAI's leave-one-speaker-out (LOSO) cross-validation on the
Fridriksson subset (12 folds, one per speaker).
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .chat_parser import Utterance, parse_cha_directory
from .preprocess import preprocess_dataset, to_single_seq

# CHAI uses seed 883 for their splits
CHAI_SPLIT_SEED = 883
CHAI_DEV_FRACTION = 0.1


class AphasiaBankDataset:
    """Manages preprocessed AphasiaBank utterances with split support."""

    def __init__(self, utterances: list[Utterance]):
        self.utterances = utterances
        self._speaker_index: dict[str, list[int]] = {}
        self._build_speaker_index()

    def _build_speaker_index(self) -> None:
        """Index utterances by speaker for cross-validation."""
        self._speaker_index.clear()
        for i, utt in enumerate(self.utterances):
            spk = utt.speaker_id
            if spk not in self._speaker_index:
                self._speaker_index[spk] = []
            self._speaker_index[spk].append(i)

    @property
    def speakers(self) -> list[str]:
        return sorted(self._speaker_index.keys())

    @property
    def num_speakers(self) -> int:
        return len(self._speaker_index)

    def loso_split(
        self,
        test_speaker: str,
        dev_fraction: float = CHAI_DEV_FRACTION,
        seed: int = CHAI_SPLIT_SEED,
    ) -> tuple[list[Utterance], list[Utterance], list[Utterance]]:
        """Leave-one-speaker-out split matching CHAI's partition_spk.py.

        CHAI's logic: one RNG seeded once, shuffles each speaker's utterances
        in order, then for each non-test speaker takes first N as dev and
        rest as train. The dev fraction is rounded, with a minimum of 1.

        Args:
            test_speaker: Speaker ID to hold out as test set.
            dev_fraction: Fraction of remaining utterances for dev set.
            seed: Random seed (CHAI uses 883).

        Returns:
            (train, dev, test) utterance lists.
        """
        rng = random.Random(seed)

        # Shuffle all speakers' utterances with a single RNG (matching CHAI)
        shuffled_by_spk: dict[str, list[Utterance]] = {}
        for spk in sorted(self._speaker_index.keys()):
            spk_utts = [self.utterances[i] for i in self._speaker_index[spk]]
            rng.shuffle(spk_utts)
            shuffled_by_spk[spk] = spk_utts

        test_utts = shuffled_by_spk[test_speaker]
        train_utts = []
        dev_utts = []

        for spk in sorted(self._speaker_index.keys()):
            if spk == test_speaker:
                continue
            spk_utts = shuffled_by_spk[spk]
            n_dev = max(1, round(dev_fraction * len(spk_utts)))
            dev_utts.extend(spk_utts[:n_dev])
            train_utts.extend(spk_utts[n_dev:])

        return train_utts, dev_utts, test_utts

    def loso_folds(
        self,
        dev_fraction: float = CHAI_DEV_FRACTION,
        seed: int = CHAI_SPLIT_SEED,
    ) -> list[tuple[str, list[Utterance], list[Utterance], list[Utterance]]]:
        """Generate all LOSO folds (one per speaker).

        Returns:
            List of (test_speaker, train, dev, test) tuples.
        """
        folds = []
        for spk in self.speakers:
            train, dev, test = self.loso_split(spk, dev_fraction, seed)
            folds.append((spk, train, dev, test))
        return folds

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to a pandas DataFrame for inspection."""
        records = []
        for utt in self.utterances:
            records.append({
                "utterance_id": utt.utterance_id,
                "speaker_id": utt.speaker_id,
                "session_id": utt.session_id,
                "database": utt.database,
                "words": " ".join(utt.words),
                "labels": " ".join(utt.labels),
                "single_seq": to_single_seq(utt.words, utt.labels),
                "audio_path": utt.audio_path,
                "start_time": utt.start_time,
                "end_time": utt.end_time,
                "num_words": len(utt.words),
                "has_paraphasia": any(l != "c" for l in utt.labels),
            })
        return pd.DataFrame(records)

    def to_hf_dataset(
        self,
        utterances: list[Utterance] | None = None,
        oversample_paraphasia: int = 1,
    ):
        """Convert utterances to a HuggingFace Dataset.

        Requires the `datasets` package. Audio loading is deferred
        (paths stored, loaded on access via Audio feature).

        Args:
            utterances: Utterances to include. Defaults to all.
            oversample_paraphasia: Repeat utterances containing paraphasias
                this many extra times (e.g., 3 = 4x total for paraphasia utts).
        """
        from datasets import Dataset, Features, Value

        utts = utterances or self.utterances

        # Oversample: repeat paraphasia-containing utterances
        if oversample_paraphasia > 1:
            expanded = []
            for utt in utts:
                expanded.append(utt)
                if any(l in ("p", "n") for l in utt.labels):
                    for _ in range(oversample_paraphasia - 1):
                        expanded.append(utt)
            utts = expanded

        records = []
        for utt in utts:
            records.append({
                "utterance_id": utt.utterance_id,
                "audio_path": utt.audio_path or "",
                "text": " ".join(utt.words),
                "single_seq": to_single_seq(utt.words, utt.labels),
                "labels": " ".join(utt.labels),
                "speaker_id": utt.speaker_id,
                "start_time": utt.start_time,
                "end_time": utt.end_time,
            })

        features = Features({
            "utterance_id": Value("string"),
            "audio_path": Value("string"),
            "text": Value("string"),
            "single_seq": Value("string"),
            "labels": Value("string"),
            "speaker_id": Value("string"),
            "start_time": Value("float64"),
            "end_time": Value("float64"),
        })

        return Dataset.from_list(records, features=features)

    def save(self, path: str | Path) -> None:
        """Save preprocessed dataset to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        records = []
        for utt in self.utterances:
            records.append({
                "utterance_id": utt.utterance_id,
                "speaker_id": utt.speaker_id,
                "session_id": utt.session_id,
                "database": utt.database,
                "words": utt.words,
                "labels": utt.labels,
                "targets": utt.targets,
                "audio_path": utt.audio_path,
                "start_time": utt.start_time,
                "end_time": utt.end_time,
            })
        path.write_text(json.dumps(records, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> AphasiaBankDataset:
        """Load preprocessed dataset from JSON."""
        path = Path(path)
        records = json.loads(path.read_text())
        utterances = []
        for r in records:
            utt = Utterance(
                utterance_id=r["utterance_id"],
                speaker="PAR",
                raw_text="",
                words=r["words"],
                labels=r["labels"],
                targets=r.get("targets", []),
                audio_path=r.get("audio_path", ""),
                start_time=r.get("start_time", 0.0),
                end_time=r.get("end_time", 0.0),
                session_id=r.get("session_id", ""),
                database=r.get("database", ""),
                speaker_id=r.get("speaker_id", ""),
            )
            utterances.append(utt)
        return cls(utterances)


def load_splits(
    cha_dir: str | Path,
    audio_dir: str | Path | None = None,
    cache_path: str | Path | None = None,
) -> AphasiaBankDataset:
    """Load and preprocess AphasiaBank data, with optional caching.

    Args:
        cha_dir: Directory containing .cha files.
        audio_dir: Directory containing .wav files.
        cache_path: If provided, save/load preprocessed data here.

    Returns:
        AphasiaBankDataset ready for split generation.
    """
    if cache_path:
        cache_path = Path(cache_path)
        if cache_path.exists():
            return AphasiaBankDataset.load(cache_path)

    utterances = parse_cha_directory(cha_dir, audio_dir)
    utterances = preprocess_dataset(utterances)

    dataset = AphasiaBankDataset(utterances)

    if cache_path:
        dataset.save(cache_path)

    return dataset
