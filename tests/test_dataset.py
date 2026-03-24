"""Tests for AphasiaBankDataset and LOSO splits."""

import json
import pytest

from aphasia_modeling.data.chat_parser import Utterance
from aphasia_modeling.data.dataset import AphasiaBankDataset


def _make_utterances(n_speakers: int = 3, utts_per_speaker: int = 10) -> list[Utterance]:
    """Create synthetic utterances for split testing."""
    utterances = []
    for s in range(n_speakers):
        spk = f"speaker_{s:02d}"
        for u in range(utts_per_speaker):
            utt = Utterance(
                utterance_id=f"{spk}_{u:03d}",
                speaker="PAR",
                raw_text=f"word_{s}_{u}",
                words=[f"word{u}"],
                labels=["c"],
                speaker_id=spk,
                session_id=spk,
            )
            utterances.append(utt)
    return utterances


class TestAphasiaBankDataset:
    def test_speaker_index(self):
        utts = _make_utterances(3, 5)
        ds = AphasiaBankDataset(utts)
        assert ds.num_speakers == 3
        assert len(ds.speakers) == 3

    def test_all_utterances_indexed(self):
        utts = _make_utterances(3, 5)
        ds = AphasiaBankDataset(utts)
        total = sum(len(v) for v in ds._speaker_index.values())
        assert total == 15


class TestLOSOSplit:
    def test_test_set_is_held_out_speaker(self):
        utts = _make_utterances(4, 10)
        ds = AphasiaBankDataset(utts)
        train, dev, test = ds.loso_split("speaker_00")
        # All test utterances belong to speaker_00
        assert all(u.speaker_id == "speaker_00" for u in test)
        assert len(test) == 10

    def test_no_speaker_leak(self):
        utts = _make_utterances(4, 10)
        ds = AphasiaBankDataset(utts)
        train, dev, test = ds.loso_split("speaker_01")
        train_dev_ids = {u.speaker_id for u in train} | {u.speaker_id for u in dev}
        test_ids = {u.speaker_id for u in test}
        assert train_dev_ids & test_ids == set()

    def test_all_utterances_accounted_for(self):
        utts = _make_utterances(4, 10)
        ds = AphasiaBankDataset(utts)
        train, dev, test = ds.loso_split("speaker_02")
        assert len(train) + len(dev) + len(test) == 40

    def test_dev_set_nonempty(self):
        utts = _make_utterances(4, 10)
        ds = AphasiaBankDataset(utts)
        train, dev, test = ds.loso_split("speaker_00")
        assert len(dev) > 0

    def test_deterministic_with_same_seed(self):
        utts = _make_utterances(4, 10)
        ds = AphasiaBankDataset(utts)
        t1, d1, _ = ds.loso_split("speaker_00", seed=42)
        t2, d2, _ = ds.loso_split("speaker_00", seed=42)
        assert [u.utterance_id for u in t1] == [u.utterance_id for u in t2]
        assert [u.utterance_id for u in d1] == [u.utterance_id for u in d2]

    def test_different_seed_gives_different_split(self):
        utts = _make_utterances(4, 20)
        ds = AphasiaBankDataset(utts)
        _, d1, _ = ds.loso_split("speaker_00", seed=42)
        _, d2, _ = ds.loso_split("speaker_00", seed=99)
        ids1 = {u.utterance_id for u in d1}
        ids2 = {u.utterance_id for u in d2}
        # Different seeds should pick different dev utterances
        assert ids1 != ids2


class TestLOSOFolds:
    def test_num_folds_equals_speakers(self):
        utts = _make_utterances(5, 8)
        ds = AphasiaBankDataset(utts)
        folds = ds.loso_folds()
        assert len(folds) == 5

    def test_each_speaker_is_test_once(self):
        utts = _make_utterances(4, 10)
        ds = AphasiaBankDataset(utts)
        folds = ds.loso_folds()
        test_speakers = [spk for spk, _, _, _ in folds]
        assert sorted(test_speakers) == sorted(ds.speakers)

    def test_fold_sizes_consistent(self):
        utts = _make_utterances(4, 10)
        ds = AphasiaBankDataset(utts)
        for spk, train, dev, test in ds.loso_folds():
            assert len(train) + len(dev) + len(test) == 40


class TestSaveLoad:
    def test_roundtrip(self, tmp_path):
        utts = _make_utterances(2, 3)
        ds = AphasiaBankDataset(utts)
        path = tmp_path / "test_dataset.json"
        ds.save(path)

        loaded = AphasiaBankDataset.load(path)
        assert len(loaded.utterances) == 6
        assert loaded.num_speakers == 2

    def test_fields_preserved(self, tmp_path):
        utt = Utterance(
            utterance_id="u1",
            speaker="PAR",
            raw_text="",
            words=["cat", "sat"],
            labels=["p", "c"],
            targets=["hat", None],
            audio_path="/audio/test.wav",
            start_time=1.5,
            end_time=3.0,
            session_id="sess1",
            database="fridriksson",
            speaker_id="spk1",
        )
        ds = AphasiaBankDataset([utt])
        path = tmp_path / "test.json"
        ds.save(path)

        loaded = AphasiaBankDataset.load(path)
        u = loaded.utterances[0]
        assert u.words == ["cat", "sat"]
        assert u.labels == ["p", "c"]
        assert u.targets == ["hat", None]
        assert u.audio_path == "/audio/test.wav"
        assert u.start_time == 1.5
        assert u.end_time == 3.0
        assert u.session_id == "sess1"
        assert u.database == "fridriksson"


class TestToDataframe:
    def test_dataframe_columns(self):
        utts = _make_utterances(2, 3)
        ds = AphasiaBankDataset(utts)
        df = ds.to_dataframe()
        assert len(df) == 6
        assert "utterance_id" in df.columns
        assert "single_seq" in df.columns
        assert "has_paraphasia" in df.columns

    def test_has_paraphasia_flag(self):
        utt_c = Utterance(
            utterance_id="u1", speaker="PAR", raw_text="",
            words=["cat"], labels=["c"], speaker_id="s1",
        )
        utt_p = Utterance(
            utterance_id="u2", speaker="PAR", raw_text="",
            words=["cat"], labels=["p"], speaker_id="s1",
        )
        ds = AphasiaBankDataset([utt_c, utt_p])
        df = ds.to_dataframe()
        assert df.iloc[0]["has_paraphasia"] == False
        assert df.iloc[1]["has_paraphasia"] == True
