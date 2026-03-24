"""Tests for evaluation metrics (WER, AWER, TD, F1)."""

import pytest

from aphasia_modeling.evaluation.metrics import (
    MetricResult,
    compute_all_metrics,
    compute_awer,
    compute_awer_pd,
    compute_td_binary,
    compute_td_multiclass,
    compute_utterance_f1,
    compute_wer,
)


# ---- WER ---------------------------------------------------------------------


class TestWER:
    def test_perfect_transcription(self):
        refs = [["the", "cat", "sat"]]
        hyps = [["the", "cat", "sat"]]
        assert compute_wer(refs, hyps) == 0.0

    def test_one_substitution(self):
        refs = [["the", "cat", "sat"]]
        hyps = [["the", "dog", "sat"]]
        assert compute_wer(refs, hyps) == pytest.approx(1 / 3)

    def test_paraphasia_tags_stripped(self):
        """WER should ignore paraphasia tags."""
        refs = [["the", "cat", "[p]", "sat"]]
        hyps = [["the", "cat", "[p]", "sat"]]
        assert compute_wer(refs, hyps) == 0.0

    def test_tags_stripped_mismatch(self):
        """Differing tags shouldn't affect WER if words match."""
        refs = [["the", "cat", "[p]", "sat"]]
        hyps = [["the", "cat", "sat"]]  # no tag
        assert compute_wer(refs, hyps) == 0.0

    def test_multiple_utterances(self):
        refs = [["a", "b"], ["c", "d"]]
        hyps = [["a", "b"], ["c", "x"]]
        wer = compute_wer(refs, hyps)
        # 1 error out of 4 words
        assert wer == pytest.approx(0.25)


# ---- AWER --------------------------------------------------------------------


class TestAWER:
    def test_perfect_with_tags(self):
        refs = [["the", "cat", "[p]", "sat"]]
        hyps = [["the", "cat", "[p]", "sat"]]
        assert compute_awer(refs, hyps) == 0.0

    def test_wrong_tag_counts_as_error(self):
        """Correct word but wrong tag -> the compound token differs."""
        refs = [["cat", "[p]"]]
        hyps = [["cat", "[s]"]]
        # ref: "cat/p", hyp: "cat/s" -> 1 sub, AWER = 1.0
        assert compute_awer(refs, hyps) == pytest.approx(1.0)

    def test_missing_tag_counts_as_error(self):
        refs = [["cat", "[p]"]]
        hyps = [["cat"]]
        # ref: "cat/p", hyp: "cat/c" -> 1 sub
        assert compute_awer(refs, hyps) == pytest.approx(1.0)

    def test_correct_words_no_tags(self):
        refs = [["the", "cat"]]
        hyps = [["the", "cat"]]
        # Both "the/c cat/c"
        assert compute_awer(refs, hyps) == 0.0


# ---- AWER-PD -----------------------------------------------------------------


class TestAWER_PD:
    def test_p_and_n_reduce_to_label_only(self):
        """For p/n, the word is dropped and only label is compared."""
        refs = [["blick", "[p]"]]
        hyps = [["gorp", "[p]"]]
        # ref: "p", hyp: "p" (words dropped for p/n)
        assert compute_awer_pd(refs, hyps) == 0.0

    def test_semantic_keeps_word(self):
        """For s, the word matters."""
        refs = [["table", "[s]"]]
        hyps = [["chair", "[s]"]]
        # ref: "table/s", hyp: "chair/s" -> different compound
        assert compute_awer_pd(refs, hyps) == pytest.approx(1.0)

    def test_correct_words_keep_word(self):
        refs = [["the", "cat"]]
        hyps = [["the", "dog"]]
        # "the/c cat/c" vs "the/c dog/c"
        assert compute_awer_pd(refs, hyps) == pytest.approx(0.5)


# ---- TD-binary ---------------------------------------------------------------


class TestTDBinary:
    def test_perfect_detection(self):
        """Paraphasias at same positions -> TD = 0."""
        refs = [["the", "cat", "[p]", "sat", "on", "mat", "[s]"]]
        hyps = [["the", "cat", "[p]", "sat", "on", "mat", "[s]"]]
        td = compute_td_binary(refs, hyps)
        assert td == pytest.approx(0.0)

    def test_no_paraphasias(self):
        """No paraphasias in ref or hyp -> 0 (no utterances contribute)."""
        refs = [["the", "cat"]]
        hyps = [["the", "cat"]]
        td = compute_td_binary(refs, hyps)
        assert td == 0.0

    def test_shifted_by_one(self):
        """Paraphasia shifted by one position."""
        # 5 words total after alignment
        refs = [["a", "b", "[p]", "c", "d", "e"]]  # paraphasia at pos 1
        hyps = [["a", "b", "c", "[p]", "d", "e"]]  # paraphasia at pos 2
        td = compute_td_binary(refs, hyps)
        # TTC: |1-2| = 1, CTT: |2-1| = 1, seq_len = 5 -> (1+1)/5 = 0.4
        assert td == pytest.approx(0.4)

    def test_missed_paraphasia(self):
        """Ref has paraphasia, hyp doesn't."""
        refs = [["a", "b", "[p]", "c"]]
        hyps = [["a", "b", "c"]]
        td = compute_td_binary(refs, hyps)
        assert td > 0


# ---- TD-multiclass -----------------------------------------------------------


class TestTDMulticlass:
    def test_per_class_separation(self):
        refs = [["a", "[p]", "b", "[n]", "c"]]
        hyps = [["a", "[p]", "b", "[n]", "c"]]
        td = compute_td_multiclass(refs, hyps)
        assert td["p"] == pytest.approx(0.0)
        assert td["n"] == pytest.approx(0.0)
        assert td["s"] == 0.0  # No s paraphasias
        assert td["all"] == pytest.approx(0.0)

    def test_wrong_class_not_matched(self):
        """p in ref, n in hyp at same position -> both miss."""
        refs = [["a", "[p]", "b"]]
        hyps = [["a", "[n]", "b"]]
        td = compute_td_multiclass(refs, hyps)
        # p: ref has one at pos 0, hyp has none -> penalty
        assert td["p"] > 0
        # n: hyp has one at pos 0, ref has none -> penalty
        assert td["n"] > 0

    def test_td_all_is_sum(self):
        refs = [["a", "[p]", "b", "[n]", "c", "[s]"]]
        hyps = [["a", "[p]", "b", "[n]", "c", "[s]"]]
        td = compute_td_multiclass(refs, hyps)
        assert td["all"] == pytest.approx(td["p"] + td["n"] + td["s"])


# ---- Utterance-level F1 ------------------------------------------------------


class TestUtteranceF1:
    def test_perfect_detection(self):
        refs = [
            ["a", "[p]", "b"],
            ["c", "d"],
        ]
        hyps = [
            ["a", "[p]", "b"],
            ["c", "d"],
        ]
        f1 = compute_utterance_f1(refs, hyps)
        assert f1["p"]["f1"] == pytest.approx(1.0)

    def test_missed_detection(self):
        refs = [["a", "[p]", "b"]]
        hyps = [["a", "b"]]  # No [p]
        f1 = compute_utterance_f1(refs, hyps)
        assert f1["p"]["recall"] == pytest.approx(0.0)

    def test_false_positive(self):
        refs = [["a", "b"]]
        hyps = [["a", "[p]", "b"]]
        f1 = compute_utterance_f1(refs, hyps)
        assert f1["p"]["precision"] == pytest.approx(0.0)

    def test_no_instances_returns_zero(self):
        refs = [["a", "b"]]
        hyps = [["a", "b"]]
        f1 = compute_utterance_f1(refs, hyps)
        assert f1["p"]["f1"] == 0.0
        assert f1["n"]["f1"] == 0.0
        assert f1["s"]["f1"] == 0.0

    def test_multi_utterance_f1(self):
        refs = [
            ["a", "[p]"],  # has p
            ["b"],          # no p
            ["c", "[p]"],  # has p
        ]
        hyps = [
            ["a", "[p]"],  # TP
            ["b", "[p]"],  # FP
            ["c"],          # FN
        ]
        f1 = compute_utterance_f1(refs, hyps)
        # precision = 1/2, recall = 1/2, f1 = 2*(0.5*0.5)/(0.5+0.5) = 0.5
        assert f1["p"]["f1"] == pytest.approx(0.5)
        assert f1["p"]["precision"] == pytest.approx(0.5)
        assert f1["p"]["recall"] == pytest.approx(0.5)

    def test_each_class_independent(self):
        refs = [["a", "[p]", "b", "[s]"]]
        hyps = [["a", "[p]", "b"]]  # Got p, missed s
        f1 = compute_utterance_f1(refs, hyps)
        assert f1["p"]["f1"] == pytest.approx(1.0)
        assert f1["s"]["recall"] == pytest.approx(0.0)


# ---- compute_all_metrics integration -----------------------------------------


class TestComputeAllMetrics:
    def test_perfect_score(self):
        refs = [["the", "cat", "[p]", "sat"]]
        hyps = [["the", "cat", "[p]", "sat"]]
        result = compute_all_metrics(refs, hyps)
        assert result.wer == 0.0
        assert result.awer == 0.0
        assert result.td_p == pytest.approx(0.0)
        assert result.f1_p == pytest.approx(1.0)
        assert result.num_utterances == 1

    def test_returns_metric_result(self):
        refs = [["a", "b"]]
        hyps = [["a", "b"]]
        result = compute_all_metrics(refs, hyps)
        assert isinstance(result, MetricResult)

    def test_str_representation(self):
        result = MetricResult(wer=0.5, num_utterances=10)
        s = str(result)
        assert "WER:" in s
        assert "0.5000" in s
        assert "10" in s
