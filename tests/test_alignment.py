"""Tests for sequence alignment used in evaluation."""

import pytest

from aphasia_modeling.evaluation.alignment import (
    EPS,
    align_sequences,
    levenshtein_alignment,
    reinsert_paraphasia_tags,
    strip_paraphasia_tags,
)


class TestStripParaphasiaTags:
    def test_no_tags(self):
        words, tags = strip_paraphasia_tags(["the", "cat", "sat"])
        assert words == ["the", "cat", "sat"]
        assert tags == {}

    def test_single_tag(self):
        words, tags = strip_paraphasia_tags(["the", "cat", "[p]", "sat"])
        assert words == ["the", "cat", "sat"]
        assert tags == {1: "p"}

    def test_multiple_tags(self):
        words, tags = strip_paraphasia_tags(
            ["the", "cat", "[p]", "sat", "on", "[s]", "mat"]
        )
        assert words == ["the", "cat", "sat", "on", "mat"]
        assert tags == {1: "p", 3: "s"}

    def test_all_three_types(self):
        words, tags = strip_paraphasia_tags(
            ["a", "[p]", "b", "[n]", "c", "[s]"]
        )
        assert words == ["a", "b", "c"]
        assert tags == {0: "p", 1: "n", 2: "s"}

    def test_c_tag(self):
        words, tags = strip_paraphasia_tags(["the", "[c]", "cat"])
        assert words == ["the", "cat"]
        assert tags == {0: "c"}

    def test_eps_filtered(self):
        words, tags = strip_paraphasia_tags(["the", EPS, "cat"])
        assert words == ["the", "cat"]

    def test_leading_tag_ignored(self):
        """Tag at the start with no preceding word is dropped."""
        words, tags = strip_paraphasia_tags(["[p]", "the", "cat"])
        assert words == ["the", "cat"]
        assert tags == {}

    def test_empty(self):
        words, tags = strip_paraphasia_tags([])
        assert words == []
        assert tags == {}


class TestLevenshteinAlignment:
    def test_identical(self):
        alignment = levenshtein_alignment(["a", "b", "c"], ["a", "b", "c"])
        assert alignment == [("a", "a"), ("b", "b"), ("c", "c")]

    def test_substitution(self):
        alignment = levenshtein_alignment(["a", "b"], ["a", "x"])
        assert alignment == [("a", "a"), ("b", "x")]

    def test_deletion(self):
        alignment = levenshtein_alignment(["a", "b", "c"], ["a", "c"])
        assert (EPS, "a") not in alignment  # 'a' matched
        # 'b' should be deleted from ref
        assert ("b", EPS) in alignment

    def test_insertion(self):
        alignment = levenshtein_alignment(["a", "c"], ["a", "b", "c"])
        assert (EPS, "b") in alignment

    def test_empty_ref(self):
        alignment = levenshtein_alignment([], ["a", "b"])
        assert alignment == [(EPS, "a"), (EPS, "b")]

    def test_empty_hyp(self):
        alignment = levenshtein_alignment(["a", "b"], [])
        assert alignment == [("a", EPS), ("b", EPS)]

    def test_both_empty(self):
        assert levenshtein_alignment([], []) == []

    def test_alignment_length(self):
        """Alignment length >= max(len(ref), len(hyp))."""
        ref = ["the", "cat", "sat"]
        hyp = ["a", "cat", "on", "mat"]
        alignment = levenshtein_alignment(ref, hyp)
        assert len(alignment) >= max(len(ref), len(hyp))

    def test_edit_distance_value(self):
        """Verify the number of non-matching pairs equals edit distance."""
        ref = ["the", "cat"]
        hyp = ["the", "dog"]
        alignment = levenshtein_alignment(ref, hyp)
        edits = sum(1 for r, h in alignment if r != h)
        assert edits == 1


class TestAlignSequences:
    def test_identical_no_tags(self):
        rw, rl, hw, hl = align_sequences(
            ["the", "cat"], ["the", "cat"]
        )
        assert rw == ["the", "cat"]
        assert hw == ["the", "cat"]
        assert rl == ["c", "c"]
        assert hl == ["c", "c"]

    def test_tags_preserved_through_alignment(self):
        ref = ["the", "cat", "[p]", "sat"]
        hyp = ["the", "cat", "[p]", "sat"]
        rw, rl, hw, hl = align_sequences(ref, hyp)
        assert rl == ["c", "p", "c"]
        assert hl == ["c", "p", "c"]

    def test_mismatched_tags(self):
        ref = ["the", "cat", "[p]", "sat"]
        hyp = ["the", "cat", "sat"]  # Missing [p]
        rw, rl, hw, hl = align_sequences(ref, hyp)
        # ref has p on "cat", hyp doesn't
        cat_idx = rw.index("cat")
        assert rl[cat_idx] == "p"
        assert hl[cat_idx] == "c"

    def test_insertion_gets_c_label(self):
        ref = ["the", "cat"]
        hyp = ["the", "big", "cat"]
        rw, rl, hw, hl = align_sequences(ref, hyp)
        eps_idx = rw.index(EPS)
        assert rl[eps_idx] == "c"

    def test_deletion_gets_c_label(self):
        ref = ["the", "big", "cat"]
        hyp = ["the", "cat"]
        rw, rl, hw, hl = align_sequences(ref, hyp)
        eps_idx = hw.index(EPS)
        assert hl[eps_idx] == "c"


class TestReinsertParaphasiaTags:
    def test_basic(self):
        result = reinsert_paraphasia_tags(
            ["the", "cat", "sat"], ["c", "p", "c"]
        )
        assert result == ["the", "cat", "[p]", "sat"]

    def test_no_tags(self):
        result = reinsert_paraphasia_tags(["the", "cat"], ["c", "c"])
        assert result == ["the", "cat"]

    def test_all_paraphasic(self):
        result = reinsert_paraphasia_tags(["a", "b"], ["p", "n"])
        assert result == ["a", "[p]", "b", "[n]"]

    def test_roundtrip_with_strip(self):
        original = ["the", "cat", "[p]", "sat", "on", "[s]", "mat"]
        words, tag_map = strip_paraphasia_tags(original)
        labels = [tag_map.get(i, "c") for i in range(len(words))]
        reconstructed = reinsert_paraphasia_tags(words, labels)
        assert reconstructed == original
