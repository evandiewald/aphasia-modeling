"""Tests for the CHAT utterance preprocessing pipeline."""

import pytest

from aphasia_modeling.data.chat_parser import Utterance
from aphasia_modeling.data.preprocess import (
    ipa_to_pseudoword,
    parse_single_seq,
    preprocess_utterance,
    to_single_seq,
)


def _make_utt(raw_text: str) -> Utterance:
    return Utterance(utterance_id="test", speaker="PAR", raw_text=raw_text)


def _preprocess(raw_text: str) -> Utterance:
    utt = _make_utt(raw_text)
    valid = preprocess_utterance(utt)
    assert valid, f"Expected valid utterance for: {raw_text!r}"
    return utt


# ---- Skip conditions --------------------------------------------------------


class TestSkipConditions:
    def test_empty(self):
        assert not preprocess_utterance(_make_utt(""))

    def test_whitespace_only(self):
        assert not preprocess_utterance(_make_utt("   "))

    def test_xxx_unintelligible(self):
        assert not preprocess_utterance(_make_utt("xxx"))
        assert not preprocess_utterance(_make_utt("xxx."))

    def test_yyy_phonological(self):
        assert not preprocess_utterance(_make_utt("yyy"))

    def test_www_untranscribed(self):
        assert not preprocess_utterance(_make_utt("www"))

    def test_zero_action_only(self):
        assert not preprocess_utterance(_make_utt("0"))

    def test_unclear_marker(self):
        assert not preprocess_utterance(_make_utt("I want [?] the thing"))

    def test_xxx_in_sentence(self):
        assert not preprocess_utterance(_make_utt("I want xxx to go"))

    def test_overlap_marker(self):
        assert not preprocess_utterance(_make_utt("I want [<] the thing"))
        assert not preprocess_utterance(_make_utt("I want [>] the thing"))

    def test_paralinguistic(self):
        assert not preprocess_utterance(_make_utt("I [=! sighs] want"))

    def test_exc_postcode(self):
        assert not preprocess_utterance(_make_utt("hello [+ exc]"))

    def test_valid_not_skipped(self):
        assert preprocess_utterance(_make_utt("the cat sat"))


# ---- Paraphasia error codes -------------------------------------------------


class TestErrorCodes:
    def test_phonemic(self):
        utt = _preprocess("the cot [* p] sat")
        assert utt.labels[1] == "p"

    def test_neologistic(self):
        utt = _preprocess("the blicket [* n] sat")
        assert utt.labels[1] == "n"

    def test_semantic_treated_as_correct(self):
        """Semantic paraphasias are handled by Stage 2 LLM, not acoustic model."""
        utt = _preprocess("the table [* s] sat")
        assert utt.labels[1] == "c"

    def test_subtype_p_colon_k(self):
        """[* p:k] should map to 'p'."""
        utt = _preprocess("the cot [* p:k] sat")
        assert utt.labels[1] == "p"

    def test_subtype_n_colon_k(self):
        """[* n:k] should map to 'n'."""
        utt = _preprocess("the blick [* n:k] sat")
        assert utt.labels[1] == "n"

    def test_other_error_codes_treated_as_correct(self):
        """[* d] (dysfluency) and similar -> 'c'."""
        utt = _preprocess("the the [* d] cat sat")
        assert utt.labels[1] == "c"

    def test_multiple_errors(self):
        utt = _preprocess("the cot [* p] sat on mat [* n]")
        assert utt.words == ["the", "cot", "sat", "on", "mat"]
        assert utt.labels == ["c", "p", "c", "c", "n"]

    def test_correct_words_default_to_c(self):
        utt = _preprocess("the cat sat")
        assert utt.labels == ["c", "c", "c"]

    def test_words_match_with_errors(self):
        utt = _preprocess("the cot [* p] sat")
        assert utt.words == ["the", "cot", "sat"]


# ---- Target word annotations ------------------------------------------------


class TestTargetAnnotations:
    def test_target_attached_to_preceding_word(self):
        utt = _preprocess("the garden [: house] [* p] is nice")
        # "garden" should have target "house"
        idx = utt.words.index("garden")
        assert utt.targets[idx] == "house"

    def test_target_without_error(self):
        utt = _preprocess("the dog [: cat] is nice")
        idx = utt.words.index("dog")
        assert utt.targets[idx] == "cat"


# ---- Punctuation and terminators ---------------------------------------------


class TestTerminators:
    def test_period_removed(self):
        utt = _preprocess("the cat sat .")
        assert "." not in utt.words

    def test_comma_removed(self):
        utt = _preprocess("the cat , sat")
        assert "," not in " ".join(utt.words)

    def test_trailing_ellipsis(self):
        utt = _preprocess("the cat sat +...")
        assert "+..." not in utt.words

    def test_retrace_markers_removed(self):
        utt = _preprocess("the the [/] cat sat")
        assert "[/]" not in utt.words
        assert "[//]" not in utt.words

    def test_pauses_removed(self):
        utt = _preprocess("the (..) cat sat")
        assert "(..)" not in utt.words


# ---- Vocal events and fragments ----------------------------------------------


class TestVocalEventsAndFragments:
    def test_vocal_event_removed(self):
        utt = _preprocess("the &=laughs cat sat")
        assert all("laughs" not in w for w in utt.words)

    def test_cough_removed(self):
        utt = _preprocess("the &=coughs cat sat")
        assert len(utt.words) == 3

    def test_fragment_removed(self):
        utt = _preprocess("the &-uh cat sat")
        assert len(utt.words) == 3

    def test_false_start_removed(self):
        utt = _preprocess("the &+well cat sat")
        assert len(utt.words) == 3


# ---- Compound words ----------------------------------------------------------


class TestCompoundWords:
    def test_plus_compound_split(self):
        utt = _preprocess("ice+cream is good")
        assert "ice" in utt.words
        assert "cream" in utt.words

    def test_underscore_compound_split(self):
        utt = _preprocess("new_york is big")
        assert "new" in utt.words
        assert "york" in utt.words


# ---- Special forms -----------------------------------------------------------


class TestSpecialForms:
    def test_interjection_skipped(self):
        utt = _preprocess("um@i the cat sat")
        assert "um" not in utt.words

    def test_at_n_neologism_skipped(self):
        """@n words without replacement are skipped."""
        assert not preprocess_utterance(_make_utt("blick@n"))

    def test_at_with_replacement_uses_target(self):
        utt = _preprocess("goed@n [: went] away")
        assert "went" in utt.words
        assert "goed" not in utt.words


# ---- Finalization (lowercase, cleanup) ---------------------------------------


class TestFinalization:
    def test_lowercase(self):
        utt = _preprocess("The CAT Sat")
        assert utt.words == ["the", "cat", "sat"]

    def test_possessive_simplified(self):
        utt = _preprocess("the cat's toy")
        assert "cats" in utt.words

    def test_partial_omission_expanded(self):
        utt = _preprocess("walk(ing) fast")
        assert "walking" in utt.words

    def test_elongation_colon_removed(self):
        utt = _preprocess("no::: way")
        assert utt.words[0] == "no"

    def test_hyphenated_word_split(self):
        utt = _preprocess("self-control")
        assert "self" in utt.words
        assert "control" in utt.words

    def test_nonalpha_stripped(self):
        utt = _preprocess("cat123 dog")
        assert utt.words[0] == "cat"


# ---- Angular brackets -------------------------------------------------------


class TestAngularBrackets:
    def test_scope_markers_removed(self):
        utt = _preprocess("< the cat > sat")
        assert "<" not in utt.words
        assert ">" not in utt.words
        assert "the" in utt.words


# ---- Repetitions -------------------------------------------------------------


class TestRepetitions:
    def test_repetition_expanded(self):
        utt = _preprocess("no [x 3] I said")
        # "no" should appear 3 times
        assert utt.words.count("no") == 3


# ---- Single-seq format -------------------------------------------------------


class TestSingleSeq:
    def test_basic_conversion(self):
        result = to_single_seq(["the", "cat", "sat"], ["c", "p", "c"])
        assert result == "the cat [p] sat"

    def test_all_correct(self):
        result = to_single_seq(["the", "cat"], ["c", "c"])
        assert result == "the cat"

    def test_multiple_labels(self):
        result = to_single_seq(
            ["the", "cat", "sat", "on", "mat"],
            ["c", "p", "c", "c", "n"],
        )
        assert result == "the cat [p] sat on mat [n]"

    def test_neologistic(self):
        result = to_single_seq(["blick", "sat"], ["n", "c"])
        assert result == "blick [n] sat"

    def test_roundtrip(self):
        words = ["the", "cat", "sat", "on", "mat"]
        labels = ["c", "p", "c", "c", "n"]
        seq = to_single_seq(words, labels)
        parsed_words, parsed_labels = parse_single_seq(seq)
        assert parsed_words == words
        assert parsed_labels == labels

    def test_parse_all_correct(self):
        words, labels = parse_single_seq("the cat sat")
        assert words == ["the", "cat", "sat"]
        assert labels == ["c", "c", "c"]

    def test_parse_leading_tag_ignored(self):
        """A tag with no preceding word is ignored."""
        words, labels = parse_single_seq("[p] the cat")
        assert words == ["the", "cat"]
        assert labels == ["c", "c"]

    def test_roundtrip_all_paraphasic(self):
        words = ["blick", "gorp"]
        labels = ["n", "p"]
        seq = to_single_seq(words, labels)
        pw, pl = parse_single_seq(seq)
        assert pw == words
        assert pl == labels


# ---- IPA to pseudoword ------------------------------------------------------


class TestIpaToPseudoword:
    def test_simple_consonants(self):
        assert ipa_to_pseudoword("pat") == "pat"

    def test_chat_unibet_vowels(self):
        result = ipa_to_pseudoword("fEkts")
        assert result == "fekts"

    def test_digraph_sh(self):
        result = ipa_to_pseudoword("ʃip")
        assert result == "sheep"  # ʃ->sh, i->ee, p->p

    def test_unknown_alpha_passthrough(self):
        result = ipa_to_pseudoword("xyz")
        # x is not in map -> lowercase passthrough
        assert "x" in result

    def test_empty(self):
        assert ipa_to_pseudoword("") == ""


# ---- End-to-end preprocessing ------------------------------------------------


class TestEndToEnd:
    def test_chai_example_format(self):
        """Test that we produce CHAI-style output: 'aphasia fekts [p] my language'."""
        utt = _preprocess("aphasia fekts [* p] my language")
        seq = to_single_seq(utt.words, utt.labels)
        assert seq == "aphasia fekts [p] my language"

    def test_multiple_paraphasia_types(self):
        utt = _preprocess("the dog [* s] ran to blick [* n] and cot [* p]")
        # [* s] -> "c" (semantic handled by Stage 2 LLM)
        assert utt.labels == ["c", "c", "c", "c", "n", "c", "p"]
        seq = to_single_seq(utt.words, utt.labels)
        assert seq == "the dog ran to blick [n] and cot [p]"

    def test_clean_utterance_all_correct(self):
        utt = _preprocess("the cat sat on the mat")
        assert all(l == "c" for l in utt.labels)
        assert to_single_seq(utt.words, utt.labels) == "the cat sat on the mat"
