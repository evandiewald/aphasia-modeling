"""Tests for model components: tokenizer, model setup, collator, inference.

All tests use openai/whisper-tiny to minimize download size and run on CPU.
"""

import numpy as np
import pytest
import torch
from transformers import WhisperFeatureExtractor

from aphasia_modeling.model.collator import ParaphasiaDataCollator, SPEC_AUGMENT_RATES
from aphasia_modeling.model.inference import ParaphasiaPredictor, _split, _token_cap, collapse_loops
from aphasia_modeling.model.tokenizer import (
    PARAPHASIA_TOKENS,
    build_tokenizer,
    get_paraphasia_token_ids,
    normalize_output,
)
from aphasia_modeling.model.whisper import (
    WhisperParaphasiaConfig,
    build_model,
    freeze_encoder,
    get_class_weight_tensor,
    unfreeze_encoder,
)

# Use whisper-tiny for fast tests
MODEL_NAME = "openai/whisper-tiny"


@pytest.fixture(scope="module")
def tokenizer():
    return build_tokenizer(model_name=MODEL_NAME)


@pytest.fixture(scope="module")
def model_and_tokenizer():
    return build_model(WhisperParaphasiaConfig(model_name=MODEL_NAME))


@pytest.fixture(scope="module")
def feature_extractor():
    return WhisperFeatureExtractor.from_pretrained(MODEL_NAME)


def _example(text: str, labels: str) -> dict:
    return {
        "audio": {"array": np.zeros(16000, dtype=np.float32), "sampling_rate": 16000},
        "text": text,
        "labels": labels,
    }


# ---- Tokenizer ---------------------------------------------------------------


class TestTokenizer:
    def test_tags_are_single_tokens(self, tokenizer):
        ids = get_paraphasia_token_ids(tokenizer)
        assert set(ids) == set(PARAPHASIA_TOKENS)
        assert len(set(ids.values())) == 3

    def test_no_standalone_space_before_tag(self, tokenizer):
        tokens = tokenizer.convert_ids_to_tokens(
            tokenizer(" aphasia fekts [p] my", add_special_tokens=False).input_ids
        )
        assert "[p]" in tokens
        assert "Ġ" not in tokens

    def test_tags_survive_skip_special_tokens(self, tokenizer):
        ids = tokenizer(" the cat [s] sat").input_ids
        text = tokenizer.decode(ids, skip_special_tokens=True)
        assert normalize_output(text) == "the cat [s] sat"

    def test_build_is_idempotent(self, tokenizer, tmp_path):
        tokenizer.save_pretrained(tmp_path)
        reloaded = build_tokenizer(model_name=str(tmp_path))
        assert len(reloaded) == len(tokenizer)


class TestNormalizeOutput:
    def test_restores_tag_spacing(self):
        assert normalize_output("fekts[p] my language[n]") == "fekts [p] my language [n]"

    def test_lowercases_and_strips_punctuation(self):
        assert normalize_output(" Here's the Umbrella, okay.") == "heres the umbrella okay"

    def test_keeps_tag_brackets(self):
        assert normalize_output("Bowl [S]!") == "bowl [s]"


# ---- Model -------------------------------------------------------------------


class TestModel:
    def test_embeddings_resized(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        assert model.get_input_embeddings().weight.shape[0] == len(tokenizer)

    def test_new_embeddings_mean_initialized(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        embed = model.get_input_embeddings().weight
        n_new = len(PARAPHASIA_TOKENS)
        expected = embed[:-n_new].mean(dim=0)
        for row in embed[-n_new:]:
            assert torch.allclose(row, expected, atol=1e-5)

    def test_tags_not_suppressed(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        suppressed = set(model.generation_config.suppress_tokens or [])
        assert not suppressed & set(get_paraphasia_token_ids(tokenizer).values())

    def test_freeze_unfreeze_encoder(self, model_and_tokenizer):
        model, _ = model_and_tokenizer
        freeze_encoder(model)
        assert not any(p.requires_grad for p in model.model.encoder.parameters())
        assert any(p.requires_grad for p in model.model.decoder.parameters())
        unfreeze_encoder(model)
        assert all(p.requires_grad for p in model.model.encoder.parameters())

    def test_class_weights(self, tokenizer):
        weights = get_class_weight_tensor(tokenizer, tag_weight=5.0)
        tag_ids = list(get_paraphasia_token_ids(tokenizer).values())
        assert weights.shape == (len(tokenizer),)
        assert (weights[tag_ids] == 5.0).all()
        assert weights.sum() == len(tokenizer) - 3 + 15.0


# ---- Collator ----------------------------------------------------------------


class TestCollator:
    @pytest.fixture
    def collator(self, tokenizer, feature_extractor):
        return ParaphasiaDataCollator(feature_extractor=feature_extractor, tokenizer=tokenizer)

    def test_batch_shapes(self, collator):
        batch = collator([_example("the cat sat", "c p c"), _example("a dog", "c c")])
        assert batch["input_features"].shape[0] == 2
        assert batch["labels"].shape[0] == 2

    def test_labels_do_not_start_with_sot(self, collator, tokenizer):
        """The model prepends <|startoftranscript|> itself when shifting labels."""
        labels = collator([_example("the cat sat", "c p c")])["labels"]
        sot = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        assert labels[0, 0].item() != sot
        assert labels[0, 0].item() == tokenizer.convert_tokens_to_ids("<|en|>")

    def test_eos_kept_padding_masked(self, collator, tokenizer):
        batch = collator([_example("the cat sat on the mat", "c c c c c c"), _example("hi", "c")])
        short = batch["labels"][1]
        content = short[short != -100]
        assert content[-1].item() == tokenizer.eos_token_id
        assert (short == -100).any()

    def test_target_includes_tags(self, collator):
        assert collator.target_text({"text": "the cat sat", "labels": "c s c"}) == " the cat [s] sat"

    def test_tag_classes_filter(self, tokenizer, feature_extractor):
        collator = ParaphasiaDataCollator(
            feature_extractor=feature_extractor, tokenizer=tokenizer, tag_classes=("p", "n")
        )
        target = collator.target_text({"text": "the cot dog", "labels": "c p s"})
        assert target == " the cot [p] dog"

    def test_loss_is_finite(self, collator, model_and_tokenizer):
        model, _ = model_and_tokenizer
        batch = collator([_example("the cat sat", "c p c")])
        with torch.no_grad():
            loss = model(**batch).loss
        assert torch.isfinite(loss)

    def test_time_perturbation_changes_length(self, collator):
        audio = np.random.randn(16000).astype(np.float32)
        lengths = {len(collator._time_perturb(audio)) for _ in range(20)}
        assert len(lengths) > 1, "Time perturbation never changed audio length"

    def test_spec_augment_rates(self):
        assert 1.0 in SPEC_AUGMENT_RATES
        assert len(SPEC_AUGMENT_RATES) == 7


# ---- Inference ---------------------------------------------------------------


class TestInference:
    def test_split_long_audio(self):
        chunks = _split(np.zeros(16000 * 70, dtype=np.float32))
        assert len(chunks) == 3
        assert all(len(c) <= 16000 * 30 for c in chunks)
        assert sum(len(c) for c in chunks) == 16000 * 70

    def test_collapse_word_loop(self):
        assert collapse_loops(["you"] + ["stand"] * 300 + ["done"]) == ["you"] + ["stand"] * 8 + ["done"]

    def test_collapse_phrase_loop(self):
        tokens = ["a"] + ["fuh", "[n]"] * 50
        assert collapse_loops(tokens) == ["a"] + ["fuh", "[n]"] * 8

    def test_real_repetition_untouched(self):
        tokens = "and stir and stir and stir suhtuh [p] stir stir stir stir stir stir done".split()
        assert collapse_loops(tokens) == tokens

    def test_token_cap_covers_reference_rates(self):
        # Scripts-Fridriksson tops out at 5.35 tokens/s on a 0.75s utterance
        assert _token_cap(0.75) >= 5.35 * 0.75
        assert _token_cap(16.0) == 6 * 16 + 10

    def test_split_short_audio(self):
        assert len(_split(np.zeros(16000, dtype=np.float32))) == 1

    def test_predictor_roundtrip(self, model_and_tokenizer, feature_extractor, tmp_path):
        """A saved checkpoint reloads with the tags and decodes to a string."""
        model, tokenizer = model_and_tokenizer
        model.save_pretrained(tmp_path)
        tokenizer.save_pretrained(tmp_path)
        feature_extractor.save_pretrained(tmp_path)

        predictor = ParaphasiaPredictor(tmp_path, device="cpu")
        assert len(predictor.tokenizer) == len(tokenizer)
        predictor._gen_kwargs["max_new_tokens"] = 5
        out = predictor.predict_batch([np.zeros(16000, dtype=np.float32)] * 2)
        assert len(out) == 2 and all(isinstance(o, str) for o in out)
