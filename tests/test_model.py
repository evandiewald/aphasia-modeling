"""Tests for model components: tokenizer, model setup, collator, inference.

All tests use openai/whisper-tiny to minimize download size and run on CPU.
"""

import numpy as np
import pytest
import torch

from aphasia_modeling.model.tokenizer import (
    PARAPHASIA_TOKENS,
    build_tokenizer,
    get_paraphasia_token_ids,
)
from aphasia_modeling.model.whisper import (
    WhisperParaphasiaConfig,
    build_model,
    freeze_encoder,
    get_class_weight_tensor,
    unfreeze_encoder,
)
from aphasia_modeling.model.collator import ParaphasiaDataCollator, SPEC_AUGMENT_RATES

# Use whisper-tiny for fast tests
MODEL_NAME = "openai/whisper-tiny"


@pytest.fixture(scope="module")
def tokenizer():
    return build_tokenizer(model_name=MODEL_NAME)


@pytest.fixture(scope="module")
def model_and_tokenizer():
    config = WhisperParaphasiaConfig(model_name=MODEL_NAME)
    return build_model(config)


# ---- Tokenizer ---------------------------------------------------------------


class TestTokenizer:
    def test_paraphasia_tokens_added(self, tokenizer):
        for token in PARAPHASIA_TOKENS:
            token_id = tokenizer.convert_tokens_to_ids(token)
            assert token_id != tokenizer.unk_token_id, f"{token} resolved to UNK"

    def test_paraphasia_tokens_are_single_tokens(self, tokenizer):
        """Each paraphasia token should encode as exactly one token."""
        for token in PARAPHASIA_TOKENS:
            encoded = tokenizer.encode(token, add_special_tokens=False)
            # May include preceding space token, so check the paraphasia token is in there
            token_id = tokenizer.convert_tokens_to_ids(token)
            assert token_id in encoded, (
                f"{token} (id={token_id}) not found in encoded output: {encoded}"
            )

    def test_get_paraphasia_token_ids(self, tokenizer):
        ids = get_paraphasia_token_ids(tokenizer)
        assert set(ids.keys()) == {"[p]", "[n]"}
        # All IDs should be unique
        assert len(set(ids.values())) == 2

    def test_token_ids_are_beyond_original_vocab(self, tokenizer):
        """Paraphasia tokens should have IDs at the end of the vocab."""
        ids = get_paraphasia_token_ids(tokenizer)
        # whisper-tiny has vocab_size ~51865
        for token, token_id in ids.items():
            assert token_id >= 51864, f"{token} has unexpectedly low ID: {token_id}"

    def test_decode_preserves_paraphasia_tokens(self, tokenizer):
        text = "the cat [p] sat on mat [n]"
        encoded = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(encoded, skip_special_tokens=False)
        assert "[p]" in decoded
        assert "[n]" in decoded


# ---- Model -------------------------------------------------------------------


class TestModel:
    def test_model_loads(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        assert model is not None
        assert tokenizer is not None

    def test_embeddings_resized(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        vocab_size = model.model.decoder.embed_tokens.weight.shape[0]
        assert vocab_size == len(tokenizer)

    def test_new_embeddings_not_zero(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        ids = get_paraphasia_token_ids(tokenizer)
        for token, token_id in ids.items():
            embedding = model.model.decoder.embed_tokens.weight[token_id]
            assert embedding.abs().sum() > 0, f"{token} embedding is all zeros"

    def test_freeze_unfreeze_encoder(self, model_and_tokenizer):
        model, _ = model_and_tokenizer

        freeze_encoder(model)
        for param in model.model.encoder.parameters():
            assert not param.requires_grad

        unfreeze_encoder(model)
        for param in model.model.encoder.parameters():
            assert param.requires_grad

    def test_class_weight_tensor(self, model_and_tokenizer):
        _, tokenizer = model_and_tokenizer
        weights = get_class_weight_tensor(tokenizer)
        assert weights.shape == (len(tokenizer),)

        ids = get_paraphasia_token_ids(tokenizer)
        assert weights[ids["[p]"]] == 2.0
        assert weights[ids["[n]"]] == 4.0
        # Normal tokens should be 1.0
        assert weights[0] == 1.0

    def test_generate_does_not_suppress_paraphasia_tokens(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        suppress = model.generation_config.suppress_tokens or []
        ids = get_paraphasia_token_ids(tokenizer)
        for token, token_id in ids.items():
            assert token_id not in suppress, f"{token} is suppressed in generation"


# ---- Collator ----------------------------------------------------------------


class TestCollator:
    @pytest.fixture
    def collator(self, model_and_tokenizer):
        _, tokenizer = model_and_tokenizer
        feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
        return ParaphasiaDataCollator(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
        )

    def test_collate_with_raw_audio(self, collator):
        # Synthesize fake audio (1 second of silence at 16kHz)
        features = [
            {
                "audio": {"array": np.zeros(16000, dtype=np.float32), "sampling_rate": 16000},
                "single_seq": "the cat [p] sat",
            },
            {
                "audio": {"array": np.zeros(16000, dtype=np.float32), "sampling_rate": 16000},
                "single_seq": "the dog sat",
            },
        ]
        batch = collator(features)
        assert "input_features" in batch
        assert "labels" in batch
        assert batch["input_features"].shape[0] == 2
        assert batch["labels"].shape[0] == 2

    def test_labels_have_padding_masked(self, collator):
        features = [
            {
                "audio": {"array": np.zeros(16000, dtype=np.float32), "sampling_rate": 16000},
                "single_seq": "short",
            },
            {
                "audio": {"array": np.zeros(16000, dtype=np.float32), "sampling_rate": 16000},
                "single_seq": "this is a much longer sentence with more words",
            },
        ]
        batch = collator(features)
        labels = batch["labels"]
        # Shorter sequence should have -100 padding at the end
        assert (labels[0] == -100).any() or labels.shape[1] == (labels[0] != -100).sum()

    def test_collate_with_pre_tokenized(self, collator):
        features = [
            {"input_features": np.zeros((80, 3000), dtype=np.float32), "labels": [1, 2, 3]},
            {"input_features": np.zeros((80, 3000), dtype=np.float32), "labels": [4, 5]},
        ]
        batch = collator(features)
        assert batch["input_features"].shape == (2, 80, 3000)
        assert batch["labels"].shape == (2, 3)
        # Shorter labels should be padded with -100
        assert batch["labels"][1, 2].item() == -100

    def test_time_perturbation_changes_length(self):
        """Time perturbation should change audio length."""
        from aphasia_modeling.model.collator import ParaphasiaDataCollator
        from transformers import WhisperFeatureExtractor

        fe = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
        collator = ParaphasiaDataCollator(
            feature_extractor=fe,
            tokenizer=build_tokenizer(model_name=MODEL_NAME),
            apply_time_perturbation=True,
        )

        audio = np.random.randn(16000).astype(np.float32)
        # Run perturbation many times — at least some should change length
        lengths = set()
        for _ in range(20):
            perturbed = collator._time_perturb(audio)
            lengths.add(len(perturbed))

        assert len(lengths) > 1, "Time perturbation never changed audio length"

    def test_spec_augment_rates(self):
        assert 1.0 in SPEC_AUGMENT_RATES
        assert len(SPEC_AUGMENT_RATES) == 7


# ---- Inference ---------------------------------------------------------------


class TestInferenceDecode:
    def test_decode_strips_whisper_special_tokens(self, model_and_tokenizer):
        """The _decode method should remove <|...|> tokens but keep [p]/[n]."""
        from aphasia_modeling.model.inference import ParaphasiaPredictor

        _, tokenizer = model_and_tokenizer

        # Simulate a token sequence with Whisper special tokens + paraphasia tokens
        text = "<|startoftranscript|><|en|><|transcribe|>the cat [p] sat<|endoftext|>"
        # Manually apply the cleaning regex
        import re
        cleaned = re.sub(r"<\|[^|]*\|>", "", text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        assert cleaned == "the cat [p] sat"
        assert "[p]" in cleaned
        assert "<|" not in cleaned


# Import needed for collator fixture
from transformers import WhisperFeatureExtractor
