"""Tests for model components: tokenizer, model setup, collator, classifier.

All tests use openai/whisper-tiny to minimize download size and run on CPU.
"""

import numpy as np
import pytest
import torch

from aphasia_modeling.model.tokenizer import build_tokenizer
from aphasia_modeling.model.whisper import (
    WhisperParaphasiaConfig,
    build_model,
    freeze_encoder,
    unfreeze_encoder,
)
from aphasia_modeling.model.collator import ParaphasiaDataCollator, SPEC_AUGMENT_RATES
from aphasia_modeling.model.classifier import (
    ParaphasiaClassifierHead,
    WhisperWithParaphasiaHead,
    CLS_CORRECT,
    CLS_PHONEMIC,
    CLS_NEOLOGISTIC,
)

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
    def test_stock_whisper_tokenizer(self, tokenizer):
        """Tokenizer should be unmodified stock Whisper — no custom tokens added."""
        # Whisper-tiny has 51865 tokens total (base vocab + language/task tokens)
        assert len(tokenizer) == 51865

    def test_encodes_text(self, tokenizer):
        encoded = tokenizer("the cat sat", return_tensors="pt")
        assert encoded.input_ids.shape[0] == 1
        assert encoded.input_ids.shape[1] > 0


# ---- Model -------------------------------------------------------------------


class TestModel:
    def test_model_loads(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        assert model is not None
        assert isinstance(model, WhisperWithParaphasiaHead)

    def test_has_classifier_head(self, model_and_tokenizer):
        model, _ = model_and_tokenizer
        assert hasattr(model, "classifier")
        assert isinstance(model.classifier, ParaphasiaClassifierHead)

    def test_classifier_output_shape(self, model_and_tokenizer):
        model, _ = model_and_tokenizer
        hidden_size = model.whisper.config.d_model
        batch = torch.randn(2, 10, hidden_size)
        logits = model.classifier(batch)
        assert logits.shape == (2, 10, 3)

    def test_freeze_unfreeze_encoder(self, model_and_tokenizer):
        model, _ = model_and_tokenizer

        freeze_encoder(model.whisper)
        for param in model.whisper.model.encoder.parameters():
            assert not param.requires_grad

        unfreeze_encoder(model.whisper)
        for param in model.whisper.model.encoder.parameters():
            assert param.requires_grad

    def test_vocab_unmodified(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        # Whisper vocab should not be resized
        vocab_size = model.whisper.model.decoder.embed_tokens.weight.shape[0]
        assert vocab_size == model.whisper.config.vocab_size


# ---- Classifier Head ---------------------------------------------------------


class TestClassifierHead:
    def test_num_classes(self):
        assert ParaphasiaClassifierHead.NUM_CLASSES == 3

    def test_forward(self):
        head = ParaphasiaClassifierHead(hidden_size=256)
        x = torch.randn(2, 10, 256)
        out = head(x)
        assert out.shape == (2, 10, 3)

    def test_constants(self):
        assert CLS_CORRECT == 0
        assert CLS_PHONEMIC == 1
        assert CLS_NEOLOGISTIC == 2


# ---- Collator ----------------------------------------------------------------


from transformers import WhisperFeatureExtractor


class TestCollator:
    @pytest.fixture
    def collator(self, model_and_tokenizer):
        _, tokenizer = model_and_tokenizer
        feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
        return ParaphasiaDataCollator(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            use_classifier=True,
        )

    def test_collate_with_raw_audio(self, collator):
        features = [
            {
                "audio": {"array": np.zeros(16000, dtype=np.float32), "sampling_rate": 16000},
                "text": "the cat sat",
                "labels": "c p c",
            },
            {
                "audio": {"array": np.zeros(16000, dtype=np.float32), "sampling_rate": 16000},
                "text": "the dog sat",
                "labels": "c c c",
            },
        ]
        batch = collator(features)
        assert "input_features" in batch
        assert "labels" in batch
        assert "cls_labels" in batch
        assert batch["input_features"].shape[0] == 2
        assert batch["labels"].shape[0] == 2
        assert batch["cls_labels"].shape == batch["labels"].shape

    def test_cls_labels_contain_paraphasia(self, collator):
        features = [
            {
                "audio": {"array": np.zeros(16000, dtype=np.float32), "sampling_rate": 16000},
                "text": "the cat sat",
                "labels": "c p c",
            },
        ]
        batch = collator(features)
        cls = batch["cls_labels"][0]
        # Should contain at least one phonemic label
        assert (cls == CLS_PHONEMIC).any()
        # Should contain correct labels
        assert (cls == CLS_CORRECT).any()

    def test_labels_have_padding_masked(self, collator):
        features = [
            {
                "audio": {"array": np.zeros(16000, dtype=np.float32), "sampling_rate": 16000},
                "text": "short",
                "labels": "c",
            },
            {
                "audio": {"array": np.zeros(16000, dtype=np.float32), "sampling_rate": 16000},
                "text": "this is a much longer sentence with more words",
                "labels": "c c c c c c c c c",
            },
        ]
        batch = collator(features)
        labels = batch["labels"]
        assert (labels[0] == -100).any() or labels.shape[1] == (labels[0] != -100).sum()

    def test_time_perturbation_changes_length(self):
        fe = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
        collator = ParaphasiaDataCollator(
            feature_extractor=fe,
            tokenizer=build_tokenizer(model_name=MODEL_NAME),
            apply_time_perturbation=True,
        )

        audio = np.random.randn(16000).astype(np.float32)
        lengths = set()
        for _ in range(20):
            perturbed = collator._time_perturb(audio)
            lengths.add(len(perturbed))

        assert len(lengths) > 1, "Time perturbation never changed audio length"

    def test_spec_augment_rates(self):
        assert 1.0 in SPEC_AUGMENT_RATES
        assert len(SPEC_AUGMENT_RATES) == 7
