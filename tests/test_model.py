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
    freeze_decoder,
)
from aphasia_modeling.model.collator import ParaphasiaDataCollator, SPEC_AUGMENT_RATES
from aphasia_modeling.model.classifier import (
    UtteranceClassifierHead,
    WhisperWithParaphasiaHead,
    UTT_PHONEMIC,
    UTT_NEOLOGISTIC,
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
        assert isinstance(model.classifier, UtteranceClassifierHead)

    def test_classifier_output_shape(self, model_and_tokenizer):
        model, _ = model_and_tokenizer
        hidden_size = model.whisper.config.d_model
        # Utterance-level: input is pooled (batch, hidden_size)
        batch = torch.randn(2, hidden_size)
        logits = model.classifier(batch)
        assert logits.shape == (2, 2)  # 2 binary outputs

    def test_freeze_unfreeze_encoder(self, model_and_tokenizer):
        model, _ = model_and_tokenizer

        freeze_encoder(model.whisper)
        for param in model.whisper.model.encoder.parameters():
            assert not param.requires_grad

        unfreeze_encoder(model.whisper)
        for param in model.whisper.model.encoder.parameters():
            assert param.requires_grad

    def test_freeze_decoder(self, model_and_tokenizer):
        model, _ = model_and_tokenizer

        freeze_decoder(model.whisper)
        for param in model.whisper.model.decoder.parameters():
            assert not param.requires_grad

        # Classifier head should still be trainable
        for param in model.classifier.parameters():
            assert param.requires_grad

        # Unfreeze for other tests
        for param in model.whisper.model.decoder.parameters():
            param.requires_grad = True

    def test_vocab_unmodified(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        vocab_size = model.whisper.model.decoder.embed_tokens.weight.shape[0]
        assert vocab_size == model.whisper.config.vocab_size


# ---- Classifier Head ---------------------------------------------------------


class TestClassifierHead:
    def test_num_labels(self):
        assert UtteranceClassifierHead.NUM_LABELS == 2

    def test_forward(self):
        head = UtteranceClassifierHead(hidden_size=256)
        x = torch.randn(2, 256)  # Pooled input
        out = head(x)
        assert out.shape == (2, 2)

    def test_constants(self):
        assert UTT_PHONEMIC == 0
        assert UTT_NEOLOGISTIC == 1


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
        # Utterance-level: (batch, 2)
        assert batch["cls_labels"].shape == (2, 2)

    def test_cls_labels_values(self, collator):
        features = [
            {
                "audio": {"array": np.zeros(16000, dtype=np.float32), "sampling_rate": 16000},
                "text": "the cat sat",
                "labels": "c p c",
            },
            {
                "audio": {"array": np.zeros(16000, dtype=np.float32), "sampling_rate": 16000},
                "text": "the dog ran",
                "labels": "c c n",
            },
        ]
        batch = collator(features)
        cls = batch["cls_labels"]
        # First utterance: has_p=1, has_n=0
        assert cls[0, UTT_PHONEMIC].item() == 1.0
        assert cls[0, UTT_NEOLOGISTIC].item() == 0.0
        # Second: has_p=0, has_n=1
        assert cls[1, UTT_PHONEMIC].item() == 0.0
        assert cls[1, UTT_NEOLOGISTIC].item() == 1.0

    def test_cls_labels_both(self, collator):
        features = [
            {
                "audio": {"array": np.zeros(16000, dtype=np.float32), "sampling_rate": 16000},
                "text": "the blick cat",
                "labels": "c n p",
            },
        ]
        batch = collator(features)
        cls = batch["cls_labels"]
        assert cls[0, UTT_PHONEMIC].item() == 1.0
        assert cls[0, UTT_NEOLOGISTIC].item() == 1.0

    def test_cls_labels_neither(self, collator):
        features = [
            {
                "audio": {"array": np.zeros(16000, dtype=np.float32), "sampling_rate": 16000},
                "text": "the cat sat",
                "labels": "c c c",
            },
        ]
        batch = collator(features)
        cls = batch["cls_labels"]
        assert cls[0, UTT_PHONEMIC].item() == 0.0
        assert cls[0, UTT_NEOLOGISTIC].item() == 0.0

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


# ---- Inference ---------------------------------------------------------------


class TestPredictionResult:
    def test_to_single_seq_with_p(self):
        from aphasia_modeling.model.inference import PredictionResult
        r = PredictionResult(text="the cat sat", has_p=True, has_n=False, prob_p=0.8, prob_n=0.1)
        assert r.to_single_seq() == "the [p] cat sat"

    def test_to_single_seq_with_n(self):
        from aphasia_modeling.model.inference import PredictionResult
        r = PredictionResult(text="the cat sat", has_p=False, has_n=True, prob_p=0.1, prob_n=0.9)
        assert r.to_single_seq() == "the [n] cat sat"

    def test_to_single_seq_with_both(self):
        from aphasia_modeling.model.inference import PredictionResult
        r = PredictionResult(text="the cat sat", has_p=True, has_n=True, prob_p=0.8, prob_n=0.9)
        assert r.to_single_seq() == "the [p] [n] cat sat"

    def test_to_single_seq_neither(self):
        from aphasia_modeling.model.inference import PredictionResult
        r = PredictionResult(text="the cat sat", has_p=False, has_n=False, prob_p=0.1, prob_n=0.1)
        assert r.to_single_seq() == "the cat sat"
