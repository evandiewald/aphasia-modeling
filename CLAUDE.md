# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project building a model for joint ASR and multiclass paraphasia detection (phonemic, neologistic, semantic) from aphasic speech. Uses Whisper as the backbone instead of HuBERT, with results comparable to the CHAI Lab "Beyond Binary" paper (Perez et al., Interspeech 2024). A secondary LLM-based stage targets semantic paraphasia detection from transcripts.

## Setup

- Python 3.12 (see `.python-version`)
- Package manager: uv (uses `pyproject.toml`)
- Install deps: `uv sync`
- Run: `uv run python main.py`

## Architecture

**Two-stage pipeline:**

1. **Stage 1 — Whisper + Paraphasia Tokens:** Fine-tuned Whisper (`openai/whisper-small`) with 3 added special tokens (`[p]`, `[n]`, `[s]`) for inline paraphasia classification. Uses HuggingFace Transformers (`WhisperForConditionalGeneration`). Training is two-phase: ASR adaptation on AphasiaBank Protocol (~100h), then paraphasia-aware fine-tuning on Fridriksson subset (~3h).

2. **Stage 2 — LLM Semantic Detection:** Prompt-based LLM pass over Stage 1 transcripts to catch semantic paraphasias (real words substituted for intended words), which are acoustically indistinguishable from correct speech. Conditional on exploration phase showing ≥20% of semantic paraphasias are detectable without target text.

**Key datasets:** AphasiaBank (primary, required), SONIVA (optional, ASR pretraining only). Must use identical train/dev/test splits as CHAI for comparability.

**Evaluation metrics:** WER, AWER, TD-binary, TD-multiclass (TD-[p], TD-[n], TD-[s], TD-all), utterance-level binary F1. Statistical significance via bootstrap (WER/AWER) and repeated measures ANOVA + Tukey (TD). See `docs/paraphasia_detection_plan.md` for baseline numbers and full metric definitions.

## Commands

```bash
uv sync --extra dev                        # Install dependencies (including pytest)
uv run pytest tests/                       # Run all tests
uv run pytest tests/test_preprocess.py -k "test_phonemic"  # Run a single test
uv run python main.py preprocess <cha_dir> # Parse & preprocess CHAT files
uv run python main.py evaluate <ref> <hyp> # Run evaluation metrics

# Training (on GPU instance)
python scripts/train.py --phase 1 --data_path data/protocol.json --output_dir checkpoints/phase1
python scripts/train.py --phase 2 --data_path data/fridriksson.json --output_dir checkpoints/phase2 --loso --class_weights

# Evaluation (on GPU instance)
python scripts/evaluate_model.py --model_path checkpoints/phase2/fold_spk --data_path data/fridriksson.json --test_speaker spk
```

## Code Layout

- `src/aphasia_modeling/data/` — Data pipeline
  - `chat_parser.py` — Parses `.cha` files via pylangacq, extracts `*PAR:` utterances with timing
  - `preprocess.py` — CHAI-compatible cleaning: bracket handling, error code extraction (`[* p]` → `p`), IPA-to-pseudoword, single-seq format (`"word [p] word [n]"`)
  - `dataset.py` — `AphasiaBankDataset` class with LOSO cross-validation (12 folds, seed 883, 10% dev), HuggingFace Dataset conversion, JSON serialization
- `src/aphasia_modeling/model/` — Whisper training and inference
  - `tokenizer.py` — Extends Whisper tokenizer with `[p]`, `[n]`, `[s]` special tokens
  - `whisper.py` — Model setup: load pretrained Whisper, resize embeddings, mean-init new tokens, freeze/unfreeze encoder, class weight tensor
  - `collator.py` — Data collator: audio feature extraction, target tokenization, padding, SpecAugment time perturbation
  - `trainer.py` — `ParaphasiaTrainer` (extends `Seq2SeqTrainer`) with per-token class-weighted cross-entropy loss
  - `inference.py` — `ParaphasiaPredictor`: load checkpoint, decode audio to single-seq format
- `src/aphasia_modeling/evaluation/` — Metrics matching CHAI exactly
  - `alignment.py` — Levenshtein alignment with paraphasia tag reinsertion
  - `metrics.py` — WER, AWER, AWER-PD, TD-binary, TD-multiclass, utterance-level F1
  - `significance.py` — Bootstrap (WER/AWER) and ANOVA+Tukey (TD)
- `scripts/` — Standalone scripts for GPU training
  - `train.py` — Main training script (Phase 1 ASR adaptation + Phase 2 paraphasia fine-tuning, LOSO CV support)
  - `evaluate_model.py` — Run inference + compute all CHAI metrics on test set

## Key Reference

- Detailed technical plan: `docs/paraphasia_detection_plan.md`
- CHAI Lab repo (baseline to compare against): https://github.com/chailab-umich/BeyondBinary-ParaphasiaDetection
