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

1. **Stage 1 — Whisper + Paraphasia Tokens:** Fine-tuned Whisper (`openai/whisper-small`) with 3 added tag tokens (`[p]`, `[n]`, `[s]`) emitted inline after paraphasic words (CHAI's single-seq format). Uses HuggingFace Transformers (`WhisperForConditionalGeneration`). Currently trained only on Scripts-Fridriksson with 12-fold LOSO; `--tag_classes pn` leaves `[s]` to Stage 2. CHAI additionally pretrained on ~100h of Protocol data (not yet replicated).

2. **Stage 2 — LLM Semantic Detection:** Prompt-based LLM pass over Stage 1 transcripts to catch semantic paraphasias (real words substituted for intended words), which are acoustically indistinguishable from correct speech. Conditional on exploration phase showing ≥20% of semantic paraphasias are detectable without target text.

**Key datasets:** CHAI evaluates on **Scripts**-Fridriksson (`datasets/Scripts/Fridriksson`, participants reading 4 fixed scripts; ~3h, 12 speakers after dropping `xxx` utterances), NOT the Protocol Fridriksson corpora in `datasets/Protocol/` (spontaneous discourse). Built dataset: `datasets/Scripts/scripts_fridriksson.json`. Because every speaker reads the same scripts, speaker-held-out folds still share text across train/test. SONIVA is optional, ASR pretraining only.

**Evaluation metrics:** WER, AWER, TD-binary, TD-multiclass (TD-[p], TD-[n], TD-[s], TD-all), utterance-level binary F1. Statistical significance via bootstrap (WER/AWER) and repeated measures ANOVA + Tukey (TD). See `docs/paraphasia_detection_plan.md` for baseline numbers and full metric definitions.

## Commands

```bash
uv sync --extra dev                        # Install dependencies (including pytest)
uv run pytest tests/                       # Run all tests
uv run pytest tests/test_preprocess.py -k "test_phonemic"  # Run a single test
uv run python main.py preprocess <cha_dir> # Parse & preprocess CHAT files
uv run python main.py evaluate <ref> <hyp> # Run evaluation metrics

uv run python main.py preprocess datasets/Scripts/Fridriksson/PWA --audio-dir datasets/Scripts/Fridriksson/audio -o datasets/Scripts/scripts_fridriksson.json

# Training (on GPU instance) — one checkpoint per fold under output_dir/fold_<spk>
python scripts/train.py --data_path datasets/Scripts/scripts_fridriksson.json --output_dir checkpoints/scripts-small --loso --bf16
python scripts/train.py ... --test_speaker P1      # single fold
python scripts/train.py ... --max_steps 4 --model_name openai/whisper-tiny  # local smoke test

# Evaluation — pools predictions over all folds (as CHAI reports)
python scripts/evaluate_model.py --model_dir checkpoints/scripts-small --data_path datasets/Scripts/scripts_fridriksson.json --loso
```

## Code Layout

- `src/aphasia_modeling/data/` — Data pipeline
  - `chat_parser.py` — Parses `.cha` files via pylangacq, extracts `*PAR:` utterances with timing
  - `preprocess.py` — CHAI-compatible cleaning: bracket handling, error code extraction (`[* p]` → `p`), IPA-to-pseudoword, single-seq format (`"word [p] word [n]"`)
  - `dataset.py` — `AphasiaBankDataset` class with LOSO cross-validation (12 folds, seed 883, 10% dev), HuggingFace Dataset conversion, JSON serialization
- `src/aphasia_modeling/model/` — Whisper training and inference
  - `tokenizer.py` — Adds `[p]`, `[n]`, `[s]` as non-special tokens with lstrip (no stray space token before tags); output normalization
  - `whisper.py` — Model setup: load pretrained Whisper, resize embeddings, mean-init new tokens, freeze/unfreeze encoder, class weight tensor
  - `collator.py` — Data collator: audio loading, single-seq targets (drops the duplicate `<|startoftranscript|>`), `tag_classes` filter, time perturbation
  - `trainer.py` — `ParaphasiaTrainer` (extends `Seq2SeqTrainer`) with optional tag-token loss weighting (`--tag_weight`)
  - `inference.py` — `ParaphasiaPredictor`: decode audio to single-seq format; >30s audio is chunked; decoder loops are collapsed (no n-gram blocking, which would delete real repetitions)
- `src/aphasia_modeling/evaluation/` — Metrics matching CHAI exactly
  - `alignment.py` — Levenshtein alignment with paraphasia tag reinsertion
  - `metrics.py` — WER, AWER, AWER-PD, TD-binary, TD-multiclass, utterance-level F1
  - `significance.py` — Bootstrap (WER/AWER) and ANOVA+Tukey (TD)
- `scripts/` — Standalone scripts for GPU training
  - `train.py` — Single-seq fine-tuning, one fold (`--test_speaker`) or all LOSO folds (`--loso`)
  - `evaluate_model.py` — Run inference per fold + compute all CHAI metrics, pooled and per fold

## Key Reference

- Detailed technical plan: `docs/paraphasia_detection_plan.md`
- CHAI Lab repo (baseline to compare against): https://github.com/chailab-umich/BeyondBinary-ParaphasiaDetection
