# Paraphasia Detection from Aphasic Speech: Technical Plan

## Problem Statement

Build a model that jointly performs ASR and multiclass paraphasia detection (phonemic, neologistic, semantic) from aphasic speech audio, using Whisper as the backbone instead of HuBERT. Results must be directly comparable to the CHAI Lab "Beyond Binary" paper (Perez et al., Interspeech 2024).

A secondary LLM-based stage targets semantic paraphasia detection from the transcript, since acoustic models are poorly suited for that error type.

---

## Prior Work This Builds On

### CHAI Lab — "Beyond Binary: Multiclass Paraphasia Detection" (Perez et al., 2024)
- **Repo:** https://github.com/chailab-umich/BeyondBinary-ParaphasiaDetection
- **Architecture:** HuBERT-large encoder (24 transformer layers) + 6-layer transformer decoder
- **Task:** Joint ASR + inline paraphasia classification as a single output sequence
- **Tokenizer:** BPE with vocab size 500 + 3 special tokens (`[p]`, `[n]`, `[s]`)
- **Key model:** "single-seq" — decoder produces interleaved word tokens and paraphasia labels (e.g., `aphasia fekts [p] my language not my ditikalt [n]`)
- **Training:** Pretrain on AphasiaBank Protocol (~100h), fine-tune on Fridriksson subset (~3h)
- **Toolkit:** SpeechBrain
- **Loss:** Joint CTC-attention with SpecAugment time perturbation

### SONIVA — "When Whisper Listens to Aphasia" (Sanguedolce et al., 2024)
- **Architecture:** Whisper (various sizes, best results with Small/Medium), fine-tuned end-to-end
- **Task:** ASR only — no paraphasia classification
- **Dataset:** SONIVA (~10h labeled aphasic speech from ~350 patients, British English, picture description task from CAT)
- **Key result:** Fine-tuning Whisper on SONIVA reduced WER from ~44% to ~22% on aphasic speech, and generalized to AphasiaBank and DementiaBank
- **Annotations:** Orthographic transcriptions + IPA phonetic tier (`%pho`), but NO word-level paraphasia type labels (`[p]`, `[n]`, `[s]`)

---

## What We're Building

**Core idea:** Whisper's architecture (encoder-decoder) with CHAI's training objective (single-seq ASR + inline paraphasia tokens). Concretely:

1. Start from a pretrained Whisper checkpoint (recommend Whisper-small as baseline, Whisper-medium as scale-up)
2. Extend Whisper's tokenizer with 3 special tokens: `[p]`, `[n]`, `[s]`
3. Train the model to produce transcripts with inline paraphasia labels, exactly matching CHAI's single-seq output format
4. Add a second-stage LLM pass over the transcript for semantic paraphasia detection

---

## Stage 1: Whisper + Paraphasia Tokens

### 1.1 Data Preparation

**Primary dataset: AphasiaBank** (same as CHAI — required for comparable evaluation)

- Obtain AphasiaBank membership (free for researchers): https://aphasia.talkbank.org
- Extract the Protocol dataset (~100h, used for pretraining/initial training)
- Extract the Fridriksson subset (~3h, used for fine-tuning and evaluation)
- You MUST use the same train/dev/test splits as CHAI for comparable results. Their splits are in the repo under `AphasiaBank/kaldi_data_prep`. Replicate these exactly.

**Preprocessing pipeline (replicate CHAI's exactly):**

1. Parse `.cha` files to extract `*PAR:` (participant) utterances with timestamps
2. Discard utterances marked as unintelligible or with overlapping speech
3. Lowercase all text, remove punctuation
4. Convert IPA-transcribed non-word errors to pseudo-words:
   - Map IPA pronunciations → phone sequences → grapheme sequences
   - e.g., `fEkts@u` → `fekts`
5. Convert CHAT error codes to inline labels:
   - `[* p]` → `[p]` (phonemic paraphasia)
   - `[* n]` → `[n]` (neologistic paraphasia)
   - `[* s]` → `[s]` (semantic paraphasia)
6. Strip target word annotations (`[: affects]`) — the model should NOT see intended targets
7. Final format per utterance: `aphasia fekts [p] my language not my ditikalt [n]`

Refer to `AphasiaBank/kaldi_data_prep` in the CHAI repo for their exact preprocessing scripts. The output format must match theirs for evaluation compatibility.

**Optional supplementary data: SONIVA** (for encoder adaptation only)

- Request access: https://github.com/Clinical-Language-Cognition-Lab/SONIVA_paper3
- SONIVA has orthographic + IPA transcriptions but NO paraphasia type labels
- Can only be used for ASR pretraining/encoder adaptation, NOT for paraphasia classification training
- Note: SONIVA is British English; AphasiaBank is American English. Be aware of accent mismatch.
- If used, fine-tune Whisper on SONIVA for ASR first, THEN add paraphasia tokens and fine-tune on AphasiaBank. This is a two-phase approach.

### 1.2 Model Architecture

**Base model:** `openai/whisper-small` (244M params) from HuggingFace

**Tokenizer modifications:**
- Load Whisper's existing tokenizer (~50k BPE tokens)
- Add 3 special tokens: `[p]`, `[n]`, `[s]`
- Resize the model's token embeddings accordingly (`model.resize_token_embeddings()`)
- Initialize new token embeddings randomly or as the mean of existing embeddings

**Architecture is otherwise unmodified Whisper** — the encoder-decoder structure is kept as-is. The decoder learns to emit paraphasia tokens in the output sequence through fine-tuning.

**Framework:** HuggingFace Transformers + a training script. The CHAI paper used SpeechBrain, but since we're using Whisper (which has excellent HuggingFace support), it's simpler to stay in that ecosystem. Use the `WhisperForConditionalGeneration` class.

### 1.3 Training

**Phase 1 — ASR adaptation (optional but recommended):**
- Fine-tune Whisper on AphasiaBank Protocol data for ASR only (no paraphasia tokens)
- This adapts the encoder to aphasic speech characteristics
- If using SONIVA, do this phase on SONIVA + Protocol combined
- Standard Whisper fine-tuning recipe (HuggingFace Seq2SeqTrainer)

**Phase 2 — Paraphasia-aware fine-tuning:**
- Fine-tune on Fridriksson subset with paraphasia token labels in the target sequences
- Use the Phase 1 checkpoint as initialization (or base Whisper if skipping Phase 1)
- Cross-entropy loss over the full vocabulary including paraphasia tokens
- Learning rate: start low (1e-5 or lower) to avoid catastrophic forgetting
- SpecAugment: use time perturbation rates matching CHAI: [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]

**Hyperparameters to sweep:**
- Learning rate: {1e-5, 5e-6, 1e-6}
- Batch size: limited by GPU memory, use gradient accumulation to reach effective batch ~16
- Number of epochs on Fridriksson: monitor dev set, expect ~10-30 epochs given small data size
- Whether to freeze encoder during Phase 2 (try both)

**Hardware:** Single A100 (40GB or 80GB) is sufficient. Whisper-small fine-tuning on ~100h should take a few hours. Fridriksson fine-tuning (~3h of audio) will be very fast.

### 1.4 Evaluation (Must Match CHAI Exactly)

**Use the same test folds as CHAI.** Their repo contains the data split definitions. They use cross-validation on Fridriksson — replicate the same fold structure.

**Metrics to compute (all from the CHAI paper):**

1. **WER** — Standard word error rate on the ASR output (paraphasia tokens stripped). Measures transcription quality.

2. **AWER (Augmented Word Error Rate)** — WER computed on the full output including paraphasia labels. Sequences are formatted as `word label word label ...` for both prediction and ground truth. Measures joint ASR + classification quality.

3. **TD-binary (Temporal Distance, binary)** — Align predicted and ground truth word sequences via minimum edit distance. For each predicted paraphasia (any type), compute the distance to the nearest ground truth paraphasia (any type). Average across utterances. Lower is better. Measures: can the model find WHERE paraphasias occur?

4. **TD-multiclass (TD-[p], TD-[n], TD-[s], TD-all)** — Same as TD-binary but per paraphasia type. TD-[p] measures proximity of predicted phonemic paraphasias to ground truth phonemic paraphasias, etc. TD-all is the sum across types. Lower is better.

5. **Utterance-level binary F1** — For each paraphasia type, compute F1 at the utterance level: does the utterance contain at least one instance of this paraphasia type? Compute separately for phonemic, neologistic, and semantic.

**Statistical significance testing:**
- WER/AWER: Bootstrap estimate, 1000 iterations, batch size 100, 95% confidence (matching CHAI)
- TD metrics: Repeated measures ANOVA + post-hoc Tukey test, p < 0.05 (matching CHAI)

**Output standardization:** Before evaluation, standardize all model outputs to the format: `word [label] word [label] ...` where every word has an explicit label (use `[c]` for correct words if needed for alignment). See Table 1 and Section 5.1 of the CHAI paper.

**Baseline numbers to beat (from CHAI paper, Table 2):**

| Model | WER | AWER | TD-bin | TD-[p] | TD-[n] | TD-[s] | TD-all |
|-------|-----|------|--------|--------|--------|--------|--------|
| ASR+GPT-4 | 38.6 | 32.7 | 0.68 | 0.86 | 0.73 | 0.29 | 1.88 |
| Single-Seq (CHAI) | 37.6 | 32.8 | 0.63 | 0.76 | 0.45 | 0.31 | 1.51 |
| Multi-Seq (CHAI) | 44.8 | 42.9 | 0.86 | 0.90 | 0.72 | 0.53 | 2.15 |

And utterance-level F1 (from their Figure 2, approximate):

| Model | F1-[p] | F1-[n] | F1-[s] |
|-------|--------|--------|--------|
| ASR+GPT-4 | ~0.52 | ~0.63 | ~0.15 |
| Single-Seq (CHAI) | ~0.56 | ~0.61 | ~0.12 |

---

## Stage 2: LLM-Based Semantic Paraphasia Detection (Exploratory)

### 2.1 Motivation

Semantic paraphasias (e.g., saying "table" instead of "chair") are acoustically indistinguishable from correct words. All models in the CHAI paper struggle with them (F1 ~0.12-0.15). An LLM operating on the transcript with full conversational context might be better positioned for this task — but this is genuinely unclear and requires exploration.

### 2.2 The Core Difficulty

**Important constraint:** The CHAI model does NOT see the target script during training or inference. It receives raw audio only. The target scripts were used only by the SLPs during annotation to identify errors. To produce comparable results, the Stage 2 LLM must also NOT see the target script. It only gets the transcript.

This makes semantic paraphasia detection very hard. Many semantic paraphasias produce perfectly plausible sentences:
- Target: "I crack the eggs into the **pan**" → Produced: "I crack the eggs into the **bowl**" — "bowl" is a completely reasonable word here
- Target: "she sat in the **chair**" → Produced: "she sat in the **table**" — this is more detectable, "sat in the table" is odd
- Target: "the **cat** climbed the tree" → Produced: "the **dog** climbed the tree" — "dog" is equally plausible

Some cases may be detectable from context (semantic anomalies, selectional restriction violations), others are fundamentally undetectable without the target text. Before building a pipeline, we need to understand the distribution of what's actually in the data.

### 2.3 Exploration Phase (Do This First)

Before writing any prompts or evaluation code, manually inspect the semantic paraphasias in the Fridriksson dataset to understand what we're dealing with.

**Step 1: Extract all semantic paraphasias from the Fridriksson CHAT files.**

Write a script that parses the `.cha` files and extracts every instance tagged `[* s]`, along with:
- The produced word
- The intended target word (from the `[: target]` annotation)
- The full utterance context (the surrounding sentence)
- The target script sentence for reference (for our analysis only, NOT for the model)

**Step 2: Categorize them by detectability.** Manually (or with LLM assistance) sort each instance into rough categories:

- **Likely detectable without target:** The produced word creates a semantic anomaly, selectional restriction violation, or contextual incoherence (e.g., "sat in the table," "drinking a fork")
- **Possibly detectable without target:** The produced word is somewhat odd in context but not impossible (e.g., wrong pronoun gender when referent is clear, a word that contradicts something said earlier in the passage)
- **Unlikely detectable without target:** The produced word is a perfectly plausible substitution that makes a grammatically and semantically coherent sentence (e.g., "bowl" for "pan," "dog" for "cat")

**Step 3: Quantify the distribution.** What percentage of semantic paraphasias in Fridriksson fall into each bucket? This tells us the ceiling for context-only detection. If 80% are in the "unlikely detectable" bucket, then this stage has limited value and we should document that finding. If there's a meaningful fraction that are detectable, proceed to implementation.

**Step 4: Generate a report** with ~10 concrete examples from each category, annotated with reasoning for why each is/isn't detectable. This report is a key deliverable — it informs whether Stage 2 is worth pursuing and sets expectations.

### 2.4 Implementation (Conditional on Exploration Results)

Only proceed if the exploration phase shows a meaningful fraction of detectable cases.

Use a strong instruction-following LLM (Claude Sonnet, GPT-4o, or an open model like Llama 3.1 70B). This is a prompt-engineering task, not a fine-tuning task.

**Prompt template:**
```
You are a speech-language pathology assistant analyzing a transcript from a patient
with aphasia.

Here is the transcript of the patient speaking:
"{asr_transcript}"

Some words are already flagged as speech errors:
[p] = phonemic paraphasia (sound-level error), [n] = neologistic paraphasia (nonsense word).

Your task: identify any remaining words that appear to be SEMANTIC PARAPHASIAS — real,
correctly-pronounced words that seem semantically out of place in context. These are
words where the patient likely intended a different but related word.

Look for:
- Words that violate selectional restrictions (e.g., "drinking a fork")
- Words that contradict the surrounding context or topic
- Category substitutions that create odd but grammatical sentences

Be conservative. Many patients with aphasia produce unusual but intentional word choices.
Only flag words where the surrounding context strongly suggests a substitution occurred.

Output format (JSON):
[{"position": <word_index>, "produced": "<word>", "likely_target": "<best_guess>",
  "reasoning": "<why this seems like a substitution>"}]

If no semantic paraphasias are detected, output an empty list: []
```

**Prompt variations to test:**
- With vs. without the task description (e.g., "The patient was describing a picture of a kitchen scene")
- With vs. without examples of semantic paraphasias in the prompt
- Different LLMs (Claude Sonnet, GPT-4o, Llama 3.1 70B) to see if capability varies
- Providing broader transcript context (full passage vs. single utterance)

### 2.5 Evaluation

**Test set:** Same Fridriksson cross-validation folds as CHAI.

**Metrics (for direct comparison to CHAI):**
- Utterance-level binary F1 for semantic paraphasias (CHAI's metric)
- TD-[s] (temporal distance for semantic paraphasias)
- Word-level precision, recall, F1 for semantic paraphasias

**Ablations:**
- Stage 1 alone (acoustic model handles all three types) — direct comparison to CHAI single-seq
- Stage 2 alone on oracle transcripts — upper bound for LLM semantic detection from text
- Stage 2 on Stage 1 ASR output — realistic pipeline performance
- Stage 1 (for [p], [n]) + Stage 2 (for [s]) combined — full system, recompute all CHAI metrics

**Key question to answer:** Does the LLM stage improve TD-[s] and F1-[s] over Stage 1 alone? Even a modest improvement over the ~0.12 F1 baseline is meaningful given how low the current numbers are.

---

## Experiment Schedule

### Phase 0: Repo Setup and Data Prep
- [ ] Clone CHAI repo, understand their data prep scripts and split definitions
- [ ] Obtain AphasiaBank membership and download Protocol + Fridriksson data
- [ ] Run CHAI preprocessing pipeline, verify output format matches paper examples
- [ ] Verify you can reproduce CHAI's data splits exactly
- [ ] Set up evaluation scripts implementing all metrics from Section 1.4

### Phase 1: Reproduce CHAI Baseline
- [ ] Attempt to reproduce CHAI single-seq results using their code and HuBERT
- [ ] Verify evaluation pipeline produces numbers in the ballpark of their Table 2
- [ ] This is a sanity check — if numbers don't match, debug before proceeding

### Phase 2: Whisper Single-Seq
- [ ] Implement Whisper single-seq model (extend tokenizer, training loop)
- [ ] Train on Protocol, fine-tune on Fridriksson with same splits as CHAI
- [ ] Evaluate with full metric suite, compare to CHAI Table 2
- [ ] Hyperparameter sweep (learning rate, frozen vs. unfrozen encoder)
- [ ] Try Whisper-small and Whisper-medium, compare

### Phase 3: SONIVA Encoder Pretraining (Optional)
- [ ] Obtain SONIVA access
- [ ] Fine-tune Whisper encoder on SONIVA for ASR
- [ ] Use SONIVA-adapted checkpoint as initialization for Phase 2
- [ ] Evaluate whether SONIVA pretraining improves paraphasia detection metrics

### Phase 4: Semantic Paraphasia Exploration
- [ ] Extract all `[* s]` instances from Fridriksson with full context (produced word, target word, utterance, script)
- [ ] Categorize each by detectability: likely / possibly / unlikely detectable without target text
- [ ] Quantify distribution across categories
- [ ] Produce report with ~10 annotated examples per category
- [ ] **Decision gate:** If <20% of cases are in "likely detectable" category, document finding and deprioritize Stage 2. If >=20%, proceed to Phase 5.

### Phase 5: LLM Semantic Paraphasia Detection (Conditional)
- [ ] Implement prompt templates (no target script — transcript only)
- [ ] Test on oracle transcripts first (isolate LLM capability from ASR errors)
- [ ] Test prompt variations (with/without task description, with/without examples)
- [ ] Test across LLMs (Claude Sonnet, GPT-4o, Llama 3.1 70B)
- [ ] Run on Stage 1 ASR output for realistic pipeline numbers
- [ ] Merge Stage 1 + Stage 2 outputs, compute full CHAI metric suite
- [ ] Evaluate: does the combined system improve TD-[s] and F1-[s] over Stage 1 alone?

### Phase 6: Analysis and Ablations
- [ ] Statistical significance tests (bootstrap for WER/AWER, ANOVA for TD)
- [ ] Error analysis by aphasia severity (mild/moderate/severe/very severe)
- [ ] Failure mode analysis: what types of errors does the model get wrong?
- [ ] Compare general ASR performance on healthy speech (LibriSpeech test-clean) to verify no catastrophic degradation
- [ ] If Stage 2 was attempted: analyze which detectability categories the LLM succeeded/failed on

---

## Key Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Can't reproduce CHAI's exact splits | Their repo has split definitions; contact authors if unclear |
| Whisper's decoder over-corrects non-words toward real words | Monitor neologism detection specifically; consider decoding with reduced language model weight or temperature; SpecAugment helps |
| SONIVA access denied or delayed | SONIVA is optional; core plan works with AphasiaBank alone |
| Most semantic paraphasias are undetectable without target text | Phase 4 exploration will quantify this early; if true, document as a finding and focus effort on Stage 1 |
| LLM over-flags unusual but correct word choices as semantic paraphasias | Precision will suffer; tune prompt for conservatism; report precision/recall tradeoff |
| Small Fridriksson dataset leads to high variance | Use same cross-validation as CHAI; report confidence intervals |
