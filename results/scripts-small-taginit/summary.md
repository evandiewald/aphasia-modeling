# scripts-small-taginit — pooled 12-fold LOSO (Scripts-Fridriksson, 985 utts)

whisper-small, tag_classes=pns, tag_init=words, tag_lr_scale=10, fp16, eff. batch 16, greedy decode + NoTagAfterTag.
**TD values below are from before the TD fix** (paraphasia-free utterances were skipped instead of scored 0, inflating TD vs. CHAI); re-score with `scripts/score_predictions.py`. WER/AWER/F1 are unaffected.
Full metrics.json / predictions.json live on the Kaggle kernel at /kaggle/working/results/scripts-small-taginit.

| | WER | AWER | TD-bin | TD-[p] | TD-[n] | TD-[s] | TD-all | F1-[p] | F1-[n] | F1-[s] |
|---|---|---|---|---|---|---|---|---|---|---|
| pooled | 40.7 | 43.6 | 0.88 | 1.11 | 1.08 | 1.37 | 3.56 | 0.70 | 0.56 | 0.33 |

Tag counts hyp/ref: [p] 1013/1106, [n] 346/601, [s] 314/275

| spk | n | WER | TD-bin | TD-all | F1 p/n/s |
|---|---|---|---|---|---|
| P1 | 91 | 64.6 | 0.99 | 4.18 | 0.76/0.53/0.38 |
| P2 | 107 | 49.0 | 1.20 | 3.51 | 0.58/0.39/0.24 |
| P3 | 109 | 25.3 | 0.80 | 2.86 | 0.66/0.58/0.14 |
| P4 | 89 | 43.3 | 0.79 | 3.17 | 0.76/0.52/0.34 |
| P5 | 66 | 53.6 | 0.82 | 3.28 | 0.78/0.59/0.46 |
| P6 | 37 | 82.1 | 0.73 | 4.18 | 0.50/0.78/0.57 |
| P7 | 106 | 18.2 | 0.75 | 3.07 | 0.65/0.35/0.00 |
| P8 | 105 | 28.6 | 0.89 | 3.29 | 0.75/0.45/0.11 |
| P10 | 92 | 44.3 | 0.87 | 3.65 | 0.72/0.45/0.08 |
| P11 | 6 | 71.8 | 0.42 | 2.01 | 0.50/1.00/0.00 |
| P12 | 59 | 76.2 | 0.80 | 3.95 | 0.78/0.72/0.51 |
| P13 | 118 | 24.7 | 0.86 | 3.22 | 0.67/0.62/0.12 |
