# ML Model — Training & Export

## Overview

Character-level TF-IDF + Logistic Regression pipeline trained on WiLI-2018 to
identify language from raw text. Reaches **98.86% accuracy** on 20 languages.

---

## Setup

```bash
cd ml
python -m venv venv

# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

---

## Train

```bash
python -X utf8 train.py
```

- Downloads WiLI-2018 from Zenodo (~62 MB zip). Falls back to Wikipedia API samples if unavailable.
- Trains `TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4), max_features=30000, sublinear_tf=True)` + `LogisticRegression(solver='lbfgs', C=5, max_iter=1000)`.
- Saves confusion matrix → `results/confusion_matrix.png` and per-class metrics → `results/metrics.md`.

---

## Export weights (used by the web app)

```bash
python export_weights.py
```

Exports files consumed by `lib/detector.ts`:

| File | Size | Contents |
|---|---|---|
| `../public/model/vocab.json` | 921 KB | `{vocabulary: {ngram→index}, idf: [...]}` |
| `../public/model/labels.json` | 1 KB | Class order (ISO 639-3 codes) |
| `../public/model/coef.bin` | 2.3 MB | LogReg coef — 20 × 30 000 float32 |
| `../public/model/intercept.bin` | 80 B | LogReg intercept — 20 float32 |
| `../public/model/svc_coef.bin` | 2.3 MB | LinearSVC coef — 20 × 30 000 float32 |
| `../public/model/svc_intercept.bin` | 80 B | LinearSVC intercept — 20 float32 |
| `../public/model/nb_log_prob.bin` | 2.3 MB | NB log P(feature\|class) — 20 × 30 000 float32 |
| `../public/model/nb_class_log_prior.bin` | 80 B | NB log P(class) — 20 float32 |

> **Why not ONNX at runtime?** skl2onnx does not support `analyzer='char_wb'`, so the vectorizer cannot be exported as a single ONNX graph. Additionally, `onnxruntime-node` uses native binaries that fail on Vercel's serverless Lambda. The pure-TypeScript inference in `lib/detector.ts` is a faithful reimplementation: same char_wb tokenisation, sublinear TF, smooth IDF, L2 norm, and softmax.

---

## Optional: export ONNX (LogReg only, for reference)

```bash
python export_onnx.py
```

Exports the LogReg layer as `../public/model/detector.onnx`. The TF-IDF step must still be run in TypeScript. This file is not used by the production app.

---

## Retrain from scratch

Delete `../public/model/*.bin` and `../public/model/vocab.json`, then re-run:

```bash
python -X utf8 train.py
python export_weights.py
```

---

## Model card

| Property | Value |
|---|---|
| Architecture | TF-IDF + LogisticRegression |
| Feature type | Character n-grams (2–4), `char_wb` analyser |
| Vocabulary size | 30 000 most-frequent n-grams |
| Sublinear TF | Yes (`1 + log(tf)`) |
| IDF smoothing | Yes (sklearn default) |
| L2 norm | Yes (per-document) |
| Solver | lbfgs, max_iter=1000, C=5 |
| Dataset | WiLI-2018 (Thoma, 2018) |
| Train samples | 20 000 (1 000 × 20 languages) |
| Test samples | 6 660 (333 × 20 languages) |
| Test accuracy | **98.86%** |

---

## Why char n-grams?

Character n-grams with `char_wb` (word-boundary padding) work well for
language identification because:

1. They capture morphological patterns (suffixes, prefixes) without a tokeniser.
2. They handle mixed-script text and OOV words gracefully — a word unseen at training still produces known sub-character sequences.
3. Word-boundary padding (`" word "`) improves word-start and word-end n-grams, which are especially distinctive across languages.
4. Training is fast (seconds on CPU), and the final model is tiny (~2.3 MB weights) compared to any neural alternative.
5. Logistic Regression with lbfgs generalises well in the high-dimensional sparse feature space that TF-IDF produces.
