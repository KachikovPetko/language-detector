# ML Model — Training & Export

## Overview

Character-level TF-IDF + Logistic Regression pipeline that detects language from
raw text. Trained on WiLI-2018 (20 languages × 1 000 samples).

## Setup

```bash
cd ml
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

## Train

```bash
python train.py
```

- Downloads WiLI-2018 from Zenodo (~30 MB zip). Falls back to Wikipedia API if unavailable.
- Trains TF-IDF (`char_wb`, ngram 2–4, max 80 000 features) + LogReg (lbfgs, C=5).
- Saves `../public/model/detector.pkl` and `../public/model/classes.json`.
- Saves `results/confusion_matrix.png` and `results/metrics.md`.

Typical accuracy on WiLI test set: **~97–99%**.

## Export to ONNX

```bash
python export_onnx.py
```

- Converts the pipeline to ONNX using skl2onnx.
- Verifies that ONNX predictions match sklearn predictions on 7 test phrases.
- Saves `../public/model/detector.onnx` and `../public/model/labels.json`.

## Retrain from scratch

Just delete `../public/model/detector.pkl` and re-run both scripts in order.
All evaluation artefacts in `results/` are overwritten automatically.

## Model card

| Property | Value |
|---|---|
| Architecture | TF-IDF + LogisticRegression |
| Feature type | Character n-grams (2–4), `char_wb` analyser |
| Max vocab | 80 000 features |
| Solver | lbfgs, max_iter=1000, C=5 |
| Dataset | WiLI-2018 (Thoma, 2018) |
| Languages | 20 (see `lib/languages.ts`) |
| Export format | ONNX opset 17 |

## Why char n-grams?

Character-level features work well for language ID because:
1. They capture morphological patterns (suffixes, prefixes) without tokenisation.
2. They handle mixed-script text and OOV words gracefully.
3. The `char_wb` analyser adds word-boundary padding, improving word-edge patterns.
4. Training is fast and the model is tiny compared to neural alternatives.
