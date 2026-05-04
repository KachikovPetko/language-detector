#!/usr/bin/env python3
"""
Export all three model weights as raw binary files for TypeScript inference.

Reads:   ../public/model/detector.pkl   (dict: tfidf, count_vec, logreg, svc, nb)
Writes:
  ../public/model/vocab.json            — {vocabulary: {ngram→index}, idf: [...]}
  ../public/model/labels.json           — {classes: [...]}
  ../public/model/coef.bin              — LogReg coef  float32[n_classes × n_features]
  ../public/model/intercept.bin         — LogReg intercept  float32[n_classes]
  ../public/model/svc_coef.bin          — LinearSVC coef  float32[n_classes × n_features]
  ../public/model/svc_intercept.bin     — LinearSVC intercept  float32[n_classes]
  ../public/model/nb_log_prob.bin       — NB feature_log_prob_  float32[n_classes × n_features]
  ../public/model/nb_class_log_prior.bin — NB class_log_prior_  float32[n_classes]
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).parent.parent / "public" / "model"
PKL_PATH  = MODEL_DIR / "detector.pkl"

if not PKL_PATH.exists():
    print(f"ERROR: {PKL_PATH} not found. Run train.py first.", file=sys.stderr)
    sys.exit(1)

with open(PKL_PATH, "rb") as f:
    pipeline = pickle.load(f)

# Support both old (sklearn Pipeline) and new (dict) format
if isinstance(pipeline, dict):
    tfidf = pipeline["tfidf"]
    lr    = pipeline["logreg"]
    svc   = pipeline["svc"]
    nb    = pipeline["nb"]
else:
    print("ERROR: detector.pkl is in the old Pipeline format. Re-run train.py first.", file=sys.stderr)
    sys.exit(1)

# ---- vocab.json ----------------------------------------------------------------
vocab_path = MODEL_DIR / "vocab.json"
vocab_data = {
    "vocabulary": {k: int(v) for k, v in tfidf.vocabulary_.items()},
    "idf": tfidf.idf_.tolist(),
}
vocab_path.write_text(json.dumps(vocab_data, ensure_ascii=False), encoding="utf-8")
print(f"vocab.json     {len(tfidf.vocabulary_)} n-grams  -> {vocab_path}")

# ---- labels.json ---------------------------------------------------------------
labels_path = MODEL_DIR / "labels.json"
classes = list(lr.classes_)
labels_path.write_text(json.dumps({"classes": classes}, ensure_ascii=False, indent=2))
print(f"labels.json    {len(classes)} classes  -> {labels_path}")

# ---- LogReg --------------------------------------------------------------------
def _save_bin(arr: np.ndarray, path: Path, label: str):
    arr = arr.astype(np.float32)
    arr.tofile(path)
    kb = arr.nbytes / 1024
    print(f"{label:<30} {str(arr.shape):<22} {kb:>8.0f} KB  -> {path.name}")

_save_bin(lr.coef_,       MODEL_DIR / "coef.bin",      "coef.bin (LogReg)")
_save_bin(lr.intercept_,  MODEL_DIR / "intercept.bin", "intercept.bin (LogReg)")

# ---- LinearSVC -----------------------------------------------------------------
_save_bin(svc.coef_,      MODEL_DIR / "svc_coef.bin",      "svc_coef.bin")
_save_bin(svc.intercept_, MODEL_DIR / "svc_intercept.bin", "svc_intercept.bin")

# ---- MultinomialNB -------------------------------------------------------------
_save_bin(nb.feature_log_prob_, MODEL_DIR / "nb_log_prob.bin",          "nb_log_prob.bin")
_save_bin(nb.class_log_prior_,  MODEL_DIR / "nb_class_log_prior.bin",   "nb_class_log_prior.bin")

print("\nAll weights exported. The TypeScript inference engine is ready.")
