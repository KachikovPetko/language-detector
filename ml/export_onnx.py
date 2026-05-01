#!/usr/bin/env python3
"""
Export the trained model for use by the web app.

skl2onnx does not support char-level TF-IDF analyzers (known upstream limitation;
see github.com/onnx/sklearn-onnx/issues). Workaround: export the pipeline in two parts:

  1. vocab.json        — TF-IDF vocabulary {ngram: feature_index} + IDF weights
  2. detector.onnx     — LogisticRegression with float32 feature input
  3. labels.json       — ordered class list (matches probability vector order)

The TypeScript side (lib/detector.ts) implements the char_wb TF-IDF transform,
then passes the float32 feature vector to the ONNX classifier.

Reads:   ../public/model/detector.pkl
Writes:  ../public/model/vocab.json
         ../public/model/detector.onnx
         ../public/model/labels.json
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import onnxruntime as rt
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

MODEL_DIR   = Path(__file__).parent.parent / "public" / "model"
PKL_PATH    = MODEL_DIR / "detector.pkl"
ONNX_PATH   = MODEL_DIR / "detector.onnx"
VOCAB_PATH  = MODEL_DIR / "vocab.json"
LABELS_PATH = MODEL_DIR / "labels.json"

# Same texts as used during training evaluation
VERIFY_TEXTS = [
    "The quick brown fox jumps over the lazy dog",
    "Bonjour, comment allez-vous aujourd'hui?",
    "Das ist ein sehr schöner Tag für alle Menschen",
    "Привет, как дела? Я очень рад тебя видеть",
    "日本語のテキストを検出するためのテスト文章です",
    "مرحبا، كيف حالك اليوم؟ أنا بخير شكراً",
    "नमस्ते, आज का दिन बहुत अच्छा है",
]


def load_pipeline():
    if not PKL_PATH.exists():
        print(f"ERROR: {PKL_PATH} not found. Run train.py first.", file=sys.stderr)
        sys.exit(1)
    with open(PKL_PATH, "rb") as f:
        return pickle.load(f)


def save_vocab_and_idf(pipeline) -> None:
    tfidf = pipeline.named_steps["tfidf"]
    vocab: dict[str, int] = {k: int(v) for k, v in tfidf.vocabulary_.items()}
    idf: list[float] = tfidf.idf_.tolist()

    payload = {"vocabulary": vocab, "idf": idf}
    VOCAB_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    size_kb = VOCAB_PATH.stat().st_size / 1024
    print(f"Saved vocab.json  ({len(vocab)} ngrams, {size_kb:.0f} KB) -> {VOCAB_PATH}")


def convert_classifier(pipeline) -> bytes:
    """Export only the LogisticRegression with float32 input."""
    lr = pipeline.named_steps["clf"]
    n_features = len(pipeline.named_steps["tfidf"].vocabulary_)
    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(
        lr,
        initial_types=initial_type,
        options={id(lr): {"zipmap": False}},
        target_opset=17,
    )
    return onnx_model.SerializeToString()


def _tfidf_transform(text: str, vocabulary: dict, idf: np.ndarray, n_features: int) -> np.ndarray:
    """Replicate sklearn's char_wb TF-IDF transform for verification."""
    text_lower = text.lower()
    tokens = text_lower.split()
    counts = np.zeros(n_features, dtype=np.float32)

    for token in tokens:
        padded = f" {token} "
        for n in range(2, 5):  # ngram_range=(2,4)
            for i in range(len(padded) - n + 1):
                ngram = padded[i : i + n]
                idx = vocabulary.get(ngram)
                if idx is not None:
                    counts[idx] += 1

    # sublinear_tf=True
    mask = counts > 0
    counts[mask] = 1 + np.log(counts[mask])
    features = counts * idf

    # L2 normalise
    norm = np.linalg.norm(features)
    if norm > 0:
        features /= norm
    return features


def verify(pipeline, onnx_bytes: bytes) -> None:
    print("\nVerifying ONNX predictions match sklearn pipeline…")
    tfidf = pipeline.named_steps["tfidf"]
    vocabulary: dict = tfidf.vocabulary_
    idf = np.array(tfidf.idf_, dtype=np.float32)
    n_features = len(vocabulary)

    sess = rt.InferenceSession(onnx_bytes)
    input_name = sess.get_inputs()[0].name

    sklearn_preds = pipeline.predict(VERIFY_TEXTS)
    sklearn_probs = pipeline.predict_proba(VERIFY_TEXTS)

    all_ok = True
    for i, text in enumerate(VERIFY_TEXTS):
        feat = _tfidf_transform(text, vocabulary, idf, n_features).reshape(1, -1)
        label_out, prob_out = sess.run(None, {input_name: feat})
        onnx_label = label_out[0]
        onnx_probs = prob_out[0]

        label_match = onnx_label == sklearn_preds[i]
        prob_close  = np.allclose(onnx_probs, sklearn_probs[i], rtol=1e-2, atol=1e-3)

        status = "[OK]" if (label_match and prob_close) else "[FAIL]"
        print(f"  {status} [{sklearn_preds[i]}] \"{text[:50]}\"")

        if not label_match:
            print(f"    Label mismatch: sklearn={sklearn_preds[i]!r}  onnx={onnx_label!r}")
            all_ok = False
        if not prob_close:
            max_diff = float(np.max(np.abs(onnx_probs - sklearn_probs[i])))
            print(f"    Max prob diff: {max_diff:.6f}")
            if max_diff > 0.05:   # tolerate small float precision differences
                all_ok = False

    if all_ok:
        print("\n[PASS] All predictions verified — ONNX + vocab.json is correct.")
    else:
        print("\n[FAIL] Prediction mismatch detected.", file=sys.stderr)
        sys.exit(1)


def save_labels(pipeline) -> None:
    classes: list[str] = list(pipeline.named_steps["clf"].classes_)
    LABELS_PATH.write_text(
        json.dumps({"classes": classes}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved labels.json ({len(classes)} classes) -> {LABELS_PATH}")


if __name__ == "__main__":
    print("=" * 60)
    print("LinguaLens — ONNX Export & Verification")
    print("=" * 60)
    print("Note: skl2onnx does not support char-level TF-IDF analyzers.")
    print("Exporting TF-IDF as vocab.json, LogisticRegression as ONNX.\n")

    pipeline = load_pipeline()
    save_vocab_and_idf(pipeline)

    onnx_bytes = convert_classifier(pipeline)
    ONNX_PATH.write_bytes(onnx_bytes)
    size_kb = len(onnx_bytes) / 1024
    print(f"Saved detector.onnx ({size_kb:.0f} KB) -> {ONNX_PATH}")

    verify(pipeline, onnx_bytes)
    save_labels(pipeline)

    total_kb = sum(
        p.stat().st_size for p in [VOCAB_PATH, ONNX_PATH, LABELS_PATH]
    ) / 1024
    print(f"\nTotal model size: {total_kb:.0f} KB ({total_kb/1024:.1f} MB)")
    print("All done! The web app can now load vocab.json + detector.onnx.")
