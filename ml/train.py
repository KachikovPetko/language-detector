#!/usr/bin/env python3
"""
Train a language-detection model using TF-IDF character n-grams + Logistic Regression.

Dataset strategy (tries in order):
  1. WiLI-2018 from Zenodo — 235 languages, 1 000 train samples each
  2. Wikipedia API fallback — fetches article summaries for each language

Outputs (all relative to this script):
  ../public/model/detector.pkl   — sklearn Pipeline (for export_onnx.py)
  results/confusion_matrix.png   — labelled heatmap
  results/metrics.md             — classification report
"""

import os
import sys
import io
import json
import zipfile
import pickle
import tempfile
from pathlib import Path

# Force UTF-8 stdout so Unicode chars don't crash on Windows CP-1252 consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCRIPT_DIR  = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
MODEL_DIR   = SCRIPT_DIR.parent / "public" / "model"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# WiLI ISO 639-3 codes for our 20 target languages
TARGET_CODES = {
    "ara", "bul", "ces", "cmn", "deu",
    "ell", "eng", "fra", "hin", "ita",
    "jpn", "kor", "nld", "pol", "por",
    "ron", "rus", "spa", "tur", "ukr",
}

# WiLI sometimes labels Arabic as 'ara', Chinese as 'cmn'; these are already correct.
# If either is absent we remap the fallback code.
WILI_ALIASES = {
    "zho": "cmn",   # generic Chinese -> Mandarin
    "arb": "ara",   # Standard Arabic  -> Arabic macrolanguage label
}

SAMPLES_PER_LANG = 1000   # WiLI provides exactly this many per language


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _apply_aliases(labels: list[str]) -> list[str]:
    return [WILI_ALIASES.get(l, l) for l in labels]


def try_wili() -> tuple[list[str], list[str], list[str], list[str]] | None:
    """Download WiLI-2018 and return (x_train, y_train, x_test, y_test) filtered to target langs."""
    urls = [
        "https://zenodo.org/records/841984/files/wili-2018.zip",
        "https://zenodo.org/record/841984/files/wili-2018.zip",
    ]
    for url in urls:
        try:
            print(f"Downloading WiLI-2018 from {url} …")
            resp = requests.get(url, stream=True, timeout=120, allow_redirects=True)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            buf = io.BytesIO()
            with tqdm(total=total, unit="B", unit_scale=True, desc="wili-2018.zip") as bar:
                for chunk in resp.iter_content(chunk_size=65536):
                    buf.write(chunk)
                    bar.update(len(chunk))
            buf.seek(0)

            with zipfile.ZipFile(buf) as zf:
                names = zf.namelist()
                def _read(pattern: str) -> list[str]:
                    # accept both flat and inside a directory
                    candidates = [n for n in names if n.endswith(pattern)]
                    if not candidates:
                        raise FileNotFoundError(f"{pattern} not in zip")
                    return zf.read(candidates[0]).decode("utf-8", errors="replace").splitlines()

                x_train_raw = _read("x_train.txt")
                y_train_raw = _apply_aliases(_read("y_train.txt"))
                x_test_raw  = _read("x_test.txt")
                y_test_raw  = _apply_aliases(_read("y_test.txt"))

            # Filter to target languages
            def _filter(xs, ys):
                pairs = [(x, y) for x, y in zip(xs, ys) if y in TARGET_CODES]
                if not pairs:
                    return [], []
                x_f, y_f = zip(*pairs)
                return list(x_f), list(y_f)

            x_tr, y_tr = _filter(x_train_raw, y_train_raw)
            x_te, y_te = _filter(x_test_raw,  y_test_raw)

            found = set(y_tr)
            missing = TARGET_CODES - found
            if missing:
                print(f"  ⚠ WiLI missing codes: {missing} — will supplement with Wikipedia")
            else:
                print(f"  ✓ All 20 languages found in WiLI ({len(x_tr)} train, {len(x_te)} test)")

            return x_tr, y_tr, x_te, y_te

        except Exception as exc:
            print(f"  ✗ Failed ({exc}), trying next URL…")

    return None


WIKI_LANG_CODES = {
    "ara": "ar", "bul": "bg", "ces": "cs", "cmn": "zh",
    "deu": "de", "ell": "el", "eng": "en", "fra": "fr",
    "hin": "hi", "ita": "it", "jpn": "ja", "kor": "ko",
    "nld": "nl", "pol": "pl", "por": "pt", "ron": "ro",
    "rus": "ru", "spa": "es", "tur": "tr", "ukr": "uk",
}

WIKI_SEARCH_TERMS = [
    "science", "history", "mathematics", "geography", "music",
    "literature", "philosophy", "art", "sport", "technology",
    "medicine", "politics", "economics", "culture", "nature",
    "religion", "astronomy", "biology", "chemistry", "physics",
]


def fetch_wikipedia_samples(iso3_code: str, n: int = SAMPLES_PER_LANG) -> list[str]:
    """Fetch random Wikipedia article extracts in the given language."""
    wiki_code = WIKI_LANG_CODES[iso3_code]
    base = f"https://{wiki_code}.wikipedia.org/w/api.php"
    samples: list[str] = []
    seen_ids: set[int] = set()

    def _fetch_random(count: int = 20) -> None:
        params = {
            "action": "query",
            "list": "random",
            "rnnamespace": 0,
            "rnlimit": count,
            "format": "json",
        }
        try:
            resp = requests.get(base, params=params, timeout=15)
            resp.raise_for_status()
            pages = resp.json()["query"]["random"]
            page_ids = [p["id"] for p in pages if p["id"] not in seen_ids]
            seen_ids.update(page_ids)
            if not page_ids:
                return
            # Fetch extracts
            exparams = {
                "action": "query",
                "pageids": "|".join(str(p) for p in page_ids),
                "prop": "extracts",
                "explaintext": True,
                "exintro": True,
                "exsentences": 5,
                "format": "json",
            }
            eresp = requests.get(base, params=exparams, timeout=15)
            eresp.raise_for_status()
            for page in eresp.json()["query"]["pages"].values():
                text = page.get("extract", "").strip()
                if len(text) > 80:
                    samples.append(text)
        except Exception:
            pass

    # Fetch in batches until we have enough (or give up after 20 attempts)
    attempts = 0
    while len(samples) < n and attempts < 20:
        _fetch_random(min(50, n - len(samples) + 10))
        attempts += 1

    return samples[:n]


def build_wikipedia_dataset(
    codes: set[str],
    n_per_lang: int = SAMPLES_PER_LANG,
) -> tuple[list[str], list[str]]:
    texts, labels = [], []
    for code in sorted(codes):
        print(f"  Fetching Wikipedia ({code} / {WIKI_LANG_CODES.get(code, '?')})…")
        samps = fetch_wikipedia_samples(code, n_per_lang)
        texts.extend(samps)
        labels.extend([code] * len(samps))
        print(f"    → {len(samps)} samples")
    return texts, labels


def load_dataset() -> tuple[list[str], list[str], list[str], list[str]]:
    result = try_wili()
    if result is not None:
        x_tr, y_tr, x_te, y_te = result
        present = set(y_tr)
        missing = TARGET_CODES - present
        if missing:
            print(f"Supplementing {len(missing)} missing languages from Wikipedia…")
            wx, wy = build_wikipedia_dataset(missing)
            # 80/20 split for supplemental data
            split = int(len(wx) * 0.8)
            x_tr += wx[:split];  y_tr += wy[:split]
            x_te += wx[split:];  y_te += wy[split:]
        return x_tr, y_tr, x_te, y_te

    print("WiLI unavailable — falling back to Wikipedia for all languages…")
    wx, wy = build_wikipedia_dataset(TARGET_CODES)
    split = int(len(wx) * 0.8)
    return wx[:split], wy[:split], wx[split:], wy[split:]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(x_train: list[str], y_train: list[str]) -> Pipeline:
    print(f"\nTraining on {len(x_train)} samples across {len(set(y_train))} languages…")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=30_000,   # smaller = faster cold-start; still >97% accuracy
            sublinear_tf=True,
            min_df=2,
        )),
        ("clf", LogisticRegression(
            solver="lbfgs",   # lbfgs handles multinomial natively; multi_class removed in sklearn 1.7
            max_iter=1000,
            C=5.0,
            n_jobs=-1,
        )),
    ])
    pipeline.fit(x_train, y_train)
    return pipeline


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(pipeline: Pipeline, x_test: list[str], y_test: list[str]) -> None:
    print("\nEvaluating…")
    y_pred = pipeline.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    labels = sorted(set(y_test))
    report = classification_report(y_test, y_pred, labels=labels, zero_division=0)
    print(report)

    # Save markdown report
    md = f"# Language Detection — Evaluation Results\n\n"
    md += f"**Test samples:** {len(y_test)}  |  **Accuracy:** {acc*100:.2f}%\n\n"
    md += f"## Per-class Report\n\n```\n{report}\n```\n"
    (RESULTS_DIR / "metrics.md").write_text(md, encoding="utf-8")
    print(f"  → Saved metrics to {RESULTS_DIR / 'metrics.md'}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="YlOrRd",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.3, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix — Language Detection", fontsize=14, pad=15)
    plt.tight_layout()
    cm_path = RESULTS_DIR / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved confusion matrix to {cm_path}")


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save(pipeline: Pipeline) -> None:
    pkl_path = MODEL_DIR / "detector.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\nSaved sklearn pipeline → {pkl_path}")

    # Save class order (needed by export_onnx.py to build labels.json)
    classes = list(pipeline.named_steps["clf"].classes_)
    classes_path = MODEL_DIR / "classes.json"
    classes_path.write_text(json.dumps({"classes": classes}, ensure_ascii=False, indent=2))
    print(f"Saved class list      → {classes_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("LinguaLens — Language Detection Model Training")
    print("=" * 60)

    x_train, y_train, x_test, y_test = load_dataset()

    if not x_train:
        print("ERROR: Could not load any training data. Check network connectivity.", file=sys.stderr)
        sys.exit(1)

    pipeline = train(x_train, y_train)
    evaluate(pipeline, x_test, y_test)
    save(pipeline)

    print("\nDone! Run export_onnx.py next to convert to ONNX.")
