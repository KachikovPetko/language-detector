#!/usr/bin/env python3
"""
Train three language-detection models on the same TF-IDF features:
  1. Logistic Regression  (primary — best accuracy, used for translation)
  2. Linear SVC           (discriminative, fast, no probability output)
  3. Multinomial NB       (generative baseline, trained on raw counts)

Dataset strategy (tries in order):
  1. WiLI-2018 from Zenodo
  2. Wikipedia API fallback

Outputs:
  ../public/model/detector.pkl     — dict of vectoriser + all 3 classifiers
  ../public/model/classes.json     — class order
  results/confusion_matrix.png     — LogReg confusion matrix
  results/metrics.md               — all 3 models' accuracy + LogReg report
"""

import os, sys, io, json, zipfile, pickle, tempfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import requests
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
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

TARGET_CODES = {
    "ara", "bul", "ces", "cmn", "deu",
    "ell", "eng", "fra", "hin", "ita",
    "jpn", "kor", "nld", "pol", "por",
    "ron", "rus", "spa", "tur", "ukr",
}

WILI_ALIASES = { "zho": "cmn", "arb": "ara" }
SAMPLES_PER_LANG = 1000


# ---------------------------------------------------------------------------
# Data loading  (unchanged from original)
# ---------------------------------------------------------------------------

def _apply_aliases(labels):
    return [WILI_ALIASES.get(l, l) for l in labels]


def try_wili():
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
                    buf.write(chunk); bar.update(len(chunk))
            buf.seek(0)
            with zipfile.ZipFile(buf) as zf:
                names = zf.namelist()
                def _read(pattern):
                    candidates = [n for n in names if n.endswith(pattern)]
                    if not candidates: raise FileNotFoundError(f"{pattern} not in zip")
                    return zf.read(candidates[0]).decode("utf-8", errors="replace").splitlines()
                x_train_raw = _read("x_train.txt")
                y_train_raw = _apply_aliases(_read("y_train.txt"))
                x_test_raw  = _read("x_test.txt")
                y_test_raw  = _apply_aliases(_read("y_test.txt"))
            def _filter(xs, ys):
                pairs = [(x, y) for x, y in zip(xs, ys) if y in TARGET_CODES]
                if not pairs: return [], []
                x_f, y_f = zip(*pairs)
                return list(x_f), list(y_f)
            x_tr, y_tr = _filter(x_train_raw, y_train_raw)
            x_te, y_te = _filter(x_test_raw,  y_test_raw)
            missing = TARGET_CODES - set(y_tr)
            if missing:
                print(f"  ⚠ WiLI missing codes: {missing}")
            else:
                print(f"  ✓ All 20 languages ({len(x_tr)} train, {len(x_te)} test)")
            return x_tr, y_tr, x_te, y_te
        except Exception as exc:
            print(f"  ✗ Failed ({exc}), trying next…")
    return None


WIKI_LANG_CODES = {
    "ara":"ar","bul":"bg","ces":"cs","cmn":"zh","deu":"de","ell":"el","eng":"en",
    "fra":"fr","hin":"hi","ita":"it","jpn":"ja","kor":"ko","nld":"nl","pol":"pl",
    "por":"pt","ron":"ro","rus":"ru","spa":"es","tur":"tr","ukr":"uk",
}

def fetch_wikipedia_samples(iso3_code, n=SAMPLES_PER_LANG):
    wiki_code = WIKI_LANG_CODES[iso3_code]
    base = f"https://{wiki_code}.wikipedia.org/w/api.php"
    samples, seen_ids = [], set()
    def _fetch_random(count=20):
        try:
            resp = requests.get(base, params={"action":"query","list":"random","rnnamespace":0,"rnlimit":count,"format":"json"}, timeout=15)
            resp.raise_for_status()
            page_ids = [p["id"] for p in resp.json()["query"]["random"] if p["id"] not in seen_ids]
            seen_ids.update(page_ids)
            if not page_ids: return
            eresp = requests.get(base, params={"action":"query","pageids":"|".join(str(p) for p in page_ids),"prop":"extracts","explaintext":True,"exintro":True,"exsentences":5,"format":"json"}, timeout=15)
            eresp.raise_for_status()
            for page in eresp.json()["query"]["pages"].values():
                text = page.get("extract","").strip()
                if len(text) > 80: samples.append(text)
        except Exception: pass
    attempts = 0
    while len(samples) < n and attempts < 20:
        _fetch_random(min(50, n - len(samples) + 10)); attempts += 1
    return samples[:n]


def build_wikipedia_dataset(codes, n_per_lang=SAMPLES_PER_LANG):
    texts, labels = [], []
    for code in sorted(codes):
        print(f"  Fetching Wikipedia ({code})…")
        samps = fetch_wikipedia_samples(code, n_per_lang)
        texts.extend(samps); labels.extend([code]*len(samps))
        print(f"    → {len(samps)} samples")
    return texts, labels


def load_dataset():
    result = try_wili()
    if result is not None:
        x_tr, y_tr, x_te, y_te = result
        missing = TARGET_CODES - set(y_tr)
        if missing:
            print(f"Supplementing {len(missing)} missing languages from Wikipedia…")
            wx, wy = build_wikipedia_dataset(missing)
            split = int(len(wx)*0.8)
            x_tr += wx[:split]; y_tr += wy[:split]
            x_te += wx[split:]; y_te += wy[split:]
        return x_tr, y_tr, x_te, y_te
    print("WiLI unavailable — falling back to Wikipedia…")
    wx, wy = build_wikipedia_dataset(TARGET_CODES)
    split = int(len(wx)*0.8)
    return wx[:split], wy[:split], wx[split:], wy[split:]


# ---------------------------------------------------------------------------
# Training — all three models
# ---------------------------------------------------------------------------

def train_all(x_train, y_train):
    print(f"\nFitting TF-IDF vectoriser on {len(x_train)} samples…")
    tfidf = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2,4),
        max_features=30_000, sublinear_tf=True, min_df=2,
    )
    X_train_tfidf = tfidf.fit_transform(x_train)
    print(f"  Vocabulary: {len(tfidf.vocabulary_)} n-grams, matrix {X_train_tfidf.shape}")

    # CountVectorizer with the same vocabulary — needed for MultinomialNB (raw counts)
    count_vec = CountVectorizer(
        vocabulary=tfidf.vocabulary_,
        analyzer="char_wb", ngram_range=(2,4),
    )
    X_train_counts = count_vec.transform(x_train)

    print("\nTraining Logistic Regression…")
    lr = LogisticRegression(solver="lbfgs", C=5.0, max_iter=1000, n_jobs=-1)
    lr.fit(X_train_tfidf, y_train)

    print("Training Linear SVC…")
    svc = LinearSVC(C=1.0, max_iter=2000, dual=True)
    svc.fit(X_train_tfidf, y_train)

    print("Training Multinomial Naive Bayes…")
    nb = MultinomialNB(alpha=0.1)
    nb.fit(X_train_counts, y_train)

    return tfidf, count_vec, lr, svc, nb


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_all(tfidf, count_vec, lr, svc, nb, x_test, y_test):
    print("\nEvaluating all three models…")

    X_test_tfidf  = tfidf.transform(x_test)
    X_test_counts = count_vec.transform(x_test)

    acc_lr  = accuracy_score(y_test, lr.predict(X_test_tfidf))
    acc_svc = accuracy_score(y_test, svc.predict(X_test_tfidf))
    acc_nb  = accuracy_score(y_test, nb.predict(X_test_counts))

    print(f"\n  {'Model':<28} {'Accuracy':>10}")
    print(f"  {'-'*40}")
    print(f"  {'Logistic Regression (primary)':<28} {acc_lr*100:>9.2f}%")
    print(f"  {'Linear SVC':<28} {acc_svc*100:>9.2f}%")
    print(f"  {'Multinomial Naive Bayes':<28} {acc_nb*100:>9.2f}%")

    labels = sorted(set(y_test))
    report = classification_report(y_test, lr.predict(X_test_tfidf), labels=labels, zero_division=0)

    # Save markdown report
    md  = "# Language Detection — Model Comparison\n\n"
    md += f"| Model | Test Accuracy |\n|---|---|\n"
    md += f"| Logistic Regression *(primary)* | **{acc_lr*100:.2f}%** |\n"
    md += f"| Linear SVC | {acc_svc*100:.2f}% |\n"
    md += f"| Multinomial Naive Bayes | {acc_nb*100:.2f}% |\n\n"
    md += f"## LogReg — Per-class Report\n\n```\n{report}\n```\n"
    (RESULTS_DIR / "metrics.md").write_text(md, encoding="utf-8")
    print(f"  → Saved metrics to {RESULTS_DIR / 'metrics.md'}")

    # Confusion matrix (LogReg only)
    cm = confusion_matrix(y_test, lr.predict(X_test_tfidf), labels=labels)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
                xticklabels=labels, yticklabels=labels,
                linewidths=0.3, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix — LogReg (primary model)", fontsize=14, pad=15)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved confusion matrix to {RESULTS_DIR / 'confusion_matrix.png'}")

    return {"logistic_regression": round(acc_lr, 6), "linear_svc": round(acc_svc, 6), "naive_bayes": round(acc_nb, 6)}


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save(tfidf, count_vec, lr, svc, nb, accuracies):
    classes = list(lr.classes_)

    payload = {
        "tfidf":     tfidf,
        "count_vec": count_vec,
        "logreg":    lr,
        "svc":       svc,
        "nb":        nb,
    }
    pkl_path = MODEL_DIR / "detector.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"\nSaved all models → {pkl_path}")

    (MODEL_DIR / "classes.json").write_text(
        json.dumps({"classes": classes}, ensure_ascii=False, indent=2))
    print(f"Saved class list → {MODEL_DIR / 'classes.json'}")

    (MODEL_DIR / "model_accuracy.json").write_text(
        json.dumps(accuracies, indent=2))
    print(f"Saved accuracies → {MODEL_DIR / 'model_accuracy.json'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("LinguaLens — Multi-Model Language Detection Training")
    print("=" * 60)

    x_train, y_train, x_test, y_test = load_dataset()
    if not x_train:
        print("ERROR: no training data.", file=sys.stderr); sys.exit(1)

    tfidf, count_vec, lr, svc, nb = train_all(x_train, y_train)
    accuracies = evaluate_all(tfidf, count_vec, lr, svc, nb, x_test, y_test)
    save(tfidf, count_vec, lr, svc, nb, accuracies)

    print("\nDone! Run export_weights.py next.")
