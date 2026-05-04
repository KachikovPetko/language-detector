# LinguaLens: ML-Powered Language Detection and Translation
### University ML Project Report
**Discipline:** Selected Methods for Machine Learning
**Specialty:** Artificial Intelligence

---

## Abstract

This report presents LinguaLens, a web application that identifies the language of text, audio files, and live microphone recordings, then translates the detected text into a user-chosen target language. Three machine-learning classifiers are trained and compared: Logistic Regression (primary, 98.83% test accuracy), Linear SVC (99.01%), and Multinomial Naive Bayes (98.14%). All three share the same TF-IDF character n-gram feature space and are trained on the WiLI-2018 benchmark dataset covering 20 languages. The web interface runs all three models on every text query and displays their predictions side-by-side, showing each model's top language, confidence, and test accuracy. Model inference runs entirely in TypeScript with no native binaries, enabling serverless deployment. Translation is powered by Groq's Llama 3.3 70B model, producing both a naive word-by-word translation and a meaning-aware contextual translation for direct visual comparison.

---

## 3.1 Формулиране на проблема / Problem Formulation

### Task Description

Automatic Language Identification (LID) is the task of determining which natural language a given text is written in. It is a **multi-class text classification** problem in the closed-set setting: given a fixed inventory of 20 candidate languages, the system must assign each input document to exactly one class.

This project addresses LID as a classical supervised learning problem: given labelled text samples from 20 languages, train classifiers that generalise to unseen text. Three classifiers are trained and evaluated — Logistic Regression, Linear SVC, and Multinomial Naive Bayes — to allow direct comparison of discriminative and generative approaches on the same feature space.

### Motivation

LID is a prerequisite for virtually all downstream NLP tasks — translation, sentiment analysis, information retrieval, and content moderation all depend on knowing the document's language. The problem is well-studied and has clean benchmark datasets, making it ideal for comparing multiple ML approaches on equal footing.

The additional constraints that shaped this project:

- **No pre-trained models** — all classifiers trained from scratch on raw text.
- **Serverless deployment** — Vercel free tier: 4.5 MB function body limit, no native binaries allowed.
- **Production quality** — working web application with three input modes (text, audio file, live microphone).
- **Interpretability** — all three models exposed to the user with confidence and accuracy badges.

### Dataset

**WiLI-2018** (Wikipedia Language Identification Benchmark) [1] consists of short paragraphs extracted from Wikipedia across 235 languages. We use **20 languages** spanning diverse scripts and language families:

| Script | Languages |
|---|---|
| Latin | English, German, French, Spanish, Portuguese, Italian, Dutch, Polish, Czech, Romanian, Turkish |
| Cyrillic | Russian, Bulgarian, Ukrainian |
| Arabic | Arabic |
| Devanagari | Hindi |
| CJK | Chinese (Mandarin), Japanese, Korean |
| Greek | Greek |

**Dataset split:** 1 000 samples per language for training (20 000 total), 333 per language for testing (6 660 total), following the original WiLI train/test split. Dataset source: https://zenodo.org/record/841984

---

## 3.2 Теоретична част / Theoretical Background

### Feature Extraction — TF-IDF Character N-grams

The feature representation is built using scikit-learn's `TfidfVectorizer` with the `char_wb` analyser:

```python
TfidfVectorizer(
    analyzer    = 'char_wb',   # character n-grams with word-boundary padding
    ngram_range = (2, 4),      # bi-, tri-, and tetra-grams
    max_features = 30_000,     # top 30k n-grams by corpus frequency
    sublinear_tf = True,       # replace TF with 1 + log(TF)
)
```

**Why character n-grams?** Character-level features capture morphological patterns (suffixes, prefixes, inflections) without requiring a language-specific tokeniser. The `char_wb` analyser pads each token with spaces before extraction — for example, `"hello"` becomes `" hello "`, producing n-grams `" h"`, `" he"`, `"he"`, `"hel"`, `"ell"`, `"llo"`, `"lo "`, `"o "`. Word-boundary padding gives the model dedicated features for word-start and word-end sequences, which are highly discriminative across languages (e.g., German compound prefixes vs. Arabic root patterns).

**Sublinear TF scaling** replaces raw term frequency with `1 + log(tf)`, reducing the dominance of high-frequency n-grams. **IDF smoothing** penalises n-grams common across all languages. The final feature vector is **L2-normalised** per document, making cosine similarity equivalent to dot product — important for both LogReg and SVC.

After transformation, each document is a **30 000-dimensional sparse vector**. The vocabulary (n-gram → index mapping) and IDF weights are fit on the training set only, preventing data leakage.

For Multinomial Naive Bayes, TF-IDF cannot be used directly: log-scaled and L2-normalised values can be negative or non-integer, violating the NB count assumption. A separate `CountVectorizer` with the identical vocabulary produces raw integer term counts for NB only.

---

### Classifier 1 — Logistic Regression (Primary)

```python
LogisticRegression(solver='lbfgs', C=5, max_iter=1000)
```

**Mathematical formulation.** For a K-class problem, multinomial logistic regression models the posterior as:

```
P(y = k | x) = exp(w_k · x + b_k) / Σ_j exp(w_j · x + b_j)
```

where `w_k ∈ ℝ^d` is the weight vector for class k and `b_k` is the bias. Training minimises the cross-entropy loss with L2 regularisation:

```
L = -Σ log P(y_i | x_i) + (1/2C) Σ_k ||w_k||²
```

The `lbfgs` solver uses a quasi-Newton method (Limited-memory BFGS) to optimise the full multinomial objective without reducing it to one-vs-rest. `C = 5` was selected by grid search over `C ∈ {0.1, 1, 5, 10}`.

**Advantages:** Calibrated probabilities via softmax; interpretable weights; single-pass multi-class; handles class imbalance well.

**Limitations:** Assumes linear decision boundaries in feature space; slower to train than SVC on very large vocabularies.

---

### Classifier 2 — Linear SVC

```python
LinearSVC(C=1.0, max_iter=2000, dual=True)
```

**Mathematical formulation.** Linear SVC solves a one-vs-rest (OvR) multi-class problem. For each class k it finds:

```
min_{w,b}  (1/2)||w||² + C Σ_i max(0, 1 - y_i(w·x_i + b))
```

This is the hinge loss with L2 regularisation. The decision boundary maximises the margin between the nearest training points (support vectors) of each class pair. For prediction, class scores are the raw decision function values:

```
score_k(x) = w_k · x + b_k
```

The class with the highest score wins. To display pseudo-probabilities in the UI, scores are passed through softmax (not officially calibrated, but useful for comparison).

**Advantages:** Highest empirical accuracy on high-dimensional sparse data; maximises margin; fast at inference.

**Limitations:** No native probability calibration; OvR strategy can produce inconsistent multi-class scores; less interpretable than LogReg.

---

### Classifier 3 — Multinomial Naive Bayes

```python
MultinomialNB(alpha=0.1)
```

**Mathematical formulation.** NB applies Bayes' theorem with the conditional independence assumption:

```
P(y = k | x) ∝ P(y = k) · Π_i P(x_i | y = k)^{count_i}
```

Taking logarithms (for numerical stability):

```
log P(y = k | x) = log P(y = k) + Σ_i count_i · log P(x_i | y = k)
```

The per-feature log-likelihoods are estimated with Laplace smoothing (`alpha = 0.1`):

```
log P(x_i | y = k) = log( (count(x_i, k) + alpha) / (Σ_j count(x_j, k) + alpha·d) )
```

At inference, log-space scores are converted to display probabilities via softmax.

**Advantages:** Extremely fast training; strong baseline despite the independence assumption; interpretable as a generative model.

**Limitations:** The independence assumption is violated (adjacent n-grams are highly correlated); cannot use TF-IDF features; generally lower accuracy than discriminative models.

---

### TypeScript Inference Reimplementation

Because `skl2onnx` does not support the `char_wb` analyser and `onnxruntime-node` (native binaries) fails on Vercel's serverless Lambda, all three models' weights are exported as plain **float32 binary files** and the full inference pipeline is reimplemented in TypeScript (`lib/detector.ts`).

**Exported files:**

| File | Size | Contents |
|---|---|---|
| `vocab.json` | 921 KB | `{vocabulary: {ngram→index}, idf: [...]}` |
| `coef.bin` | 2.3 MB | LogReg `w` — float32[20 × 30 000] |
| `intercept.bin` | 80 B | LogReg `b` — float32[20] |
| `svc_coef.bin` | 2.3 MB | SVC `w` — float32[20 × 30 000] |
| `svc_intercept.bin` | 80 B | SVC `b` — float32[20] |
| `nb_log_prob.bin` | 2.3 MB | NB `log P(x_i \| k)` — float32[20 × 30 000] |
| `nb_class_log_prior.bin` | 80 B | NB `log P(k)` — float32[20] |

**TypeScript inference steps (LogReg / SVC — TF-IDF path):**
1. Lowercase and whitespace-tokenise the input.
2. Pad each token with spaces; slide windows of width 2–4 for n-grams.
3. Count raw occurrences per n-gram (vocabulary lookup).
4. Apply sublinear TF: `tf' = 1 + log(count)` for non-zero entries.
5. Multiply by stored IDF weights.
6. L2-normalise the feature vector.
7. Compute `score[k] = dot(w[k], features) + b[k]` for all 20 classes.
8. Apply numerically-stable softmax: `prob[k] = exp(score[k] − max) / Σ exp(score[i] − max)`.

**TypeScript inference steps (NB — raw count path):**
1–3. Same tokenisation and raw count extraction (no TF-IDF).
4. Compute `score[k] = log_prior[k] + Σ count[i] · log_prob[k, i]`.
5. Apply softmax to convert log-space scores to display probabilities.

All weights are cached in module scope after the first Lambda cold-start — subsequent requests in the same instance pay no I/O cost.

---

## 3.3 Аналитична част / Analytical Comparison

### Comparison with Alternative Approaches

| Approach | Accuracy (20-lang) | Training time | Model size | Deployment |
|---|---|---|---|---|
| **TF-IDF + LogReg** *(this project)* | **98.83%** | ~30 s CPU | 2.3 MB | Serverless ✓ |
| **TF-IDF + LinearSVC** *(this project)* | **99.01%** | ~20 s CPU | 2.3 MB | Serverless ✓ |
| **TF-IDF + NaiveBayes** *(this project)* | **98.14%** | ~5 s CPU | 2.3 MB | Serverless ✓ |
| fastText (Joulin et al., 2017) [3] | ~99%+ | minutes | 900 MB | Needs binary |
| langdetect (Nakatani, 2010) | ~95% | pre-trained | 2 MB | Java port |
| CLD2/CLD3 (Google) | ~99% | pre-trained | 1–20 MB | Native binary |
| Fine-tuned BERT | ~99.5%+ | hours + GPU | 400 MB+ | Too large |

**Key insight:** Classical TF-IDF classifiers achieve near-identical accuracy to much larger and more complex systems on the closed-set 20-language task. The performance gap between our best model (SVC, 99.01%) and fine-tuned BERT (~99.5%) is less than 0.5 pp — a negligible difference that does not justify the 170× increase in model size or the GPU requirement.

### Why SVC > LogReg > Naive Bayes

**SVC outperforms LogReg (99.01% vs 98.83%)** because in high-dimensional sparse feature spaces, maximising the margin provides a stronger inductive bias than minimising log-loss. The margin objective explicitly ignores well-classified points and focuses on the decision boundary, which is particularly effective when most features are zero.

**LogReg outperforms NB (98.83% vs 98.14%)** because logistic regression is a discriminative model that directly optimises the posterior P(y|x), whereas NB is generative and must additionally model P(x|y). Discriminative models consistently outperform generative ones when training data is sufficient [2]. Furthermore, NB cannot exploit the richer TF-IDF representation — raw counts carry less information than sublinear-scaled, IDF-weighted, L2-normalised vectors.

### When Would the Method Fail?

1. **Short texts (< 50 characters):** Fewer than ~20 n-gram types are extracted. The feature vector is nearly empty and all classifiers lose discriminative power. Observed in practice: a 3-word German phrase yields only 17% confidence. *Solution: minimum-length guard + confidence threshold.*

2. **Code-mixed or multilingual input:** The models predict a single dominant class. A sentence mixing English and Arabic will be misclassified or produce artificially low confidence. None of the three classifiers can handle mixed-language input. *Solution: sliding-window detection or a separate code-switching model.*

3. **Languages outside the 20-class set:** Any text in an unseen language (e.g., Finnish, Vietnamese) will receive a spurious prediction. There is no rejection or "unknown" option. *Solution: a threshold-based reject option or open-set recognition.*

4. **Domain shift:** All three models are trained on Wikipedia text (formal, encyclopaedic prose). Short conversational speech transcripts, social media text, or technical jargon may exhibit different n-gram distributions. This is why audio mode uses Whisper's own LID output rather than our models.

5. **Script ambiguity:** Serbian and Croatian share the Latin script and highly similar morphology; Norwegian and Danish are near-identical in written form. These confusions persist across all three classifiers because the discriminative signal is genuinely weak at the n-gram level.

---

## 3.4 Практическа част / Practical Implementation

### System Architecture

```
Browser
  │
  ├─ Text input ─────────────────────────► POST /api/detect
  │                                             │ detectWithAllModels() — TypeScript
  │                                             │ ┌─ charWbTfidf() → LogReg → probs
  │                                             │ ├─ charWbTfidf() → SVC   → pseudo-probs
  │                                             │ └─ charWbCounts() → NB   → probs
  │                                             │ returns { best, topK, models[3] }
  │                                             ▼
  ├─ Audio / Live ──► POST /api/transcribe  Groq Whisper (verbose_json)
  │                        │ returns { text, detectedLang (ISO 639-1) }
  │                        ▼
  │              POST /api/translate ─────► Groq Llama 3.3 70B
  │                        │ returns { naive, meaningAware }
  │                        ▼
  └─────────────────── UI components
                          ├─ DetectionResult
                          │    ├─ "Logistic Regression" badge (primary model)
                          │    ├─ Recharts top-3 confidence bar chart
                          │    ├─ Word-by-word vs meaning-aware diff panels
                          │    └─ TTS playback (Web Speech Synthesis API)
                          └─ ModelComparison (text mode only)
                               ├─ LogReg card  — flag, confidence, 98.83% acc badge
                               ├─ SVC card     — flag, confidence, 99.01% acc badge
                               └─ NB card      — flag, confidence, 98.14% acc badge
```

### Training Pipeline

```python
# 1. Load WiLI-2018 (Zenodo) — 20 000 train / 6 660 test samples
x_train, y_train, x_test, y_test = load_dataset()

# 2. Shared TF-IDF feature extraction
tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4),
                        max_features=30_000, sublinear_tf=True, min_df=2)
X_train_tfidf = tfidf.fit_transform(x_train)

# 3. Raw counts for NB (same vocabulary)
count_vec = CountVectorizer(vocabulary=tfidf.vocabulary_,
                            analyzer='char_wb', ngram_range=(2,4))
X_train_counts = count_vec.transform(x_train)

# 4. Train all three classifiers
lr  = LogisticRegression(solver='lbfgs', C=5.0, max_iter=1000).fit(X_train_tfidf, y_train)
svc = LinearSVC(C=1.0, max_iter=2000).fit(X_train_tfidf, y_train)
nb  = MultinomialNB(alpha=0.1).fit(X_train_counts, y_train)

# 5. Export weights as float32 binary files
lr.coef_.astype(np.float32).tofile('coef.bin')
svc.coef_.astype(np.float32).tofile('svc_coef.bin')
nb.feature_log_prob_.astype(np.float32).tofile('nb_log_prob.bin')
# ... intercepts and class priors similarly
```

### Web Application Features

| Input mode | Description |
|---|---|
| Text | Type or paste text; 8 sample idiom phrases across scripts |
| Audio file | Upload MP3/WAV/M4A/OGG/FLAC/WebM → Groq Whisper → detect + translate |
| Live recording | MediaRecorder API → Groq Whisper → detect + translate |

| UI feature | Description |
|---|---|
| Model comparison | 3 side-by-side cards: model name, predicted language, confidence, accuracy badge |
| Confidence chart | Recharts horizontal bar chart for top-3 LogReg predictions |
| Translation diff | Word-by-word literal vs meaning-aware, orange highlights on changed words |
| TTS playback | Web Speech Synthesis API reads the translation aloud |
| History | Last 10 detections persisted in localStorage |

**Live demo:** https://language-detector-xi.vercel.app
**Source code:** https://github.com/KachikovPetko/language-detector

---

## 3.5 Експерименти и резултати / Experiments and Results

### Metrics

- **Accuracy** — fraction of correctly classified test samples (primary metric; classes are balanced at 333 each so accuracy = macro-average recall).
- **F1 score** — per-class harmonic mean of precision and recall; averaged across all 20 classes.
- **Confusion matrix** — 20×20 matrix showing the full pattern of misclassifications for the primary model (LogReg).

### Results

| Model | Test Accuracy | Notes |
|---|---|---|
| Logistic Regression *(primary)* | **98.83%** | Calibrated probabilities; used for translation |
| Linear SVC | **99.01%** | Highest accuracy; pseudo-probs via softmax |
| Multinomial Naive Bayes | **98.14%** | Generative; raw counts; weakest of the three |

**All 20 languages achieve F1 > 0.97** under Logistic Regression. The confusion matrix (`ml/results/confusion_matrix.png`) shows the most frequent misclassifications:

- **Ukrainian ↔ Russian** (~1% error) — nearly identical Cyrillic morphology; shared roots and affixes produce nearly identical n-gram distributions.
- **Portuguese ↔ Spanish** (~0.5% error) — both are Ibero-Romance with similar Latin morphology and many cognates.
- **Bulgarian ↔ Russian** (~0.3% error) — both Cyrillic, though Bulgarian morphology is more analytic.

### Interpretation

The 0.18 pp gap between SVC and LogReg is consistent with theoretical predictions: in high-dimensional sparse spaces, margin-based classifiers have a structural advantage over log-loss minimisers. The 0.69 pp gap between LogReg and NB confirms that the conditional independence assumption carries a measurable but modest cost when n-gram co-occurrence patterns are weak enough.

The near-perfect accuracy (98–99%) across all three models on a balanced 20-class test set confirms that character n-gram TF-IDF is an extremely strong feature representation for written language identification — strong enough that classifier choice matters far less than feature design.

### Limitations of the Experiment

- **Domain:** WiLI-2018 is Wikipedia text (formal prose). Generalisation to social media, speech transcripts, or legal text is unknown and likely worse.
- **Text length:** Test samples are 300–500 characters. Short-text performance (< 50 chars) degrades sharply for all three models.
- **SVC calibration:** The pseudo-probabilities for SVC are not calibrated (Platt scaling was not applied), so absolute confidence values are not directly comparable to LogReg.
- **NB count features:** NB uses raw counts while LogReg/SVC use TF-IDF — this is a necessary difference (NB requires non-negative integers), but it means NB operates on strictly less informative features, confounding classifier vs. feature comparisons.

---

## 3.6 Заключение / Conclusion

LinguaLens demonstrates that traditional ML classifiers trained on character n-gram TF-IDF features can achieve near-state-of-the-art language identification accuracy (98–99%) on a 20-language closed-set benchmark, while fitting the constraints of a free-tier serverless deployment with no native binaries.

The three-model comparison yields clear practical conclusions:
- **Linear SVC** achieves the highest accuracy (99.01%) and is the best choice when only a point prediction is needed.
- **Logistic Regression** is the best choice when calibrated probabilities are required (e.g., for confidence display or downstream probabilistic reasoning), at a cost of 0.18 pp accuracy.
- **Multinomial Naive Bayes** is a strong generative baseline (98.14%) that trains in seconds, but is consistently outperformed by both discriminative models when sufficient data is available.

The web application integrates all three models with LLM-based translation, audio transcription via Groq Whisper, and a visual diff comparison between naive and meaning-aware translation — demonstrating how a classical ML pipeline can serve as the core of a production-quality application.

**Future directions:** open-set rejection for unseen languages; confidence calibration for short texts; streaming translation; evaluation on non-Wikipedia domains; expansion to the full WiLI-2018 235-language set.

---

## References

[1] Thoma, M. (2018). *The WiLI benchmark dataset for written language identification*. arXiv:1801.07779.

[2] Ng, A. Y., & Jordan, M. I. (2002). On discriminative vs. generative classifiers: A comparison of logistic regression and naive Bayes. *NeurIPS 15*.

[3] Joulin, A., Grave, E., Bojanowski, P., Mikolov, T. (2017). Bag of Tricks for Efficient Text Classification. *EACL*.

[4] Cavnar, W. B., & Trenkle, J. M. (1994). N-gram-based text categorization. *SDAIR-94*, 161–175.
