# LinguaLens: ML-Powered Language Detection and Translation
### University ML Project Report

---

## Abstract

This report presents LinguaLens, a web application that identifies the language of text, audio files, and live microphone recordings, and then translates the detected text into a target language chosen by the user. Three machine-learning classifiers are trained and compared: Logistic Regression (primary, 98.83% test accuracy), Linear SVC (99.01%), and Multinomial Naive Bayes (98.14%). All share the same TF-IDF character n-gram feature space and are trained on the WiLI-2018 benchmark dataset covering 20 languages. The web interface runs all three models on every text query and displays their predictions side-by-side, showing each model's top language, confidence, and test accuracy. Model inference runs entirely in TypeScript with no native binaries, enabling serverless deployment. Translation is powered by Groq's Llama 3.3 70B model, which produces both a naive word-by-word translation and a meaning-aware contextual translation for direct comparison.

---

## 1. Introduction

Automatic Language Identification (LID) is a well-studied problem in Natural Language Processing. Knowing a document's language is a prerequisite for downstream tasks such as translation, sentiment analysis, and information retrieval. Classical approaches based on character n-gram statistics have proven highly competitive against neural methods, particularly for the closed-set scenario where the number of candidate languages is fixed in advance [1].

This project implements and deploys three classical classifiers on a shared feature space, with the deliberate constraints that the models must:

- Be trained from scratch (no pre-trained embeddings or language models for detection),
- Run in a serverless environment (Vercel free tier, 4.5 MB body limit, no native binaries),
- Achieve high accuracy on a standard benchmark,
- Serve as the ML backend for a production-quality web application that visually compares all three models.

The web application additionally demonstrates the difference between naive word-by-word machine translation and modern meaning-aware neural translation, using visual diff highlighting to make the linguistic contrast tangible.

---

## 2. Dataset

**WiLI-2018** (Wikipedia Language Identification Benchmark) [1] consists of paragraphs extracted from Wikipedia across 235 languages. Each sample is a short text passage (typically 300–500 characters). We use 20 languages that cover a diverse range of scripts and language families:

| Script | Languages |
|---|---|
| Latin | English, German, French, Spanish, Portuguese, Italian, Dutch, Polish, Czech, Romanian, Turkish |
| Cyrillic | Russian, Bulgarian, Ukrainian |
| Arabic | Arabic |
| Devanagari | Hindi |
| CJK | Chinese (Mandarin), Japanese, Korean |
| Greek | Greek |

**Split:** 1 000 samples per language for training (20 000 total), 333 per language for testing (6 660 total), following the original WiLI train/test split.

---

## 3. Method

### 3.1 Feature Extraction

We use scikit-learn's `TfidfVectorizer` with the `char_wb` analyser:

```python
TfidfVectorizer(
    analyzer    = 'char_wb',   # character n-grams with word-boundary padding
    ngram_range = (2, 4),      # bi-, tri-, and tetra-grams
    max_features = 30_000,     # top 30k n-grams by corpus frequency
    sublinear_tf = True,       # replace TF with 1 + log(TF)
)
```

The `char_wb` analyser pads each token with spaces before extracting n-grams — for example, the word *"hello"* yields the padded form `" hello "`, producing n-grams like `" h"`, `" he"`, `" hel"`, `"he"`, `"hel"`, `"hell"`, etc. Word-boundary padding means word-start and word-end character sequences receive their own features, which are particularly informative for distinguishing inflected languages.

After TF-IDF transformation, each document is a 30 000-dimensional sparse vector. The TF component uses sublinear scaling (`1 + log(tf)`) to reduce the dominance of high-frequency n-grams. The IDF component uses add-one smoothing. The final feature vector is L2-normalised per document.

For Multinomial Naive Bayes, TF-IDF cannot be used directly (log-scaled and L2-normalised values can be negative or non-integer, violating the NB count assumption). A separate `CountVectorizer` with the identical vocabulary produces raw integer term counts for NB training and inference only.

### 3.2 Classifiers

Three classifiers are trained on the same feature space and compared side-by-side in the web interface.

**Logistic Regression (primary)** — used for the final language detection result and translation:
```python
LogisticRegression(solver='lbfgs', C=5, max_iter=1000)
```
The lbfgs solver handles the 20-class problem natively via a multinomial (softmax) objective. L2 regularisation with `C = 5` performed best in preliminary experiments over `C ∈ {0.1, 1, 5, 10}`. The model outputs calibrated class probabilities directly via softmax, making it the most interpretable choice for the confidence display.

**Linear SVC** — discriminative baseline:
```python
LinearSVC(C=1.0, max_iter=2000, dual=True)
```
Trained on the same TF-IDF features as LogReg using a one-vs-rest (OvR) strategy. Linear SVC maximises the margin between classes and often achieves slightly higher accuracy than LogReg on high-dimensional sparse data. It does not produce native probability estimates; for display, the raw decision function scores are passed through softmax to yield pseudo-probabilities.

**Multinomial Naive Bayes** — generative baseline:
```python
MultinomialNB(alpha=0.1)
```
Trained on raw term counts (not TF-IDF). The generative model estimates `log P(class | features) ∝ Σ count_i · log P(x_i | class) + log P(class)`, where the per-feature log-likelihoods and class priors are fit on the training set. Laplace smoothing with `alpha=0.1` avoids zero-probability issues for unseen n-grams. Despite its strong independence assumption (features treated as conditionally independent given the class), NB achieves competitive accuracy on this task because character n-gram frequencies are naturally sparse and nearly independent across languages.

### 3.3 Inference in TypeScript

Because `skl2onnx` does not support the `char_wb` analyser, and because `onnxruntime-node` (native binaries) fails on Vercel's serverless Lambda, all three models' weights are exported as plain float32 binary files and the inference pipeline is reimplemented in TypeScript (`lib/detector.ts`).

**Shared feature extraction (all models):**
1. Lowercase and whitespace-tokenise the input.
2. Pad each token with spaces: `"word"` → `" word "`.
3. Slide windows of width 2–4 to extract character n-grams.
4. Look up each n-gram in the stored vocabulary → sparse count vector.

**LogReg and SVC inference (TF-IDF features):**
5. Apply sublinear TF: `tf' = 1 + log(count)` for each non-zero entry.
6. Multiply by stored IDF weights.
7. L2-normalise the feature vector.
8. Compute `score[c] = dot(coef[c], features) + intercept[c]` for each of 20 classes.
9. Apply numerically-stable softmax: `prob[c] = exp(score[c] − max) / Σ exp(score[i] − max)`.

**NB inference (raw count features):**
5. Use raw integer counts directly (no TF-IDF step).
6. Compute `score[c] = class_log_prior[c] + Σ count[i] · feature_log_prob[c, i]`.
7. Apply softmax to convert log-space scores to display probabilities.

All three models run in the same API route (`POST /api/detect`) and return their top prediction in a single response. The `detectWithAllModels()` function in `lib/detector.ts` caches all weight files in module scope, so subsequent requests in the same Lambda instance pay no I/O cost.

### 3.4 Translation

Translation uses Groq's `llama-3.3-70b-versatile` model via the Groq API. Two translations are generated in parallel per request:

- **Naive (word-by-word):** the model is instructed to translate each word independently in the original order, without adjusting grammar or word order. This deliberately produces unnatural output to illustrate how context-free translation fails.
- **Meaning-aware:** the model is instructed to translate naturally, preserving meaning, tone, and register, using only the target language's script.

A word-level diff algorithm highlights tokens in the meaning-aware translation that do not appear in the naive translation, making the improvements visually explicit.

### 3.5 Audio Transcription and Language Detection

For audio inputs (file upload and live microphone recording), Groq's `whisper-large-v3` model transcribes the speech and — via the `verbose_json` response format — returns the detected language as an ISO 639-1 code. This Whisper-based language tag replaces the ML model's output for audio modes, because the ML model (trained on Wikipedia text) performs poorly on short conversational speech transcripts. The ML model comparison panel is shown only for text-input mode, where all three models are most reliable.

---

## 4. Evaluation

### 4.1 Overall Accuracy

| Model | Test Accuracy | Notes |
|---|---|---|
| Logistic Regression *(primary)* | **98.83%** | Calibrated probabilities via softmax |
| Linear SVC | **99.01%** | Highest accuracy; pseudo-probs via softmax on decision scores |
| Multinomial Naive Bayes | **98.14%** | Raw counts; strong independence assumption |

All three models are evaluated on the same 6 660-sample held-out test set (333 samples × 20 languages).

Linear SVC edges out LogReg by 0.18 percentage points — consistent with the literature showing SVM-based methods outperforming logistic regression on high-dimensional sparse text. NB trails by 0.69 pp, reflecting the cost of the independence assumption and the inability to use the richer TF-IDF representation.

### 4.2 Per-language Performance

The confusion matrix (see `ml/results/confusion_matrix.png`) shows near-perfect classification for most languages. The most frequent confusions are:

- **Ukrainian ↔ Russian** — closely related Cyrillic scripts with shared n-grams.
- **Portuguese ↔ Spanish** — similar Latin-script morphology.
- **Norwegian / Danish** (not in our set, but present in WiLI) contamination in the Wikipedia source texts occasionally introduces near-misses.

All 20 languages achieve F1 > 0.97 under Logistic Regression.

### 4.3 Limitations

- **Short texts:** all three models are unreliable on texts shorter than ~50 characters, as there are too few n-grams to produce a confident prediction. The UI shows a low-confidence warning below 40% and audio mode uses Whisper instead.
- **Mixed-language text:** each model predicts a single dominant language; code-switching or mixed documents will produce low confidence.
- **Closed vocabulary:** any language not in the 20-class training set will receive a spurious prediction with no rejection option.
- **SVC pseudo-probabilities:** LinearSVC has no native probability calibration. The softmax-transformed decision scores are useful for relative comparison but are not well-calibrated in an absolute sense.

---

## 5. System Architecture

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
                          │    ├─ "Logistic Regression" badge (top-right)
                          │    ├─ Recharts confidence bar chart (top-3 languages)
                          │    ├─ Word-by-word vs meaning-aware diff panels
                          │    └─ TTS playback (Web Speech Synthesis API)
                          └─ ModelComparison (text mode only)
                               ├─ LogReg card  — prediction, confidence, 98.83% acc
                               ├─ SVC card     — prediction, confidence, 99.01% acc
                               └─ NB card      — prediction, confidence, 98.14% acc
```

The application is a Next.js 16 App Router project deployed on Vercel's free tier. API routes run as serverless Node.js functions. All ML inference happens server-side at request time; model weights (8 binary files, ~7 MB total) are loaded once per Lambda cold-start and cached in module scope.

---

## 6. Discussion

### Model Comparison: LogReg vs SVC vs NB

The three models are trained on an identical feature space, so accuracy differences reflect the classifiers themselves rather than feature engineering choices.

Linear SVC achieves the highest accuracy (99.01%) because it directly maximises the decision margin, making it well-suited to high-dimensional sparse feature spaces where the margin is the most informative signal. Logistic Regression is a close second (98.83%) — its probabilistic formulation provides better-calibrated output at a marginal accuracy cost. Naive Bayes ranks third (98.14%): the conditional independence assumption is clearly violated (adjacent n-grams are correlated), but the violation is mild enough that the model still generalises well.

In the web UI, users can observe the models agreeing or disagreeing on edge cases — short texts, code-mixed input, or less-represented scripts — making the comparison pedagogically valuable.

### Classical vs Neural Detection

The character n-gram approach achieves 98–99% accuracy across all three classifiers — comparable to results reported in the WiLI-2018 paper and competitive with significantly more complex neural language models. For the closed-set 20-language scenario with adequate training data per class, classical methods remain the pragmatic choice: they are interpretable, fast to train (under 2 minutes on CPU), and produce tiny deployable artefacts (< 7 MB total for all three models).

### Deployment Constraints

The Vercel free tier imposes a 4.5 MB body limit on serverless functions and does not allow arbitrary native binaries. This drove two architectural decisions: (1) exporting model weights as plain float32 binary files rather than ONNX, and (2) reimplementing all three inference pipelines in TypeScript. Both constraints proved beneficial: the TypeScript implementation is fully transparent, easily debuggable, and adds zero cold-start overhead beyond a file read.

### Translation Quality

The side-by-side comparison reveals a consistent pattern: word-by-word translation preserves individual lexical items but breaks idioms, inverts modifier–noun order in languages with adjective-postposition, and misses grammatical agreement entirely. Meaning-aware translation handles all of these correctly. The diff highlighting makes this contrast immediately legible to users without any linguistic background.

---

## 7. Conclusion

LinguaLens demonstrates that traditional ML classifiers can achieve near-state-of-the-art language identification accuracy while fitting the constraints of a free-tier serverless deployment. By training and comparing three algorithms — Logistic Regression, Linear SVC, and Multinomial Naive Bayes — on the same feature space, the project illustrates the practical trade-offs between discriminative and generative approaches. The web interface exposes these trade-offs interactively, allowing users to observe where models agree, where they diverge, and how confidence correlates with text length. The project further integrates the detection pipeline with LLM-based translation to build a practical, demo-ready application with three input modalities, visual diff highlighting, and full audio support.

Future directions include: expanding to more languages, adding confidence calibration for short texts, implementing streaming translation, and evaluating all three models on non-Wikipedia domains (social media, legal text, etc.).

---

## References

[1] Thoma, M. (2018). *The WiLI benchmark dataset for written language identification*. arXiv:1801.07779.

[2] Cavnar, W. B., & Trenkle, J. M. (1994). N-gram-based text categorization. *SDAIR-94*, 161–175.

[3] Joulin, A., Grave, E., Bojanowski, P., Mikolov, T. (2017). Bag of Tricks for Efficient Text Classification. *EACL*.
