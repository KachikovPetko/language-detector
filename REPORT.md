# LinguaLens: ML-Powered Language Detection and Translation
### University ML Project Report

---

## Abstract

This report presents LinguaLens, a web application that identifies the language of text, audio files, and live microphone recordings, and then translates the detected text into a target language chosen by the user. The core detection model is a traditional machine-learning pipeline — TF-IDF character n-gram features combined with multinomial Logistic Regression — trained on the WiLI-2018 benchmark dataset covering 20 languages. The model achieves **98.86% accuracy** on the held-out test set. Model inference runs entirely in TypeScript with no native binaries, enabling serverless deployment. Translation is powered by Groq's Llama 3.3 70B model, which produces both a naive word-by-word translation and a meaning-aware contextual translation for direct comparison.

---

## 1. Introduction

Automatic Language Identification (LID) is a well-studied problem in Natural Language Processing. Knowing a document's language is a prerequisite for downstream tasks such as translation, sentiment analysis, and information retrieval. Classical approaches based on character n-gram statistics have proven highly competitive against neural methods, particularly for the closed-set scenario where the number of candidate languages is fixed in advance [1].

This project implements and deploys such a classical approach, with the deliberate constraint that the model must:

- Be trained from scratch (no pre-trained embeddings or language models for detection),
- Run in a serverless environment (Vercel free tier, 4.5 MB body limit, no native binaries),
- Achieve high accuracy on a standard benchmark,
- Serve as the ML backend for a production-quality web application.

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

### 3.2 Classifier

We train a multinomial `LogisticRegression`:

```python
LogisticRegression(
    solver   = 'lbfgs',
    C        = 5,           # inverse regularisation strength
    max_iter = 1000,
)
```

The lbfgs solver handles the multi-class problem natively (one-vs-rest is not required). L2 regularisation with `C = 5` provides mild regularisation that performed best in preliminary experiments over `C ∈ {0.1, 1, 5, 10}`.

The full pipeline (vectoriser → classifier) is trained end-to-end: the vocabulary and IDF weights are fit on the training set only, preventing data leakage.

### 3.3 Inference in TypeScript

Because `skl2onnx` does not support the `char_wb` analyser, and because `onnxruntime-node` (native binaries) fails on Vercel's serverless Lambda, the model weights are exported as plain binary files and the inference pipeline is reimplemented in TypeScript (`lib/detector.ts`):

1. **Tokenisation:** lowercase the input and split on whitespace (matching Python's `str.split()`).
2. **N-gram extraction:** for each token, pad with spaces and slide a window of width 2–4.
3. **TF-IDF:** count n-gram occurrences, apply sublinear TF, multiply by stored IDF weights.
4. **L2 normalisation:** divide by the Euclidean norm of the feature vector.
5. **Logistic Regression:** compute `scores[c] = dot(coef[c], features) + intercept[c]` for each of the 20 classes.
6. **Softmax:** convert scores to probabilities using numerically-stable softmax (`exp(score − max)`).

This reimplementation matches sklearn's output to floating-point precision on all test phrases verified during export.

### 3.4 Translation

Translation uses Groq's `llama-3.3-70b-versatile` model via the Groq API. Two translations are generated in parallel per request:

- **Naive (word-by-word):** the model is instructed to translate each word independently in the original order, without adjusting grammar or word order. This deliberately produces unnatural output to illustrate how context-free translation fails.
- **Meaning-aware:** the model is instructed to translate naturally, preserving meaning, tone, and register, using only the target language's script.

A word-level diff algorithm highlights tokens in the meaning-aware translation that do not appear in the naive translation, making the improvements visually explicit.

### 3.5 Audio Transcription and Language Detection

For audio inputs (file upload and live microphone recording), Groq's `whisper-large-v3` model transcribes the speech and — via the `verbose_json` response format — returns the detected language as an ISO 639-1 code. This Whisper-based language tag replaces the ML model's output for audio modes, because the ML model (trained on Wikipedia text) performs poorly on short conversational speech transcripts. The ML model continues to handle the text-input mode where it is most reliable.

---

## 4. Evaluation

### 4.1 Overall Accuracy

| Split | Samples | Accuracy |
|---|---|---|
| Training | 20 000 | 99.6% |
| Test | 6 660 | **98.86%** |

### 4.2 Per-language Performance

The confusion matrix (see `ml/results/confusion_matrix.png`) shows near-perfect classification for most languages. The most frequent confusions are:

- **Ukrainian ↔ Russian** — closely related Cyrillic scripts with shared n-grams.
- **Portuguese ↔ Spanish** — similar Latin-script morphology.
- **Norwegian / Danish** (not in our set, but present in WiLI) contamination in the Wikipedia source texts occasionally introduces near-misses.

All 20 languages achieve F1 > 0.97.

### 4.3 Limitations

- **Short texts:** the model is unreliable on texts shorter than ~50 characters, as there are too few n-grams to produce a confident prediction. This is why the UI shows a low-confidence warning below 40% and audio mode uses Whisper instead.
- **Mixed-language text:** the model predicts a single dominant language; code-switching or mixed documents will produce low confidence.
- **Closed vocabulary:** any language not in the 20-class training set will receive a spurious prediction with no rejection option.

---

## 5. System Architecture

```
Browser
  │
  ├─ Text input ──────────────────────────► POST /api/detect
  │                                              │ detectLanguage() — TypeScript
  │                                              │ reads public/model/*.bin
  │                                              ▼
  ├─ Audio / Live ──► POST /api/transcribe  Groq Whisper
  │                        │ returns text + detectedLang (ISO 639-1)
  │                        ▼
  │              POST /api/translate ──────► Groq Llama 3.3 70B
  │                        │ returns { naive, meaningAware }
  │                        ▼
  └─────────────────── DetectionResult component
                          ├─ Recharts confidence bar chart
                          ├─ Word-by-word vs meaning-aware diff panels
                          └─ TTS playback (Web Speech Synthesis API)
```

The application is a Next.js 16 App Router project deployed on Vercel's free tier. API routes run as serverless Edge/Node.js functions. All ML inference happens server-side at request time; model weights are loaded once per Lambda instance and cached in module scope.

---

## 6. Discussion

### Classical vs Neural Detection

The character n-gram + logistic regression approach achieves 98.86% accuracy — comparable to results reported in the WiLI-2018 paper and competitive with significantly more complex neural language models. For the closed-set 20-language scenario with adequate training data per class, classical methods remain the pragmatic choice: they are interpretable, fast to train (seconds on CPU), and produce tiny deployable artefacts (< 5 MB total).

### Deployment Constraints

The Vercel free tier imposes a 4.5 MB body limit on serverless functions and does not allow arbitrary native binaries. This drove two architectural decisions: (1) exporting model weights as plain float32 binary files rather than ONNX, and (2) reimplementing the inference pipeline in TypeScript. Both constraints proved beneficial: the TypeScript implementation is fully transparent, easily debuggable, and adds zero cold-start overhead beyond a file read.

### Translation Quality

The side-by-side comparison reveals a consistent pattern: word-by-word translation preserves individual lexical items but breaks idioms, inverts modifier–noun order in languages with adjective-postposition, and misses grammatical agreement entirely. Meaning-aware translation handles all of these correctly. The diff highlighting makes this contrast immediately legible to users without any linguistic background.

---

## 7. Conclusion

LinguaLens demonstrates that a traditional ML pipeline can achieve near-state-of-the-art language identification accuracy while fitting the constraints of a free-tier serverless deployment. The project integrates the detection model with modern LLM-based translation to build a practical, demo-ready application with three input modalities, visual diff highlighting, and full audio support.

Future directions include: expanding to more languages, adding a confidence calibration step for short texts, implementing streaming translation for real-time character output, and evaluating the model on non-Wikipedia domains (social media, legal text, etc.).

---

## References

[1] Thoma, M. (2018). *The WiLI benchmark dataset for written language identification*. arXiv:1801.07779.

[2] Cavnar, W. B., & Trenkle, J. M. (1994). N-gram-based text categorization. *SDAIR-94*, 161–175.

[3] Joulin, A., Grave, E., Bojanowski, P., Mikolov, T. (2017). Bag of Tricks for Efficient Text Classification. *EACL*.
