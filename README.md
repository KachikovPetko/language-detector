# LinguaLens — ML Language Detector & Translator

> University ML Project — three ML algorithms trained and compared side-by-side, deployed as a full-stack web application.

**Live demo:** https://language-detector-xi.vercel.app

---

## What it does

1. **Detects** the language of any text, audio file, or live microphone recording — using three ML models trained from scratch on WiLI-2018.
2. **Compares** all three models side-by-side (Logistic Regression, Linear SVC, Multinomial Naive Bayes) showing each model's prediction, confidence, and test accuracy.
3. **Translates** the detected text into one of 20 supported languages via Groq Llama 3.3 70B.
4. **Contrasts** a naive word-by-word translation against a meaning-aware one, with word-level diff highlighting to show where context matters.

---

## Features

| Feature | Detail |
|---|---|
| Language detection | TF-IDF char n-grams + Logistic Regression (primary), 98.83% accuracy |
| Model comparison | LogReg vs Linear SVC vs Naive Bayes — 3 cards showing prediction + confidence + test accuracy |
| 20 languages | Arabic, Bulgarian, Czech, Chinese, German, Greek, English, French, Hindi, Italian, Japanese, Korean, Dutch, Polish, Portuguese, Romanian, Russian, Spanish, Turkish, Ukrainian |
| Text input | Type or paste; 8 sample idiom phrases to try |
| Audio file | Upload MP3/WAV/M4A/OGG/FLAC/WebM → Groq Whisper transcription |
| Live recording | Record mic → MediaRecorder + Groq Whisper → detect + translate |
| Translation comparison | Word-by-word literal vs meaning-aware, orange diff highlights |
| Confidence chart | Recharts horizontal bar chart for top-3 language scores |
| TTS playback | Listen to the translated text via Web Speech Synthesis |
| History | Last 10 translations persisted in localStorage |

---

## Quick start

```bash
# 1. Install dependencies
npm install

# 2. Add your Groq API key  (get one free at console.groq.com)
cp .env.example .env.local
# Edit .env.local:  GROQ_API_KEY=gsk_...

# 3. Run dev server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

The trained model files are already committed to `public/model/`. To retrain from scratch see [ml/README.md](ml/README.md).

---

## Project structure

```
app/
  api/detect/       POST — runs all 3 ML models, returns predictions + model comparison
  api/translate/    POST — dual translation (naive + meaning-aware)
  api/transcribe/   POST — audio → text via Groq Whisper
  page.tsx          Main app shell with useReducer state

components/
  TextInput.tsx           Textarea + sample phrase pills
  FileUpload.tsx          Drag-drop audio upload + transcription
  LiveRecorder.tsx        MediaRecorder + Whisper live recording
  DetectionResult.tsx     Recharts chart, diff panels, TTS button
  ModelComparison.tsx     3-card side-by-side model comparison
  HistoryPanel.tsx        localStorage history (last 10)
  ModeToggle.tsx          Text / Audio File / Live Recording switcher
  TargetLanguagePicker.tsx  Dropdown for 20 target languages

lib/
  detector.ts       Pure-TypeScript inference for all 3 models
  groq.ts           Groq API client (Whisper + Llama)
  languages.ts      ISO 639-1/3 language table
  types.ts          Shared TypeScript types

ml/
  train.py          Download WiLI-2018, train all 3 models, save detector.pkl
  export_weights.py Export 8 binary weight files for TypeScript inference
  export_onnx.py    Optional ONNX export (LogReg only, for reference)
  results/          Confusion matrix, per-class metrics

public/model/
  vocab.json               TF-IDF vocabulary + IDF weights  (921 KB)
  labels.json              Class order (20 ISO 639-3 codes)
  coef.bin                 LogReg coefficients  (2.3 MB, float32)
  intercept.bin            LogReg intercepts    (80 bytes)
  svc_coef.bin             LinearSVC coefficients  (2.3 MB, float32)
  svc_intercept.bin        LinearSVC intercepts    (80 bytes)
  nb_log_prob.bin          NB log P(feature|class)  (2.3 MB, float32)
  nb_class_log_prior.bin   NB log P(class)          (80 bytes)
  model_accuracy.json      Test accuracy for all 3 models
```

---

## ML models

Three classifiers are trained on the same TF-IDF feature space and compared in the UI:

| Model | Role | Test Accuracy |
|---|---|---|
| `LogisticRegression(solver='lbfgs', C=5)` | Primary — used for detection & translation | **98.83%** |
| `LinearSVC(C=1.0)` | Discriminative baseline | 99.01% |
| `MultinomialNB(alpha=0.1)` | Generative baseline (raw counts) | 98.14% |

**Features:** `TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4), max_features=30000, sublinear_tf=True)`

**Dataset:** WiLI-2018 (Thoma, 2018) — Wikipedia excerpts, 235 languages. We use 20 of them, 1 000 samples each for training, 333 for testing.

**Inference:** All three models run entirely in TypeScript with no native binaries. `lib/detector.ts` reimplements `char_wb` TF-IDF, LinearSVC decision function, NB log-probability scoring, and numerically-stable softmax — loading weights from `public/model/` at cold-start.

> `onnxruntime-node` was removed after testing: skl2onnx does not support char-level analyzers, and native binaries caused failures on Vercel's serverless Lambda runtime.

---

## Stack

| Layer | Technology |
|---|---|
| Framework | Next.js 16 (App Router), React 19, TypeScript strict |
| Styling | Tailwind CSS v4 |
| ML inference | Pure TypeScript (no native deps) |
| Charts | Recharts 3 |
| Icons | lucide-react |
| Translation | Groq API — `llama-3.3-70b-versatile` |
| Transcription | Groq API — `whisper-large-v3` (verbose_json) |
| Deployment | Vercel free tier |

---

## Deployment

1. Push to GitHub (model files are committed — they fit Vercel's 250 MB limit).
2. Import repo on vercel.com.
3. Add environment variable: `GROQ_API_KEY = gsk_...`
4. Deploy.

No build-time ML step — all model weights load from `public/model/` at request time.
