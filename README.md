# LinguaLens — Language Detector & Translator

A university ML project: detect the language of any text using a model **trained from scratch**, then translate it using Groq's Llama 3.1 70B.

## Live Demo

🚀 **[add URL after deploy]**

## Features

- **20 languages** — Arabic, Bulgarian, Czech, Chinese, German, Greek, English, French, Hindi, Italian, Japanese, Korean, Dutch, Polish, Portuguese, Romanian, Russian, Spanish, Turkish, Ukrainian
- **Custom-trained ML model** — TF-IDF character n-grams + Logistic Regression, 98.86% accuracy on WiLI-2018
- **Meaning-aware translation** via Groq Llama 3.1 70B
- Phase 2+: Audio file upload (Groq Whisper), Live recording (Web Speech API), word-by-word vs meaning-aware comparison

## Quick Start

```bash
# 1. Install dependencies
npm install

# 2. Add your Groq API key
cp .env.example .env.local
# Edit .env.local: GROQ_API_KEY=your_key_here

# 3. Run dev server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

> The app requires the trained ONNX model in `public/model/`. See [ml/README.md](ml/README.md) to train it.

## Project Structure

```
ml/              Python training code (train.py, export_onnx.py)
public/model/    Trained model files (detector.onnx, vocab.json, labels.json)
app/             Next.js App Router pages and API routes
components/      React UI components
lib/             Shared TypeScript utilities
```

## ML Model

- **Architecture**: `TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4), max_features=30000)` + `LogisticRegression(lbfgs)`
- **Dataset**: WiLI-2018 — 10,000 samples across 20 languages
- **Accuracy**: 98.86% on held-out test set
- **Export**: vocab+IDF as JSON, LogisticRegression as ONNX (skl2onnx does not support char-level analyzers directly)
- See [ml/README.md](ml/README.md) for retrain instructions

## Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 16, React 19, Tailwind CSS v4 |
| ML Inference | onnxruntime-node + TypeScript TF-IDF impl |
| Translation | Groq Llama 3.1 70B |
| Transcription (Ph2) | Groq Whisper-large-v3 |

## Deployment

Vercel free tier. Set `GROQ_API_KEY` in Vercel → Settings → Environment Variables.
