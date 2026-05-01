import path from 'path'
import fs from 'fs'
import { getByIso3 } from './languages'
import type { Detection } from './types'

// ---- File paths -------------------------------------------------------------

const MODEL_DIR      = path.join(process.cwd(), 'public', 'model')
const VOCAB_PATH     = path.join(MODEL_DIR, 'vocab.json')
const LABELS_PATH    = path.join(MODEL_DIR, 'labels.json')
const COEF_PATH      = path.join(MODEL_DIR, 'coef.bin')
const INTERCEPT_PATH = path.join(MODEL_DIR, 'intercept.bin')

// ---- Types ------------------------------------------------------------------

interface VocabJson   { vocabulary: Record<string, number>; idf: number[] }
interface LabelsJson  { classes: string[] }

// ---- Module-level cache — one load per Lambda cold-start -------------------

let vocabulary:  Record<string, number> | null = null
let idf:         Float32Array | null = null
let coef:        Float32Array | null = null   // [n_classes × n_features] row-major
let intercept:   Float32Array | null = null
let classes:     string[]     | null = null
let nFeatures = 0
let nClasses  = 0

function loadModels() {
  if (vocabulary) return  // already loaded

  for (const p of [VOCAB_PATH, LABELS_PATH, COEF_PATH, INTERCEPT_PATH]) {
    if (!fs.existsSync(p)) {
      throw new Error(
        `Model file missing: ${path.basename(p)}. ` +
        'Run ml/train.py → ml/export_onnx.py → ml/export_weights.py first.'
      )
    }
  }

  const vocabRaw  = JSON.parse(fs.readFileSync(VOCAB_PATH,  'utf-8')) as VocabJson
  const labelsRaw = JSON.parse(fs.readFileSync(LABELS_PATH, 'utf-8')) as LabelsJson

  vocabulary = vocabRaw.vocabulary
  idf        = new Float32Array(vocabRaw.idf)
  nFeatures  = idf.length
  classes    = labelsRaw.classes
  nClasses   = classes.length

  // Binary files: raw IEEE-754 float32, little-endian (NumPy default on x86)
  const coefBuf  = fs.readFileSync(COEF_PATH)
  const intBuf   = fs.readFileSync(INTERCEPT_PATH)
  coef      = new Float32Array(coefBuf.buffer,  coefBuf.byteOffset,  coefBuf.byteLength  / 4)
  intercept = new Float32Array(intBuf.buffer,   intBuf.byteOffset,   intBuf.byteLength   / 4)
}

// ---- TF-IDF char_wb transform -----------------------------------------------
// Mirrors sklearn TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4),
// sublinear_tf=True, norm='l2', smooth_idf=True)

function charWbTfidf(text: string): Float32Array {
  const vocab  = vocabulary!
  const idfArr = idf!

  const tokens = text.toLowerCase().split(/\s+/).filter(t => t.length > 0)
  const counts = new Float32Array(nFeatures)

  for (const token of tokens) {
    const padded = ` ${token} `
    for (let n = 2; n <= 4; n++) {
      for (let i = 0; i <= padded.length - n; i++) {
        const idx = vocab[padded.slice(i, i + n)]
        if (idx !== undefined) counts[idx]++
      }
    }
  }

  // sublinear TF × IDF, then L2 normalise
  const features = new Float32Array(nFeatures)
  let norm = 0
  for (let i = 0; i < nFeatures; i++) {
    if (counts[i] > 0) {
      features[i] = (1 + Math.log(counts[i])) * idfArr[i]
      norm += features[i] * features[i]
    }
  }
  if (norm > 0) {
    const inv = 1 / Math.sqrt(norm)
    for (let i = 0; i < nFeatures; i++) features[i] *= inv
  }

  return features
}

// ---- Logistic regression inference ------------------------------------------
// scores[c] = dot(coef[c], features) + intercept[c]
// probs = softmax(scores)

function logregPredict(features: Float32Array): Float32Array {
  const c_ = coef!
  const b_  = intercept!

  // Compute raw scores
  const scores = new Float32Array(nClasses)
  for (let c = 0; c < nClasses; c++) {
    let dot = b_[c]
    const offset = c * nFeatures
    for (let f = 0; f < nFeatures; f++) dot += c_[offset + f] * features[f]
    scores[c] = dot
  }

  // Numerically-stable softmax
  let maxScore = scores[0]
  for (let c = 1; c < nClasses; c++) if (scores[c] > maxScore) maxScore = scores[c]

  const probs = new Float32Array(nClasses)
  let sum = 0
  for (let c = 0; c < nClasses; c++) { probs[c] = Math.exp(scores[c] - maxScore); sum += probs[c] }
  for (let c = 0; c < nClasses; c++) probs[c] /= sum

  return probs
}

// ---- Public API -------------------------------------------------------------

export function detectLanguage(text: string, topK = 3): Detection[] {
  loadModels()  // no-op if already loaded

  const features = charWbTfidf(text)
  const probs    = logregPredict(features)

  return Array.from(probs)
    .map((confidence, i) => {
      const lang = getByIso3(classes![i])
      return lang ? { language: lang, confidence } : null
    })
    .filter((d): d is Detection => d !== null)
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, topK)
}
