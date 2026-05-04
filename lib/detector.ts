import path from 'path'
import fs from 'fs'
import { getByIso3 } from './languages'
import type { Detection, ModelPrediction } from './types'

// ---- File paths -------------------------------------------------------------

const MODEL_DIR              = path.join(process.cwd(), 'public', 'model')
const VOCAB_PATH             = path.join(MODEL_DIR, 'vocab.json')
const LABELS_PATH            = path.join(MODEL_DIR, 'labels.json')
const COEF_PATH              = path.join(MODEL_DIR, 'coef.bin')
const INTERCEPT_PATH         = path.join(MODEL_DIR, 'intercept.bin')
const SVC_COEF_PATH          = path.join(MODEL_DIR, 'svc_coef.bin')
const SVC_INTERCEPT_PATH     = path.join(MODEL_DIR, 'svc_intercept.bin')
const NB_LOG_PROB_PATH       = path.join(MODEL_DIR, 'nb_log_prob.bin')
const NB_CLASS_LOG_PRIOR_PATH = path.join(MODEL_DIR, 'nb_class_log_prior.bin')
const MODEL_ACCURACY_PATH    = path.join(MODEL_DIR, 'model_accuracy.json')

// ---- Types ------------------------------------------------------------------

interface VocabJson  { vocabulary: Record<string, number>; idf: number[] }
interface LabelsJson { classes: string[] }
interface AccuracyJson { logistic_regression: number; linear_svc: number; naive_bayes: number }

// ---- Module-level cache — one load per Lambda cold-start -------------------

let vocabulary:       Record<string, number> | null = null
let idf:              Float32Array | null = null
let coef:             Float32Array | null = null
let intercept:        Float32Array | null = null
let svcCoef:          Float32Array | null = null
let svcIntercept:     Float32Array | null = null
let nbLogProb:        Float32Array | null = null
let nbClassLogPrior:  Float32Array | null = null
let classes:          string[] | null = null
let accuracies:       AccuracyJson | null = null
let nFeatures = 0
let nClasses  = 0
let hasAllModels = false

function loadBin(p: string): Float32Array {
  const buf = fs.readFileSync(p)
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4)
}

function loadModels() {
  if (vocabulary) return

  for (const p of [VOCAB_PATH, LABELS_PATH, COEF_PATH, INTERCEPT_PATH]) {
    if (!fs.existsSync(p)) {
      throw new Error(
        `Model file missing: ${path.basename(p)}. ` +
        'Run ml/train.py then ml/export_weights.py first.'
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

  coef      = loadBin(COEF_PATH)
  intercept = loadBin(INTERCEPT_PATH)

  // Additional models (optional — only present after re-training with train.py v2)
  const extraPaths = [SVC_COEF_PATH, SVC_INTERCEPT_PATH, NB_LOG_PROB_PATH, NB_CLASS_LOG_PRIOR_PATH]
  if (extraPaths.every(p => fs.existsSync(p))) {
    svcCoef         = loadBin(SVC_COEF_PATH)
    svcIntercept    = loadBin(SVC_INTERCEPT_PATH)
    nbLogProb       = loadBin(NB_LOG_PROB_PATH)
    nbClassLogPrior = loadBin(NB_CLASS_LOG_PRIOR_PATH)
    hasAllModels    = true
  }

  if (fs.existsSync(MODEL_ACCURACY_PATH)) {
    try {
      accuracies = JSON.parse(fs.readFileSync(MODEL_ACCURACY_PATH, 'utf-8')) as AccuracyJson
    } catch { /* use zeros */ }
  }
}

// ---- Feature extraction -----------------------------------------------------

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

// Raw integer counts without TF-IDF scaling — needed for MultinomialNB
function charWbCounts(text: string): Float32Array {
  const vocab = vocabulary!

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

  return counts
}

// ---- Shared numerically-stable softmax --------------------------------------

function softmax(scores: Float32Array): Float32Array {
  let maxScore = scores[0]
  for (let c = 1; c < nClasses; c++) if (scores[c] > maxScore) maxScore = scores[c]

  const probs = new Float32Array(nClasses)
  let sum = 0
  for (let c = 0; c < nClasses; c++) {
    probs[c] = Math.exp(scores[c] - maxScore)
    sum += probs[c]
  }
  for (let c = 0; c < nClasses; c++) probs[c] /= sum
  return probs
}

// ---- Per-model inference ----------------------------------------------------

function linearPredict(features: Float32Array, w: Float32Array, b: Float32Array): Float32Array {
  const scores = new Float32Array(nClasses)
  for (let c = 0; c < nClasses; c++) {
    let dot = b[c]
    const offset = c * nFeatures
    for (let f = 0; f < nFeatures; f++) dot += w[offset + f] * features[f]
    scores[c] = dot
  }
  return softmax(scores)
}

function nbPredict(counts: Float32Array): Float32Array {
  const logProb      = nbLogProb!
  const classPrior   = nbClassLogPrior!
  const scores       = new Float32Array(nClasses)

  for (let c = 0; c < nClasses; c++) {
    let score = classPrior[c]
    const offset = c * nFeatures
    for (let f = 0; f < nFeatures; f++) {
      if (counts[f] > 0) score += counts[f] * logProb[offset + f]
    }
    scores[c] = score
  }
  return softmax(scores)
}

// ---- Probabilities → sorted Detection[] ------------------------------------

function probsToDetections(probs: Float32Array, topK: number): Detection[] {
  return Array.from(probs)
    .map((confidence, i) => {
      const lang = getByIso3(classes![i])
      return lang ? { language: lang, confidence } : null
    })
    .filter((d): d is Detection => d !== null)
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, topK)
}

// ---- Public API -------------------------------------------------------------

export function detectLanguage(text: string, topK = 3): Detection[] {
  loadModels()
  const features = charWbTfidf(text)
  const probs    = linearPredict(features, coef!, intercept!)
  return probsToDetections(probs, topK)
}

export function detectWithAllModels(
  text: string,
  topK = 3
): { topK: Detection[]; models?: ModelPrediction[] } {
  loadModels()

  const features = charWbTfidf(text)
  const lrProbs  = linearPredict(features, coef!, intercept!)
  const topKList = probsToDetections(lrProbs, topK)

  if (!hasAllModels) return { topK: topKList }

  const svcProbs = linearPredict(features, svcCoef!, svcIntercept!)
  const counts   = charWbCounts(text)
  const nbProbs  = nbPredict(counts)

  function topPred(probs: Float32Array): { iso3: string; confidence: number } {
    let best = 0
    for (let c = 1; c < nClasses; c++) if (probs[c] > probs[best]) best = c
    return { iso3: classes![best], confidence: probs[best] }
  }

  const acc = accuracies ?? { logistic_regression: 0, linear_svc: 0, naive_bayes: 0 }
  const lrTop  = topPred(lrProbs)
  const svcTop = topPred(svcProbs)
  const nbTop  = topPred(nbProbs)

  const models: ModelPrediction[] = [
    { model: 'logreg', topLanguageIso3: lrTop.iso3,  confidence: lrTop.confidence,  accuracy: acc.logistic_regression },
    { model: 'svc',    topLanguageIso3: svcTop.iso3, confidence: svcTop.confidence, accuracy: acc.linear_svc },
    { model: 'nb',     topLanguageIso3: nbTop.iso3,  confidence: nbTop.confidence,  accuracy: acc.naive_bayes },
  ]

  return { topK: topKList, models }
}
