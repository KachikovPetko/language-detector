import * as ort from 'onnxruntime-node'
import path from 'path'
import fs from 'fs'
import { getByIso3 } from './languages'
import type { Detection } from './types'

// ---- Model file paths -------------------------------------------------------

const MODEL_DIR   = path.join(process.cwd(), 'public', 'model')
const ONNX_PATH   = path.join(MODEL_DIR, 'detector.onnx')
const VOCAB_PATH  = path.join(MODEL_DIR, 'vocab.json')
const LABELS_PATH = path.join(MODEL_DIR, 'labels.json')

// ---- Types ------------------------------------------------------------------

interface VocabJson {
  vocabulary: Record<string, number>
  idf: number[]
}

interface LabelsJson {
  classes: string[]
}

// ---- Module-level cache — survives across requests in the same Lambda -------

let session: ort.InferenceSession | null = null
let vocabulary: Record<string, number> | null = null
let idf: Float32Array | null = null
let classes: string[] | null = null
let nFeatures = 0

function ensureModelsExist() {
  if (!fs.existsSync(ONNX_PATH) || !fs.existsSync(VOCAB_PATH) || !fs.existsSync(LABELS_PATH)) {
    throw new Error(
      'Model files not found in public/model/. ' +
      'Run ml/train.py then ml/export_onnx.py first.'
    )
  }
}

async function loadModels() {
  if (session && vocabulary && idf && classes) return

  ensureModelsExist()

  const vocabRaw = JSON.parse(fs.readFileSync(VOCAB_PATH, 'utf-8')) as VocabJson
  vocabulary = vocabRaw.vocabulary
  idf = new Float32Array(vocabRaw.idf)
  nFeatures = idf.length

  const labelsRaw = JSON.parse(fs.readFileSync(LABELS_PATH, 'utf-8')) as LabelsJson
  classes = labelsRaw.classes

  session = await ort.InferenceSession.create(ONNX_PATH)
}

// ---- TF-IDF char_wb transform -----------------------------------------------
// Replicates sklearn's TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4),
// sublinear_tf=True, norm='l2')

function charWbTfidf(text: string): Float32Array {
  const vocab = vocabulary!
  const idfArr = idf!

  const textLower = text.toLowerCase()
  const tokens = textLower.split(/\s+/).filter(t => t.length > 0)

  // Count char n-gram occurrences
  const counts = new Float32Array(nFeatures)
  for (const token of tokens) {
    const padded = ` ${token} `
    for (let n = 2; n <= 4; n++) {
      for (let i = 0; i <= padded.length - n; i++) {
        const ngram = padded.slice(i, i + n)
        const idx = vocab[ngram]
        if (idx !== undefined) counts[idx]++
      }
    }
  }

  // sublinear_tf=True: tf = 1 + log(tf) for tf > 0
  // multiply by IDF
  const features = new Float32Array(nFeatures)
  let norm = 0
  for (let i = 0; i < nFeatures; i++) {
    if (counts[i] > 0) {
      const tf = 1 + Math.log(counts[i])
      features[i] = tf * idfArr[i]
      norm += features[i] * features[i]
    }
  }

  // L2 normalise
  if (norm > 0) {
    const sqrtNorm = Math.sqrt(norm)
    for (let i = 0; i < nFeatures; i++) features[i] /= sqrtNorm
  }

  return features
}

// ---- Public API -------------------------------------------------------------

export async function detectLanguage(text: string, topK = 3): Promise<Detection[]> {
  await loadModels()

  const features = charWbTfidf(text)
  const inputTensor = new ort.Tensor('float32', features, [1, nFeatures])
  const results = await session!.run({ float_input: inputTensor })

  // 'probabilities' is Float32Array[num_classes] in same order as classes_
  const probData = results['probabilities'].data as Float32Array

  return Array.from(probData)
    .map((confidence, i) => {
      const lang = getByIso3(classes![i])
      return lang ? { language: lang, confidence } : null
    })
    .filter((d): d is Detection => d !== null)
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, topK)
}
