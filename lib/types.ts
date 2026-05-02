export interface Language {
  iso1: string  // ISO 639-1 (e.g. 'en') — used for browser TTS lang tag
  iso3: string  // ISO 639-3 (e.g. 'eng') — matches ONNX model output labels
  name: string
  flag: string
}

export interface Detection {
  language: Language
  confidence: number
}

export interface DetectApiRequest {
  text: string
}

export interface DetectApiResponse {
  best: Detection
  topK: Detection[]
}

export interface TranslateApiRequest {
  text: string
  sourceLang: string  // iso3
  targetLang: string  // iso3
}

export interface TranslateApiResponse {
  meaningAware: string
}

export type InputMode = 'text' | 'audio' | 'live'

export interface AppState {
  mode: InputMode
  inputText: string
  targetLangIso3: string
  detection: DetectApiResponse | null
  translation: TranslateApiResponse | null
  isDetecting: boolean
  isTranslating: boolean
  error: string | null
}
