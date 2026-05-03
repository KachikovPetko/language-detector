'use client'

import { useEffect, useReducer, useState } from 'react'
import { ExternalLink } from 'lucide-react'
import TextInput from '@/components/TextInput'
import FileUpload from '@/components/FileUpload'
import LiveRecorder from '@/components/LiveRecorder'
import ModeToggle from '@/components/ModeToggle'
import TargetLanguagePicker from '@/components/TargetLanguagePicker'
import DetectionResult from '@/components/DetectionResult'
import HistoryPanel from '@/components/HistoryPanel'
import type { AppState, InputMode, DetectApiResponse, TranslateApiResponse, HistoryItem } from '@/lib/types'
import { getByIso1 } from '@/lib/languages'

const HISTORY_KEY = 'lingualens_history'

type Action =
  | { type: 'SET_MODE';    mode: InputMode }
  | { type: 'SET_TARGET';  lang: string }
  | { type: 'DETECTING' }
  | { type: 'DETECTED';    detection: DetectApiResponse }
  | { type: 'TRANSLATING' }
  | { type: 'TRANSLATED';  translation: TranslateApiResponse }
  | { type: 'ERROR';       error: string }

const initial: AppState = {
  mode: 'text',
  inputText: '',
  targetLangIso3: 'eng',
  detection: null,
  translation: null,
  isDetecting: false,
  isTranslating: false,
  error: null,
}

function reducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case 'SET_MODE':    return { ...initial, mode: action.mode, targetLangIso3: state.targetLangIso3 }
    case 'SET_TARGET':  return { ...state, targetLangIso3: action.lang }
    case 'DETECTING':   return { ...state, isDetecting: true, error: null, detection: null, translation: null }
    case 'DETECTED':    return { ...state, isDetecting: false, detection: action.detection }
    case 'TRANSLATING': return { ...state, isTranslating: true }
    case 'TRANSLATED':  return { ...state, isTranslating: false, translation: action.translation }
    case 'ERROR':       return { ...state, isDetecting: false, isTranslating: false, error: action.error }
    default:            return state
  }
}

export default function Home() {
  const [state, dispatch] = useReducer(reducer, initial)
  const [history, setHistory] = useState<HistoryItem[]>([])

  useEffect(() => {
    try {
      const raw = localStorage.getItem(HISTORY_KEY)
      if (raw) setHistory(JSON.parse(raw) as HistoryItem[])
    } catch { /* localStorage unavailable */ }
  }, [])

  function saveHistory(item: HistoryItem) {
    setHistory(prev => {
      const updated = [item, ...prev.filter(h => h.text !== item.text)].slice(0, 10)
      try { localStorage.setItem(HISTORY_KEY, JSON.stringify(updated)) } catch { /* ignore */ }
      return updated
    })
  }

  function clearHistory() {
    setHistory([])
    try { localStorage.removeItem(HISTORY_KEY) } catch { /* ignore */ }
  }

  // whisperLang: ISO 639-1 code from Groq Whisper (audio/live modes).
  // Skips the ML model when present — more reliable on short speech transcripts.
  async function handleDetect(text: string, whisperLang?: string) {
    dispatch({ type: 'DETECTING' })

    let detection: DetectApiResponse

    const whisperLanguage = whisperLang ? getByIso1(whisperLang) : undefined
    if (whisperLanguage) {
      detection = {
        best: { language: whisperLanguage, confidence: 0.97 },
        topK: [{ language: whisperLanguage, confidence: 0.97 }],
      }
      dispatch({ type: 'DETECTED', detection })
    } else {
      try {
        const res = await fetch('/api/detect', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text }),
        })
        const data = (await res.json()) as DetectApiResponse & { error?: string }
        if (!res.ok) { dispatch({ type: 'ERROR', error: data.error ?? 'Detection failed' }); return }
        detection = data
        dispatch({ type: 'DETECTED', detection })
      } catch {
        dispatch({ type: 'ERROR', error: 'Network error during detection' })
        return
      }
    }

    const sourceLang = detection.best.language.iso3
    if (sourceLang === state.targetLangIso3) {
      dispatch({ type: 'TRANSLATED', translation: { naive: text, meaningAware: text } })
      saveHistory({
        id: Date.now().toString(),
        text: text.slice(0, 200),
        sourceLangIso3: sourceLang,
        targetLangIso3: state.targetLangIso3,
        meaningAware: text,
        timestamp: Date.now(),
      })
      return
    }

    dispatch({ type: 'TRANSLATING' })
    try {
      const res = await fetch('/api/translate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, sourceLang, targetLang: state.targetLangIso3 }),
      })
      const data = (await res.json()) as TranslateApiResponse & { error?: string }
      if (!res.ok) { dispatch({ type: 'ERROR', error: data.error ?? 'Translation failed' }); return }
      dispatch({ type: 'TRANSLATED', translation: data })
      saveHistory({
        id: Date.now().toString(),
        text: text.slice(0, 200),
        sourceLangIso3: sourceLang,
        targetLangIso3: state.targetLangIso3,
        meaningAware: data.meaningAware,
        timestamp: Date.now(),
      })
    } catch {
      dispatch({ type: 'ERROR', error: 'Network error during translation' })
    }
  }

  const isLoading = state.isDetecting || state.isTranslating

  return (
    <div className="flex min-h-screen flex-col">
      <header className="border-b px-6 py-4" style={{ borderColor: 'rgba(255,255,255,0.08)' }}>
        <div className="mx-auto flex max-w-2xl items-center justify-between">
          <div>
            <h1 className="text-xl font-bold text-white">
              Lingua
              <span style={{ background: 'linear-gradient(to right, #ff6b35, #f7931e)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                Lens
              </span>
            </h1>
            <p className="text-xs" style={{ color: 'rgba(255,255,255,0.4)' }}>ML-powered language detection &amp; translation</p>
          </div>
          <a href="https://github.com/KachikovPetko/language-detector" target="_blank" rel="noopener noreferrer"
            className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm transition"
            style={{ color: 'rgba(255,255,255,0.5)' }}>
            <ExternalLink className="h-4 w-4" />
            GitHub
          </a>
        </div>
      </header>

      <main className="mx-auto w-full max-w-2xl flex-1 px-6 py-10">
        <div className="flex flex-col gap-5">
          <ModeToggle value={state.mode} onChange={mode => dispatch({ type: 'SET_MODE', mode })} />
          <TargetLanguagePicker value={state.targetLangIso3} onChange={lang => dispatch({ type: 'SET_TARGET', lang })} />

          {state.mode === 'text'  && <TextInput    onSubmit={handleDetect} isLoading={isLoading} />}
          {state.mode === 'audio' && <FileUpload   onSubmit={handleDetect} isLoading={isLoading} />}
          {state.mode === 'live'  && <LiveRecorder onSubmit={handleDetect} isLoading={isLoading} />}

          {state.error && (
            <div className="rounded-xl px-4 py-3 text-sm"
              style={{ background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)', color: '#fca5a5' }}>
              {state.error}
            </div>
          )}

          {state.detection && (
            <DetectionResult
              detection={state.detection}
              translation={state.translation}
              isTranslating={state.isTranslating}
              targetLangIso3={state.targetLangIso3}
            />
          )}

          <HistoryPanel
            items={history}
            onSelect={text => handleDetect(text)}
            onClear={clearHistory}
          />
        </div>
      </main>

      <footer className="border-t py-4 text-center text-xs"
        style={{ borderColor: 'rgba(255,255,255,0.08)', color: 'rgba(255,255,255,0.25)' }}>
        University ML Project · TF-IDF char n-grams · Logistic Regression · 20 languages
      </footer>
    </div>
  )
}
