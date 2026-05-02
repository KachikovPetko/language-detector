'use client'

import { useEffect, useRef, useState } from 'react'
import { Mic, Square, Loader2, RotateCcw } from 'lucide-react'

// Minimal Web Speech API types — not shipped in every TS dom lib configuration
interface ISpeechRecognitionAlternative { transcript: string }
interface ISpeechRecognitionResult {
  isFinal: boolean
  readonly length: number
  [index: number]: ISpeechRecognitionAlternative
}
interface ISpeechRecognitionResultList {
  readonly length: number
  [index: number]: ISpeechRecognitionResult
}
interface ISpeechRecognitionEvent extends Event {
  readonly resultIndex: number
  readonly results: ISpeechRecognitionResultList
}
interface ISpeechRecognitionErrorEvent extends Event { error: string }
interface ISpeechRecognition extends EventTarget {
  continuous: boolean
  interimResults: boolean
  start(): void
  stop(): void
  abort(): void
  onresult:  ((e: ISpeechRecognitionEvent)      => void) | null
  onerror:   ((e: ISpeechRecognitionErrorEvent) => void) | null
  onend:     (() => void) | null
}
type SpeechRecognitionCtor = new () => ISpeechRecognition

function getSpeechRecognitionCtor(): SpeechRecognitionCtor | null {
  if (typeof window === 'undefined') return null
  const w = window as unknown as Record<string, unknown>
  return (w['SpeechRecognition'] ?? w['webkitSpeechRecognition'] ?? null) as SpeechRecognitionCtor | null
}

// ---- Component --------------------------------------------------------------

interface LiveRecorderProps {
  onSubmit: (text: string) => void
  isLoading: boolean
}

type RecordState = 'idle' | 'recording' | 'done'

export default function LiveRecorder({ onSubmit, isLoading }: LiveRecorderProps) {
  const recognitionRef = useRef<ISpeechRecognition | null>(null)
  // Refs prevent stale closures inside SpeechRecognition event callbacks
  const finalTextRef   = useRef('')
  const isActiveRef    = useRef(false)

  const [recordState, setRecordState] = useState<RecordState>('idle')
  const [finalText,   setFinalText]   = useState('')
  const [interimText, setInterimText] = useState('')
  const [error,       setError]       = useState<string | null>(null)
  const [supported,   setSupported]   = useState(true)

  useEffect(() => {
    setSupported(getSpeechRecognitionCtor() !== null)
    return () => { recognitionRef.current?.abort() }
  }, [])

  function buildRecognition(): ISpeechRecognition {
    const Ctor = getSpeechRecognitionCtor()!
    const rec = new Ctor()
    rec.continuous     = true
    rec.interimResults = true

    rec.onresult = (e: ISpeechRecognitionEvent) => {
      let interim = ''
      for (let i = e.resultIndex; i < e.results.length; i++) {
        if (e.results[i].isFinal) finalTextRef.current += e.results[i][0].transcript + ' '
        else interim = e.results[i][0].transcript
      }
      setFinalText(finalTextRef.current)
      setInterimText(interim)
    }

    rec.onerror = (e: ISpeechRecognitionErrorEvent) => {
      if (e.error === 'no-speech') return
      isActiveRef.current = false
      setRecordState('idle')
      setError(
        e.error === 'not-allowed'
          ? 'Microphone access denied — allow permission and try again.'
          : `Speech recognition error: ${e.error}`
      )
    }

    // Some browsers auto-stop on silence even with continuous=true; restart if still active
    rec.onend = () => {
      if (isActiveRef.current) {
        try { rec.start() } catch { /* stopped externally */ }
      }
    }

    return rec
  }

  function startRecording() {
    const rec = buildRecognition()
    finalTextRef.current   = ''
    isActiveRef.current    = true
    recognitionRef.current = rec
    setFinalText(''); setInterimText(''); setError(null)
    setRecordState('recording')
    rec.start()
  }

  function stopAndTranslate() {
    isActiveRef.current = false
    recognitionRef.current?.stop()
    const text = (finalTextRef.current + interimText).trim()
    setInterimText('')
    setRecordState('done')
    if (text) onSubmit(text)
  }

  function reset() {
    isActiveRef.current = false
    recognitionRef.current?.abort()
    finalTextRef.current = ''
    setFinalText(''); setInterimText(''); setError(null)
    setRecordState('idle')
  }

  const hasText = finalText || interimText

  if (!supported) {
    return (
      <div className="flex flex-col items-center gap-2 rounded-xl py-10 text-center"
        style={{ background: 'rgba(255,255,255,0.03)', border: '1px dashed rgba(255,255,255,0.12)' }}>
        <p className="text-sm font-medium text-white">Web Speech API not supported</p>
        <p className="text-xs" style={{ color: 'rgba(255,255,255,0.35)' }}>Use Chrome or Edge for live recording</p>
      </div>
    )
  }

  return (
    <div className="flex flex-col items-center gap-5">
      {/* Large circular record button */}
      <div className="relative mt-2">
        {recordState === 'recording' && (
          <span className="absolute inset-0 rounded-full animate-ping"
            style={{ background: 'rgba(239,68,68,0.35)' }} />
        )}
        <button
          type="button"
          onClick={recordState === 'recording' ? undefined : startRecording}
          disabled={isLoading}
          className="relative flex h-24 w-24 items-center justify-center rounded-full transition-transform active:scale-95 disabled:opacity-40"
          style={{
            background: recordState === 'recording'
              ? '#ef4444'
              : 'linear-gradient(135deg, #ff6b35, #f7931e)',
            boxShadow: recordState === 'recording'
              ? '0 0 0 4px rgba(239,68,68,0.25)'
              : '0 0 0 4px rgba(255,107,53,0.2)',
          }}
        >
          <Mic className="h-9 w-9 text-white" />
        </button>
      </div>

      {/* Status label */}
      <p className="text-sm" style={{ color: 'rgba(255,255,255,0.45)' }}>
        {recordState === 'idle'      && 'Click to start recording'}
        {recordState === 'recording' && '● Listening — speak now'}
        {recordState === 'done'      && 'Recording stopped'}
      </p>

      {/* Live transcript */}
      {(hasText || recordState === 'recording') && (
        <div className="w-full rounded-xl px-4 py-3 text-sm leading-relaxed"
          style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)', minHeight: '72px' }}>
          {hasText ? (
            <>
              <span className="text-white">{finalText}</span>
              <span className="italic" style={{ color: 'rgba(255,255,255,0.4)' }}>{interimText}</span>
            </>
          ) : (
            <span className="italic" style={{ color: 'rgba(255,255,255,0.25)' }}>Transcript will appear here…</span>
          )}
        </div>
      )}

      {/* Error */}
      {error && (
        <p className="w-full rounded-xl px-4 py-3 text-sm"
          style={{ background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)', color: '#fca5a5' }}>
          {error}
        </p>
      )}

      {/* Stop & Translate — big red button while recording */}
      {recordState === 'recording' && (
        <button
          type="button"
          onClick={stopAndTranslate}
          className="flex items-center gap-2 rounded-xl px-8 py-3 font-semibold text-white transition-all active:scale-[0.98]"
          style={{ background: '#ef4444' }}
        >
          <Square className="h-4 w-4 fill-white" />
          Stop &amp; Translate
        </button>
      )}

      {/* Post-recording */}
      {recordState === 'done' && (
        isLoading ? (
          <span className="flex items-center gap-2 text-sm" style={{ color: 'rgba(255,255,255,0.45)' }}>
            <Loader2 className="h-4 w-4 animate-spin" />
            Detecting &amp; translating…
          </span>
        ) : (
          <button
            type="button"
            onClick={reset}
            className="flex items-center gap-1.5 rounded-lg px-4 py-2 text-sm transition"
            style={{ color: 'rgba(255,255,255,0.5)' }}
          >
            <RotateCcw className="h-3.5 w-3.5" />
            Record again
          </button>
        )
      )}
    </div>
  )
}
