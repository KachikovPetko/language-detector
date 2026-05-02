'use client'

import { useEffect, useRef, useState } from 'react'
import { Mic, Square, Loader2, RotateCcw } from 'lucide-react'

interface LiveRecorderProps {
  onSubmit: (text: string) => void
  isLoading: boolean
}

type RecordState = 'idle' | 'recording' | 'transcribing' | 'done'

export default function LiveRecorder({ onSubmit, isLoading }: LiveRecorderProps) {
  const streamRef   = useRef<MediaStream | null>(null)
  const recorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef   = useRef<Blob[]>([])
  const audioCtxRef = useRef<AudioContext | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const rafRef      = useRef<number>(0)

  const [recordState, setRecordState] = useState<RecordState>('idle')
  const [transcript,  setTranscript]  = useState('')
  const [volume,      setVolume]      = useState(0)   // 0–100
  const [error,       setError]       = useState<string | null>(null)

  useEffect(() => () => teardown(), [])

  function teardown() {
    cancelAnimationFrame(rafRef.current)
    streamRef.current?.getTracks().forEach(t => t.stop())
    streamRef.current = null
    try { audioCtxRef.current?.close() } catch { /* ignore */ }
    audioCtxRef.current = null
    analyserRef.current = null
    setVolume(0)
  }

  function startVolumeMeter(stream: MediaStream) {
    try {
      const ctx    = new AudioContext()
      const source = ctx.createMediaStreamSource(stream)
      const node   = ctx.createAnalyser()
      node.fftSize = 256
      source.connect(node)
      audioCtxRef.current = ctx
      analyserRef.current = node
      const buf = new Uint8Array(node.frequencyBinCount)
      function tick() {
        node.getByteTimeDomainData(buf)
        const rms = Math.sqrt(buf.reduce((s, v) => s + (v - 128) ** 2, 0) / buf.length)
        setVolume(Math.min(100, rms * 3))
        rafRef.current = requestAnimationFrame(tick)
      }
      tick()
    } catch { /* AudioContext unavailable */ }
  }

  async function startRecording() {
    setError(null)

    let stream: MediaStream
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    } catch {
      setError('Microphone access denied — check browser and OS permissions, then try again.')
      return
    }
    streamRef.current = stream
    startVolumeMeter(stream)

    const recorder = new MediaRecorder(stream)
    chunksRef.current = []

    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data)
    }

    recorder.onstop = async () => {
      teardown()
      setRecordState('transcribing')

      const mimeType = recorder.mimeType || 'audio/webm'
      const ext      = mimeType.includes('ogg') ? 'ogg' : mimeType.includes('mp4') ? 'mp4' : 'webm'
      const blob     = new Blob(chunksRef.current, { type: mimeType })

      const fd = new FormData()
      fd.append('file', blob, `recording.${ext}`)

      try {
        const res  = await fetch('/api/transcribe', { method: 'POST', body: fd })
        const data = await res.json() as { text?: string; error?: string }
        if (!res.ok || !data.text) {
          setError(data.error ?? 'Transcription failed — try again')
          setRecordState('idle')
          return
        }
        setTranscript(data.text)
        setRecordState('done')
        onSubmit(data.text)
      } catch {
        setError('Network error during transcription')
        setRecordState('idle')
      }
    }

    recorder.start()
    recorderRef.current = recorder
    setTranscript('')
    setRecordState('recording')
  }

  function stopAndTranslate() {
    recorderRef.current?.stop()
    recorderRef.current = null
  }

  function reset() {
    recorderRef.current?.stop()
    recorderRef.current = null
    teardown()
    chunksRef.current = []
    setTranscript('')
    setError(null)
    setRecordState('idle')
  }

  // 12 volume bars
  const bars = Array.from({ length: 12 }, (_, i) => ({
    lit:    volume > (i / 12) * 100,
    height: 5 + i * 2,
  }))

  return (
    <div className="flex flex-col items-center gap-5">

      {/* Volume meter — real mic level via AudioContext */}
      {recordState === 'recording' && (
        <div className="flex items-end gap-[3px] h-8">
          {bars.map((b, i) => (
            <div key={i}
              className="w-1.5 rounded-sm transition-all duration-75"
              style={{
                height: `${b.height}px`,
                background: b.lit ? '#ff6b35' : 'rgba(255,255,255,0.12)',
              }}
            />
          ))}
        </div>
      )}

      {/* Record button */}
      <div className="relative mt-2">
        {recordState === 'recording' && (
          <span className="absolute inset-0 rounded-full animate-ping"
            style={{ background: 'rgba(239,68,68,0.35)' }} />
        )}
        <button
          type="button"
          onClick={recordState === 'recording' ? stopAndTranslate : startRecording}
          disabled={isLoading || recordState === 'transcribing'}
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
          {recordState === 'recording'    && <Square  className="h-8 w-8 fill-white text-white" />}
          {recordState === 'transcribing' && <Loader2 className="h-8 w-8 animate-spin text-white" />}
          {(recordState === 'idle' || recordState === 'done') && <Mic className="h-9 w-9 text-white" />}
        </button>
      </div>

      {/* Status */}
      <p className="text-sm" style={{ color: 'rgba(255,255,255,0.45)' }}>
        {recordState === 'idle'         && 'Click to start recording'}
        {recordState === 'recording'    && '● Recording — click to stop & transcribe'}
        {recordState === 'transcribing' && 'Transcribing with Whisper…'}
        {recordState === 'done'         && 'Transcription complete'}
      </p>

      {/* Transcript */}
      {transcript && (
        <div className="w-full rounded-xl px-4 py-3 text-sm leading-relaxed"
          style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)' }}>
          <p className="mb-1 text-xs font-medium" style={{ color: 'rgba(255,255,255,0.4)' }}>Transcript</p>
          <span className="text-white">{transcript}</span>
        </div>
      )}

      {/* Error */}
      {error && (
        <p className="w-full rounded-xl px-4 py-3 text-sm"
          style={{ background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)', color: '#fca5a5' }}>
          {error}
        </p>
      )}

      {/* Stop & Translate (secondary) */}
      {recordState === 'recording' && (
        <button
          type="button"
          onClick={stopAndTranslate}
          className="flex items-center gap-2 rounded-xl px-8 py-3 font-semibold text-white transition-all active:scale-[0.98]"
          style={{ background: '#ef4444' }}
        >
          <Square className="h-4 w-4 fill-white" />
          Stop &amp; Transcribe
        </button>
      )}

      {/* Post-done */}
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
