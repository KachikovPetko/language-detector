'use client'

import { useRef, useState } from 'react'
import { Upload, X, FileAudio, Loader2, CheckCircle } from 'lucide-react'

const MAX_MB = 25
const ACCEPTED = '.mp3,.wav,.m4a,.aac,.ogg,.flac,.webm'

interface FileUploadProps {
  onSubmit: (text: string, whisperLang?: string) => void
  isLoading: boolean
}

type Status = 'idle' | 'transcribing' | 'done' | 'error'

export default function FileUpload({ onSubmit, isLoading }: FileUploadProps) {
  const inputRef      = useRef<HTMLInputElement>(null)
  const whisperLangRef = useRef('')
  const [isDragging, setIsDragging] = useState(false)
  const [file, setFile]             = useState<File | null>(null)
  const [status, setStatus]         = useState<Status>('idle')
  const [transcript, setTranscript] = useState('')
  const [errorMsg, setErrorMsg]     = useState('')

  function reset() {
    setFile(null); setStatus('idle'); setTranscript(''); setErrorMsg('')
    if (inputRef.current) inputRef.current.value = ''
  }

  async function handleFile(f: File) {
    if (f.size > MAX_MB * 1024 * 1024) {
      setFile(f); setStatus('error')
      setErrorMsg(`File is ${(f.size / 1048576).toFixed(1)} MB — max ${MAX_MB} MB`)
      return
    }
    setFile(f); setStatus('transcribing'); setTranscript(''); setErrorMsg('')

    const fd = new FormData()
    fd.append('file', f)
    try {
      const res = await fetch('/api/transcribe', { method: 'POST', body: fd })
      const data = (await res.json()) as { text?: string; detectedLang?: string; error?: string }
      if (!res.ok) { setStatus('error'); setErrorMsg(data.error ?? 'Transcription failed'); return }
      setTranscript(data.text ?? ''); setStatus('done')
      // stash whisperLang so the Detect & Translate button can forward it
      whisperLangRef.current = data.detectedLang ?? ''
    } catch {
      setStatus('error'); setErrorMsg('Network error during transcription')
    }
  }

  function onDrop(e: React.DragEvent) {
    e.preventDefault(); setIsDragging(false)
    const f = e.dataTransfer.files[0]
    if (f) handleFile(f)
  }

  const sizeLabel = file ? `${(file.size / 1048576).toFixed(1)} MB` : ''

  return (
    <div className="flex flex-col gap-3">
      {/* Drop zone — hidden once a file is loaded */}
      {!file && (
        <div
          onDragOver={e => { e.preventDefault(); setIsDragging(true) }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={onDrop}
          onClick={() => inputRef.current?.click()}
          className="flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed py-10 text-center transition-all"
          style={{
            borderColor: isDragging ? '#ff6b35' : 'rgba(255,255,255,0.12)',
            background:  isDragging ? 'rgba(255,107,53,0.06)' : 'rgba(255,255,255,0.03)',
          }}
        >
          <Upload className="h-7 w-7" style={{ color: 'rgba(255,255,255,0.3)' }} />
          <p className="text-sm font-medium text-white">Drop an audio file or <span style={{ color: '#ff6b35' }}>browse</span></p>
          <p className="text-xs" style={{ color: 'rgba(255,255,255,0.3)' }}>MP3, WAV, M4A, OGG, FLAC, WebM · max {MAX_MB} MB</p>
        </div>
      )}

      <input ref={inputRef} type="file" accept={ACCEPTED} className="hidden"
        onChange={e => { const f = e.target.files?.[0]; if (f) handleFile(f) }} />

      {/* File info row */}
      {file && (
        <div className="flex items-center gap-3 rounded-xl px-4 py-3"
          style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.08)' }}>
          <FileAudio className="h-5 w-5 shrink-0" style={{ color: '#ff6b35' }} />
          <div className="min-w-0 flex-1">
            <p className="truncate text-sm font-medium text-white">{file.name}</p>
            <p className="text-xs" style={{ color: 'rgba(255,255,255,0.4)' }}>{sizeLabel}</p>
          </div>
          <button type="button" onClick={reset}
            className="rounded-lg p-1 transition hover:bg-white/10"
            style={{ color: 'rgba(255,255,255,0.4)' }}>
            <X className="h-4 w-4" />
          </button>
        </div>
      )}

      {/* Progress / transcript */}
      {status === 'transcribing' && (
        <div className="flex items-center gap-2 text-sm" style={{ color: 'rgba(255,255,255,0.5)' }}>
          <Loader2 className="h-4 w-4 animate-spin" style={{ color: '#ff6b35' }} />
          Transcribing with Whisper…
          {/* Indeterminate bar */}
          <div className="ml-auto h-1 w-24 overflow-hidden rounded-full" style={{ background: 'rgba(255,255,255,0.1)' }}>
            <div className="h-full w-1/2 rounded-full animate-pulse" style={{ background: 'linear-gradient(to right, #ff6b35, #f7931e)' }} />
          </div>
        </div>
      )}

      {status === 'error' && (
        <p className="rounded-xl px-4 py-3 text-sm"
          style={{ background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)', color: '#fca5a5' }}>
          {errorMsg}
        </p>
      )}

      {status === 'done' && (
        <div className="flex flex-col gap-2 rounded-xl px-4 py-3"
          style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)' }}>
          <div className="flex items-center gap-1.5 text-xs font-medium" style={{ color: 'rgba(255,255,255,0.5)' }}>
            <CheckCircle className="h-3.5 w-3.5" style={{ color: '#4ade80' }} />
            Transcript
          </div>
          <p className="text-sm leading-relaxed text-white">{transcript}</p>
        </div>
      )}

      {/* Detect & Translate button — enabled only after transcription */}
      {status === 'done' && (
        <button
          type="button"
          onClick={() => onSubmit(transcript, whisperLangRef.current || undefined)}
          disabled={isLoading}
          className="flex items-center justify-center gap-2 rounded-xl px-6 py-3 font-semibold text-white transition-all active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-40"
          style={{ background: 'linear-gradient(to right, #ff6b35, #f7931e)' }}
        >
          {isLoading ? (
            <><Loader2 className="h-4 w-4 animate-spin" /> Detecting…</>
          ) : (
            'Detect & Translate'
          )}
        </button>
      )}
    </div>
  )
}
