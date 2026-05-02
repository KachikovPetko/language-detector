import { NextRequest, NextResponse } from 'next/server'
import { transcribeAudio } from '@/lib/groq'

const MAX_BYTES = 25 * 1024 * 1024  // 25 MB — Vercel free tier enforces 4.5 MB in practice

const ACCEPTED_TYPES = new Set([
  'audio/mpeg', 'audio/mp3', 'audio/wav', 'audio/x-wav', 'audio/wave',
  'audio/mp4', 'audio/m4a', 'audio/x-m4a', 'audio/aac',
  'audio/ogg', 'audio/flac', 'audio/x-flac',
  'audio/webm', 'video/webm',  // browsers record as video/webm even for audio-only
])

export async function POST(req: NextRequest): Promise<NextResponse> {
  if (!process.env.GROQ_API_KEY) {
    return NextResponse.json({ error: 'GROQ_API_KEY is not configured' }, { status: 503 })
  }

  let formData: FormData
  try {
    formData = await req.formData()
  } catch {
    return NextResponse.json({ error: 'Expected multipart/form-data' }, { status: 400 })
  }

  const file = formData.get('file')
  if (!(file instanceof File)) {
    return NextResponse.json({ error: 'No audio file provided (field name: "file")' }, { status: 400 })
  }

  if (file.size > MAX_BYTES) {
    return NextResponse.json(
      { error: `File too large (${(file.size / 1048576).toFixed(1)} MB). Max 25 MB.` },
      { status: 413 }
    )
  }

  // Allow audio/* and video/webm (browser MediaRecorder output)
  const mime = file.type || 'audio/mpeg'
  if (!ACCEPTED_TYPES.has(mime) && !mime.startsWith('audio/')) {
    return NextResponse.json(
      { error: `Unsupported file type: ${mime}. Upload an audio file (mp3, wav, m4a, ogg, flac, webm).` },
      { status: 415 }
    )
  }

  try {
    const { text, whisperLang } = await transcribeAudio(file)
    if (!text) {
      return NextResponse.json({ error: 'Transcription returned empty text' }, { status: 422 })
    }
    return NextResponse.json({ text, detectedLang: whisperLang })
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Transcription failed'
    return NextResponse.json({ error: message }, { status: 500 })
  }
}
