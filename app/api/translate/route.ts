import { NextRequest, NextResponse } from 'next/server'
import { translateMeaningAware } from '@/lib/groq'
import { getByIso3 } from '@/lib/languages'
import type { TranslateApiRequest, TranslateApiResponse } from '@/lib/types'

export async function POST(req: NextRequest): Promise<NextResponse> {
  let body: TranslateApiRequest
  try {
    body = (await req.json()) as TranslateApiRequest
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 })
  }

  const { text, sourceLang, targetLang } = body
  if (!text?.trim()) {
    return NextResponse.json({ error: 'text is required' }, { status: 400 })
  }
  if (!getByIso3(sourceLang)) {
    return NextResponse.json({ error: `Unknown sourceLang: ${sourceLang}` }, { status: 400 })
  }
  if (!getByIso3(targetLang)) {
    return NextResponse.json({ error: `Unknown targetLang: ${targetLang}` }, { status: 400 })
  }
  if (!process.env.GROQ_API_KEY) {
    return NextResponse.json({ error: 'GROQ_API_KEY is not configured' }, { status: 503 })
  }

  try {
    const meaningAware = await translateMeaningAware(text.trim(), sourceLang, targetLang)
    const response: TranslateApiResponse = { meaningAware }
    return NextResponse.json(response)
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Translation failed'
    return NextResponse.json({ error: message }, { status: 500 })
  }
}
