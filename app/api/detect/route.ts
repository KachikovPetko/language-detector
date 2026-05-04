import { NextRequest, NextResponse } from 'next/server'
import { detectWithAllModels } from '@/lib/detector'
import type { DetectApiRequest, DetectApiResponse } from '@/lib/types'

export async function POST(req: NextRequest): Promise<NextResponse> {
  let body: DetectApiRequest
  try {
    body = (await req.json()) as DetectApiRequest
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 })
  }

  const text = body.text?.trim()
  if (!text) {
    return NextResponse.json({ error: 'text is required' }, { status: 400 })
  }
  if (text.length > 10_000) {
    return NextResponse.json({ error: 'text must be under 10 000 characters' }, { status: 400 })
  }

  try {
    const { topK, models } = detectWithAllModels(text, 3)
    if (topK.length === 0) {
      return NextResponse.json({ error: 'Detection returned no results' }, { status: 500 })
    }
    const response: DetectApiResponse = { best: topK[0], topK, models }
    return NextResponse.json(response)
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Detection failed'
    const status = message.includes('not found') ? 503 : 500
    return NextResponse.json({ error: message }, { status })
  }
}
