import { NextRequest, NextResponse } from 'next/server'

// Phase 2 — stub that returns a clear "not yet" message in Phase 1
export async function POST(_req: NextRequest): Promise<NextResponse> {
  return NextResponse.json(
    { error: 'Transcription endpoint is not available yet (Phase 2)' },
    { status: 501 }
  )
}
