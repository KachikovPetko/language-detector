'use client'

import type { DetectApiResponse, TranslateApiResponse } from '@/lib/types'
import { getByIso3 } from '@/lib/languages'

interface DetectionResultProps {
  detection:     DetectApiResponse
  translation:   TranslateApiResponse | null
  isTranslating: boolean
  targetLangIso3: string
}

// Returns tokens from `meaningful` annotated as changed/unchanged vs `naive`.
// A token is "changed" if the lowercased word doesn't appear anywhere in naive.
function diffTokens(naive: string, meaningful: string): { text: string; changed: boolean }[] {
  const naiveWords = new Set((naive.match(/\w+/g) ?? []).map(w => w.toLowerCase()))
  const tokens = meaningful.match(/\S+|\s+/g) ?? []
  return tokens.map(tok => {
    const word = tok.match(/\w+/)?.[0]?.toLowerCase()
    return { text: tok, changed: !!word && !naiveWords.has(word) }
  })
}

export default function DetectionResult({
  detection,
  translation,
  isTranslating,
  targetLangIso3,
}: DetectionResultProps) {
  const { best } = detection
  const targetLang   = getByIso3(targetLangIso3)
  const confidencePct = (best.confidence * 100).toFixed(1)

  const showDiff = translation && translation.naive !== translation.meaningAware
  const diffed   = showDiff ? diffTokens(translation.naive, translation.meaningAware) : null

  return (
    <div className="flex flex-col gap-4 rounded-2xl border border-white/10 bg-white/5 p-6">

      {/* Detected language */}
      <div className="flex items-center gap-3">
        <span className="text-4xl">{best.language.flag}</span>
        <div>
          <p className="text-lg font-semibold text-white">{best.language.name}</p>
          <p className="text-sm text-white/50">{confidencePct}% confidence</p>
        </div>
      </div>

      {/* Confidence bar */}
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/10">
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{
            width: `${confidencePct}%`,
            background: 'linear-gradient(to right, #ff6b35, #f7931e)',
          }}
        />
      </div>

      {/* Runner-up hints */}
      {detection.topK.length > 1 && (
        <p className="text-xs text-white/40">
          Also possible:{' '}
          {detection.topK
            .slice(1)
            .map(d => `${d.language.flag} ${d.language.name} (${(d.confidence * 100).toFixed(1)}%)`)
            .join(', ')}
        </p>
      )}

      {/* Translation section */}
      <div className="border-t border-white/10 pt-4 flex flex-col gap-3">
        <p className="text-sm font-medium text-white/60">
          Translation → {targetLang ? `${targetLang.flag} ${targetLang.name}` : targetLangIso3}
        </p>

        {isTranslating ? (
          <div className="flex items-center gap-2 text-white/40">
            <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-white/20 border-t-[#ff6b35]" />
            Translating…
          </div>
        ) : translation ? (
          showDiff ? (
            /* Two-panel diff view */
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">

              {/* Naive / word-by-word panel */}
              <div className="flex flex-col gap-1.5 rounded-xl p-3"
                style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.08)' }}>
                <p className="text-[11px] font-semibold uppercase tracking-wide text-white/30">
                  Word-by-word (literal)
                </p>
                <p className="text-sm leading-relaxed text-white/60">{translation.naive}</p>
              </div>

              {/* Meaning-aware panel with diff highlights */}
              <div className="flex flex-col gap-1.5 rounded-xl p-3"
                style={{ background: 'rgba(255,107,53,0.07)', border: '1px solid rgba(255,107,53,0.2)' }}>
                <p className="text-[11px] font-semibold uppercase tracking-wide"
                  style={{ color: '#ff6b35' }}>
                  Meaning-aware
                </p>
                <p className="text-sm leading-relaxed text-white">
                  {diffed!.map((tok, i) =>
                    tok.changed ? (
                      <mark key={i}
                        style={{
                          background: 'rgba(255,107,53,0.25)',
                          color: '#f7931e',
                          borderRadius: '3px',
                          padding: '0 2px',
                        }}>
                        {tok.text}
                      </mark>
                    ) : (
                      <span key={i}>{tok.text}</span>
                    )
                  )}
                </p>
              </div>

              {/* Legend */}
              <p className="sm:col-span-2 text-[11px] text-white/25">
                <mark style={{ background: 'rgba(255,107,53,0.25)', color: '#f7931e', borderRadius: '3px', padding: '0 2px' }}>
                  highlighted
                </mark>
                {' '}= words chosen for meaning/context, not literal substitution
              </p>
            </div>
          ) : (
            /* Same-language or identical result — single panel */
            <p className="text-base leading-relaxed text-white">{translation.meaningAware}</p>
          )
        ) : (
          <p className="text-sm italic text-white/30">Translation unavailable</p>
        )}
      </div>
    </div>
  )
}
