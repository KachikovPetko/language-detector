'use client'

import type { DetectApiResponse, TranslateApiResponse } from '@/lib/types'
import { getByIso3 } from '@/lib/languages'

interface DetectionResultProps {
  detection: DetectApiResponse
  translation: TranslateApiResponse | null
  isTranslating: boolean
  targetLangIso3: string
}

export default function DetectionResult({
  detection,
  translation,
  isTranslating,
  targetLangIso3,
}: DetectionResultProps) {
  const { best } = detection
  const targetLang = getByIso3(targetLangIso3)
  const confidencePct = (best.confidence * 100).toFixed(1)

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
          className="h-full rounded-full bg-gradient-to-r from-[#ff6b35] to-[#f7931e] transition-all duration-700"
          style={{ width: `${confidencePct}%` }}
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

      {/* Translation */}
      <div className="border-t border-white/10 pt-4">
        <p className="mb-2 text-sm font-medium text-white/60">
          Translation → {targetLang ? `${targetLang.flag} ${targetLang.name}` : targetLangIso3}
        </p>
        {isTranslating ? (
          <div className="flex items-center gap-2 text-white/40">
            <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-white/20 border-t-[#ff6b35]" />
            Translating…
          </div>
        ) : translation ? (
          <p className="text-base leading-relaxed text-white">{translation.meaningAware}</p>
        ) : (
          <p className="text-sm text-white/30 italic">Translation unavailable</p>
        )}
      </div>
    </div>
  )
}
