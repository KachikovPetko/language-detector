'use client'

import { useState } from 'react'
import { Volume2, VolumeX } from 'lucide-react'
import {
  BarChart, Bar, XAxis, YAxis, Cell, ResponsiveContainer, Tooltip,
} from 'recharts'
import type { DetectApiResponse, TranslateApiResponse } from '@/lib/types'
import { getByIso3 } from '@/lib/languages'

interface DetectionResultProps {
  detection:      DetectApiResponse
  translation:    TranslateApiResponse | null
  isTranslating:  boolean
  targetLangIso3: string
}

// Words in `meaningful` absent from `naive` word set are marked changed.
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
  const targetLang    = getByIso3(targetLangIso3)
  const confidencePct = (best.confidence * 100).toFixed(1)

  const [speaking, setSpeaking] = useState(false)

  const chartData = detection.topK.map(d => ({
    name: `${d.language.flag} ${d.language.name}`,
    pct:  parseFloat((d.confidence * 100).toFixed(1)),
  }))

  const showDiff = translation && translation.naive !== translation.meaningAware
  const diffed   = showDiff ? diffTokens(translation.naive, translation.meaningAware) : null

  function toggleTts() {
    if (typeof window === 'undefined' || !translation?.meaningAware) return
    if (speaking) {
      window.speechSynthesis.cancel()
      setSpeaking(false)
      return
    }
    const utt  = new SpeechSynthesisUtterance(translation.meaningAware)
    utt.lang   = targetLang?.iso1 ?? 'en'
    utt.onend  = () => setSpeaking(false)
    utt.onerror = () => setSpeaking(false)
    setSpeaking(true)
    window.speechSynthesis.cancel()
    window.speechSynthesis.speak(utt)
  }

  return (
    <div className="flex flex-col gap-4 rounded-2xl border border-white/10 bg-white/5 p-6">

      {/* Detected language header */}
      <div className="flex items-center gap-3">
        <span className="text-4xl">{best.language.flag}</span>
        <div className="flex-1">
          <p className="text-lg font-semibold text-white">{best.language.name}</p>
          <p className="text-sm text-white/50">{confidencePct}% confidence</p>
        </div>
        <span
          className="self-start rounded-lg px-2 py-1 text-[10px] font-semibold uppercase tracking-wide"
          style={{ background: 'rgba(255,107,53,0.12)', color: '#ff6b35', border: '1px solid rgba(255,107,53,0.2)' }}
        >
          Logistic Regression
        </span>
      </div>

      {/* Recharts horizontal confidence chart */}
      <ResponsiveContainer width="100%" height={chartData.length * 30}>
        <BarChart
          layout="vertical"
          data={chartData}
          margin={{ top: 0, right: 8, bottom: 0, left: 0 }}
        >
          <XAxis type="number" domain={[0, 100]} hide />
          <YAxis
            type="category"
            dataKey="name"
            width={148}
            tick={{ fill: 'rgba(255,255,255,0.55)', fontSize: 12 }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip
            cursor={{ fill: 'rgba(255,255,255,0.04)' }}
            contentStyle={{
              background: '#1a1a1a',
              border: '1px solid rgba(255,255,255,0.12)',
              borderRadius: '8px',
              color: '#fff',
              fontSize: '12px',
            }}
            formatter={(v) => [`${v}%`, 'Confidence']}
          />
          <Bar dataKey="pct" radius={[0, 4, 4, 0]} maxBarSize={14}>
            {chartData.map((_, i) => (
              <Cell
                key={i}
                fill={i === 0 ? '#ff6b35' : 'rgba(255,255,255,0.12)'}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      {/* Low-confidence warning */}
      {best.confidence < 0.4 && best.confidence < 0.96 && (
        <p className="text-xs rounded-lg px-3 py-1.5"
          style={{ background: 'rgba(255,165,0,0.1)', border: '1px solid rgba(255,165,0,0.25)', color: 'rgba(255,165,0,0.8)' }}>
          Low confidence — enter more text for a better result
        </p>
      )}

      {/* Translation */}
      <div className="border-t border-white/10 pt-4 flex flex-col gap-3">

        {/* Header with TTS button */}
        <div className="flex items-center justify-between">
          <p className="text-sm font-medium text-white/60">
            Translation → {targetLang ? `${targetLang.flag} ${targetLang.name}` : targetLangIso3}
          </p>
          {translation?.meaningAware && !isTranslating && (
            <button
              type="button"
              onClick={toggleTts}
              title={speaking ? 'Stop' : 'Listen to translation'}
              className="rounded-lg p-1.5 transition hover:bg-white/10"
              style={{ color: speaking ? '#ff6b35' : 'rgba(255,255,255,0.4)' }}
            >
              {speaking
                ? <VolumeX className="h-4 w-4" />
                : <Volume2 className="h-4 w-4" />
              }
            </button>
          )}
        </div>

        {isTranslating ? (
          <div className="flex items-center gap-2 text-white/40">
            <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-white/20 border-t-[#ff6b35]" />
            Translating…
          </div>
        ) : translation ? (
          showDiff ? (
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              {/* Naive */}
              <div className="flex flex-col gap-1.5 rounded-xl p-3"
                style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.08)' }}>
                <p className="text-[11px] font-semibold uppercase tracking-wide text-white/30">
                  Word-by-word (literal)
                </p>
                <p className="text-sm leading-relaxed text-white/60">{translation.naive}</p>
              </div>
              {/* Meaning-aware with diff */}
              <div className="flex flex-col gap-1.5 rounded-xl p-3"
                style={{ background: 'rgba(255,107,53,0.07)', border: '1px solid rgba(255,107,53,0.2)' }}>
                <p className="text-[11px] font-semibold uppercase tracking-wide" style={{ color: '#ff6b35' }}>
                  Meaning-aware
                </p>
                <p className="text-sm leading-relaxed text-white">
                  {diffed!.map((tok, i) =>
                    tok.changed ? (
                      <mark key={i} style={{ background: 'rgba(255,107,53,0.25)', color: '#f7931e', borderRadius: '3px', padding: '0 2px' }}>
                        {tok.text}
                      </mark>
                    ) : (
                      <span key={i}>{tok.text}</span>
                    )
                  )}
                </p>
              </div>
              <p className="sm:col-span-2 text-[11px] text-white/25">
                <mark style={{ background: 'rgba(255,107,53,0.25)', color: '#f7931e', borderRadius: '3px', padding: '0 2px' }}>
                  highlighted
                </mark>
                {' '}= words chosen for meaning/context, not literal substitution
              </p>
            </div>
          ) : (
            <p className="text-base leading-relaxed text-white">{translation.meaningAware}</p>
          )
        ) : (
          <p className="text-sm italic text-white/30">Translation unavailable</p>
        )}
      </div>
    </div>
  )
}
