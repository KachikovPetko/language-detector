'use client'

import { getByIso3 } from '@/lib/languages'
import type { ModelPrediction } from '@/lib/types'

interface Props {
  models: ModelPrediction[]
}

const MODEL_META: Record<string, { label: string; short: string; primary: boolean }> = {
  logreg: { label: 'Logistic Regression', short: 'LogReg', primary: true  },
  svc:    { label: 'Linear SVC',          short: 'SVC',    primary: false },
  nb:     { label: 'Naive Bayes',         short: 'NB',     primary: false },
}

export default function ModelComparison({ models }: Props) {
  return (
    <div className="flex flex-col gap-2">
      <p className="text-xs font-semibold uppercase tracking-wide" style={{ color: 'rgba(255,255,255,0.3)' }}>
        Model Comparison
      </p>
      <div className="grid grid-cols-3 gap-2">
        {models.map(m => {
          const meta = MODEL_META[m.model]
          const lang = getByIso3(m.topLanguageIso3)
          const isPrimary = meta.primary
          const confPct   = (m.confidence * 100).toFixed(1)
          const accPct    = m.accuracy > 0 ? `${(m.accuracy * 100).toFixed(1)}%` : '—'

          return (
            <div
              key={m.model}
              className="relative flex flex-col gap-2 rounded-xl p-3"
              style={{
                background: isPrimary ? 'rgba(255,107,53,0.07)' : 'rgba(255,255,255,0.03)',
                border: `1px solid ${isPrimary ? 'rgba(255,107,53,0.2)' : 'rgba(255,255,255,0.08)'}`,
              }}
            >
              {/* Top-right accuracy badge */}
              <span
                className="absolute top-2 right-2 rounded px-1.5 py-0.5 text-[10px] font-medium"
                style={{ background: 'rgba(255,255,255,0.06)', color: 'rgba(255,255,255,0.35)' }}
              >
                {accPct} acc
              </span>

              {/* Model name + primary badge */}
              <div className="flex flex-col gap-0.5 pr-12">
                <p
                  className="text-[10px] font-bold uppercase tracking-wide leading-tight"
                  style={{ color: isPrimary ? '#ff6b35' : 'rgba(255,255,255,0.35)' }}
                >
                  {meta.label}
                </p>
                {isPrimary && (
                  <span
                    className="self-start rounded px-1 py-0.5 text-[9px] font-semibold uppercase"
                    style={{ background: 'rgba(255,107,53,0.2)', color: '#ff6b35' }}
                  >
                    Primary
                  </span>
                )}
              </div>

              {/* Detected language */}
              {lang ? (
                <div className="flex items-center gap-1.5 mt-0.5">
                  <span className="text-xl leading-none">{lang.flag}</span>
                  <div>
                    <p className="text-sm font-medium text-white leading-tight">{lang.name}</p>
                    <p className="text-[11px]" style={{ color: 'rgba(255,255,255,0.4)' }}>
                      {confPct}%
                    </p>
                  </div>
                </div>
              ) : (
                <p className="text-xs italic" style={{ color: 'rgba(255,255,255,0.3)' }}>Unknown</p>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
