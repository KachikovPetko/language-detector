'use client'

import { Clock, Trash2 } from 'lucide-react'
import type { HistoryItem } from '@/lib/types'
import { getByIso3 } from '@/lib/languages'

interface HistoryPanelProps {
  items:     HistoryItem[]
  onSelect:  (text: string) => void
  onClear:   () => void
}

function timeAgo(ts: number): string {
  const d = Date.now() - ts
  if (d < 60_000)       return 'just now'
  if (d < 3_600_000)    return `${Math.floor(d / 60_000)}m ago`
  if (d < 86_400_000)   return `${Math.floor(d / 3_600_000)}h ago`
  return `${Math.floor(d / 86_400_000)}d ago`
}

export default function HistoryPanel({ items, onSelect, onClear }: HistoryPanelProps) {
  if (items.length === 0) return null

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5 text-xs font-medium" style={{ color: 'rgba(255,255,255,0.35)' }}>
          <Clock className="h-3.5 w-3.5" />
          Recent
        </div>
        <button
          type="button"
          onClick={onClear}
          className="flex items-center gap-1 rounded-md px-2 py-0.5 text-xs transition hover:bg-white/10"
          style={{ color: 'rgba(255,255,255,0.25)' }}
        >
          <Trash2 className="h-3 w-3" />
          Clear
        </button>
      </div>

      <div className="flex flex-col gap-1">
        {items.map(item => {
          const src = getByIso3(item.sourceLangIso3)
          const tgt = getByIso3(item.targetLangIso3)
          return (
            <button
              key={item.id}
              type="button"
              onClick={() => onSelect(item.text)}
              className="flex items-start gap-3 rounded-xl px-3 py-2.5 text-left transition hover:bg-white/5 active:bg-white/8"
              style={{ border: '1px solid rgba(255,255,255,0.06)' }}
            >
              <span className="text-lg leading-none mt-0.5 shrink-0">
                {src?.flag ?? '🌐'}
              </span>
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm text-white/70">
                  {item.text.length > 60 ? item.text.slice(0, 60) + '…' : item.text}
                </p>
                <p className="mt-0.5 text-xs" style={{ color: 'rgba(255,255,255,0.3)' }}>
                  {src?.name ?? item.sourceLangIso3}
                  {tgt && ` → ${tgt.flag} ${tgt.name}`}
                  {' · '}{timeAgo(item.timestamp)}
                </p>
              </div>
            </button>
          )
        })}
      </div>
    </div>
  )
}
