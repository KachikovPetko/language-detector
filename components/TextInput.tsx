'use client'

import { useState } from 'react'
import { Loader2 } from 'lucide-react'

const SAMPLES = [
  { flag: '🇬🇧', text: "It's raining cats and dogs outside." },
  { flag: '🇪🇸', text: 'No hay mal que por bien no venga.' },
  { flag: '🇩🇪', text: 'Morgenstund hat Gold im Mund.' },
  { flag: '🇫🇷', text: 'Mieux vaut prévenir que guérir.' },
  { flag: '🇷🇺', text: 'Тише едешь — дальше будешь.' },
  { flag: '🇯🇵', text: '七転び八起き。' },
  { flag: '🇨🇳', text: '书山有路勤为径，学海无涯苦作舟。' },
  { flag: '🇸🇦', text: 'الصبر مفتاح الفرج.' },
]

interface TextInputProps {
  onSubmit: (text: string) => void
  isLoading: boolean
}

export default function TextInput({ onSubmit, isLoading }: TextInputProps) {
  const [value, setValue] = useState('')

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    const trimmed = value.trim()
    if (trimmed) onSubmit(trimmed)
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-3">

      {/* Sample phrase pills */}
      <div className="flex flex-wrap gap-1.5">
        <span className="text-xs self-center mr-1" style={{ color: 'rgba(255,255,255,0.3)' }}>Try:</span>
        {SAMPLES.map((s, i) => (
          <button
            key={i}
            type="button"
            onClick={() => setValue(s.text)}
            disabled={isLoading}
            className="rounded-full px-2.5 py-0.5 text-xs transition hover:bg-white/10 disabled:opacity-40"
            style={{ border: '1px solid rgba(255,255,255,0.12)', color: 'rgba(255,255,255,0.5)' }}
          >
            {s.flag}
          </button>
        ))}
      </div>

      <textarea
        value={value}
        onChange={e => setValue(e.target.value)}
        rows={4}
        placeholder="Type or paste text here to detect its language…"
        disabled={isLoading}
        className="w-full resize-none rounded-xl border border-white/10 bg-white/5 px-4 py-3 text-white placeholder-white/30 focus:outline-none focus:ring-2 focus:ring-[#ff6b35]/60 disabled:opacity-50 transition"
      />
      <button
        type="submit"
        disabled={isLoading || !value.trim()}
        className="flex items-center justify-center gap-2 rounded-xl px-6 py-3 font-semibold text-white
          bg-gradient-to-r from-[#ff6b35] to-[#f7931e]
          hover:opacity-90 active:scale-[0.98] disabled:opacity-40 disabled:cursor-not-allowed transition-all"
      >
        {isLoading ? (
          <>
            <Loader2 className="h-4 w-4 animate-spin" />
            Detecting…
          </>
        ) : (
          'Detect & Translate'
        )}
      </button>
    </form>
  )
}
