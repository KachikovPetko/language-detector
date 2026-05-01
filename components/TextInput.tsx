'use client'

import { useState } from 'react'
import { Loader2 } from 'lucide-react'

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
