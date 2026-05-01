'use client'

import { LANGUAGES } from '@/lib/languages'

interface TargetLanguagePickerProps {
  value: string
  onChange: (iso3: string) => void
}

export default function TargetLanguagePicker({ value, onChange }: TargetLanguagePickerProps) {
  return (
    <div className="flex flex-col gap-1.5">
      <label className="text-sm font-medium text-white/60">Translate to</label>
      <select
        value={value}
        onChange={e => onChange(e.target.value)}
        className="w-full rounded-xl border border-white/10 bg-white/5 px-4 py-2.5 text-white focus:outline-none focus:ring-2 focus:ring-[#ff6b35]/60 transition cursor-pointer"
      >
        {LANGUAGES.map(lang => (
          <option key={lang.iso3} value={lang.iso3} className="bg-[#1a1612] text-white">
            {lang.flag} {lang.name}
          </option>
        ))}
      </select>
    </div>
  )
}
