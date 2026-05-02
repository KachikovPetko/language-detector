'use client'

import { Type, FileAudio, Mic } from 'lucide-react'
import type { InputMode } from '@/lib/types'

interface ModeToggleProps {
  value: InputMode
  onChange: (mode: InputMode) => void
}

const MODES: { id: InputMode; label: string; Icon: React.ComponentType<{ className?: string }> }[] = [
  { id: 'text',  label: 'Text',           Icon: Type      },
  { id: 'audio', label: 'Audio File',     Icon: FileAudio },
  { id: 'live',  label: 'Live Recording', Icon: Mic       },
]

export default function ModeToggle({ value, onChange }: ModeToggleProps) {
  return (
    <div className="flex gap-1 rounded-xl p-1" style={{ background: 'rgba(255,255,255,0.05)' }}>
      {MODES.map(({ id, label, Icon }) => {
        const active = value === id
        return (
          <button
            key={id}
            type="button"
            onClick={() => onChange(id)}
            className="flex flex-1 items-center justify-center gap-1.5 rounded-lg px-3 py-2 text-sm font-medium transition-all"
            style={
              active
                ? { background: 'linear-gradient(to right, #ff6b35, #f7931e)', color: '#fff' }
                : { color: 'rgba(255,255,255,0.45)' }
            }
          >
            <Icon className="h-3.5 w-3.5 shrink-0" />
            <span className="hidden sm:inline">{label}</span>
          </button>
        )
      })}
    </div>
  )
}
