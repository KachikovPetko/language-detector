import type { Language } from './types'

// iso3 codes match the WiLI-2018 dataset labels that the ONNX model outputs
export const LANGUAGES: Language[] = [
  { iso3: 'ara', iso1: 'ar', name: 'Arabic',     flag: '🇸🇦' },
  { iso3: 'bul', iso1: 'bg', name: 'Bulgarian',  flag: '🇧🇬' },
  { iso3: 'ces', iso1: 'cs', name: 'Czech',       flag: '🇨🇿' },
  { iso3: 'cmn', iso1: 'zh', name: 'Chinese',    flag: '🇨🇳' },
  { iso3: 'deu', iso1: 'de', name: 'German',     flag: '🇩🇪' },
  { iso3: 'ell', iso1: 'el', name: 'Greek',      flag: '🇬🇷' },
  { iso3: 'eng', iso1: 'en', name: 'English',    flag: '🇬🇧' },
  { iso3: 'fra', iso1: 'fr', name: 'French',     flag: '🇫🇷' },
  { iso3: 'hin', iso1: 'hi', name: 'Hindi',      flag: '🇮🇳' },
  { iso3: 'ita', iso1: 'it', name: 'Italian',    flag: '🇮🇹' },
  { iso3: 'jpn', iso1: 'ja', name: 'Japanese',   flag: '🇯🇵' },
  { iso3: 'kor', iso1: 'ko', name: 'Korean',     flag: '🇰🇷' },
  { iso3: 'nld', iso1: 'nl', name: 'Dutch',      flag: '🇳🇱' },
  { iso3: 'pol', iso1: 'pl', name: 'Polish',     flag: '🇵🇱' },
  { iso3: 'por', iso1: 'pt', name: 'Portuguese', flag: '🇵🇹' },
  { iso3: 'ron', iso1: 'ro', name: 'Romanian',   flag: '🇷🇴' },
  { iso3: 'rus', iso1: 'ru', name: 'Russian',    flag: '🇷🇺' },
  { iso3: 'spa', iso1: 'es', name: 'Spanish',    flag: '🇪🇸' },
  { iso3: 'tur', iso1: 'tr', name: 'Turkish',    flag: '🇹🇷' },
  { iso3: 'ukr', iso1: 'uk', name: 'Ukrainian',  flag: '🇺🇦' },
]

const byIso3 = new Map(LANGUAGES.map(l => [l.iso3, l]))
const byIso1 = new Map(LANGUAGES.map(l => [l.iso1, l]))

export function getByIso3(iso3: string): Language | undefined {
  return byIso3.get(iso3)
}

export function getByIso1(iso1: string): Language | undefined {
  return byIso1.get(iso1)
}
