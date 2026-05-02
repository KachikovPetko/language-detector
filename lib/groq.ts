import Groq from 'groq-sdk'
import { getByIso3 } from './languages'

// Module-level singleton — one connection pool per Lambda instance
let client: Groq | null = null

function getClient(): Groq {
  if (!client) {
    const apiKey = process.env.GROQ_API_KEY
    if (!apiKey) throw new Error('GROQ_API_KEY is not set')
    client = new Groq({ apiKey })
  }
  return client
}

export async function translateNaive(
  text: string,
  sourceLangIso3: string,
  targetLangIso3: string
): Promise<string> {
  const src = getByIso3(sourceLangIso3)
  const tgt = getByIso3(targetLangIso3)
  if (!src || !tgt) throw new Error('Unknown language code')

  const response = await getClient().chat.completions.create({
    model: 'llama-3.3-70b-versatile',
    messages: [
      {
        role: 'system',
        content:
          `You are a mechanical word-by-word translator from ${src.name} to ${tgt.name}. ` +
          `Translate each word individually in the exact original order. ` +
          `Do NOT adjust grammar, word order, or sentence structure. ` +
          `Do NOT consider context or idioms — translate each word in isolation using its most common literal meaning. ` +
          `Use ONLY ${tgt.name} script/characters. ` +
          `Output ONLY the word-for-word translation with no explanation.`,
      },
      { role: 'user', content: text },
    ],
    temperature: 0.1,
    max_tokens: 1024,
  })

  return response.choices[0]?.message?.content?.trim() ?? ''
}

export async function translateMeaningAware(
  text: string,
  sourceLangIso3: string,
  targetLangIso3: string
): Promise<string> {
  const src = getByIso3(sourceLangIso3)
  const tgt = getByIso3(targetLangIso3)
  if (!src || !tgt) throw new Error('Unknown language code')

  const response = await getClient().chat.completions.create({
    model: 'llama-3.3-70b-versatile',
    messages: [
      {
        role: 'system',
        content:
          `You are a professional translator. Translate the given text from ${src.name} to ${tgt.name}. ` +
          `Rules: ` +
          `(1) Output ONLY the translated text — no explanations, no labels, no quotes. ` +
          `(2) Use ONLY ${tgt.name} characters and script. Never mix in characters from any other language. ` +
          `(3) For untranslatable foreign words or neologisms, transliterate them into the ${tgt.name} alphabet/script, or keep them as-is in Latin script if transliteration is not natural. ` +
          `(4) Preserve the original meaning, tone, and register.`,
      },
      { role: 'user', content: text },
    ],
    temperature: 0.3,
    max_tokens: 1024,
  })

  return response.choices[0]?.message?.content?.trim() ?? ''
}

export async function transcribeAudio(file: File): Promise<{ text: string; whisperLang: string }> {
  // verbose_json returns `language` (ISO 639-1 code) alongside the transcript
  const response = await getClient().audio.transcriptions.create({
    file,
    model: 'whisper-large-v3',
    response_format: 'verbose_json',
  }) as unknown as { text: string; language?: string }

  return {
    text:        (response.text     ?? '').trim(),
    whisperLang: (response.language ?? '').toLowerCase(),
  }
}
