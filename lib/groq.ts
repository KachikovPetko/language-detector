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
          `You are a professional translator. Translate from ${src.name} to ${tgt.name}. ` +
          `Preserve meaning, tone, and idioms naturally. Respond with only the translated text, nothing else.`,
      },
      { role: 'user', content: text },
    ],
    temperature: 0.3,
    max_tokens: 1024,
  })

  return response.choices[0]?.message?.content?.trim() ?? ''
}

export async function transcribeAudio(file: File): Promise<string> {
  const response = await getClient().audio.transcriptions.create({
    file,
    model: 'whisper-large-v3',
  })
  return response.text.trim()
}
