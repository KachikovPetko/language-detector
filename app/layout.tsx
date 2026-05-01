import type { Metadata } from 'next'
import { Geist } from 'next/font/google'
import './globals.css'

const geist = Geist({ subsets: ['latin'], variable: '--font-geist-sans' })

export const metadata: Metadata = {
  title: 'LinguaLens — Language Detector & Translator',
  description:
    'Detect the language of any text using a custom-trained ML model, ' +
    'then translate with naive vs. meaning-aware comparison.',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${geist.variable} h-full antialiased`}>
      <body className="flex min-h-full flex-col" style={{ background: '#1a1612' }}>
        {children}
      </body>
    </html>
  )
}
