import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import ThemeProvider from '@/components/ThemeProvider'
import { LocaleProvider } from '@/components/LocaleProvider'
import AIChat from '@/components/AIChat'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'Techmania - Dashboard Návštěvnosti',
  description: 'Analýza a predikce návštěvnosti Techmanie',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="cs" suppressHydrationWarning>
      <body className={inter.className}>
        <LocaleProvider>
          <ThemeProvider>
            {children}
            <AIChat />
          </ThemeProvider>
        </LocaleProvider>
      </body>
    </html>
  )
}
