'use client';

import { useEffect, useState } from 'react';
import { NextIntlClientProvider } from 'next-intl';
import { useTranslations } from '@/lib/i18n';

function ErrorContent({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  const t = useTranslations('errors');
  
  useEffect(() => {
    console.error(error);
  }, [error]);

  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-50 dark:bg-gray-900 px-4">
      <div className="max-w-md text-center">
        <div className="mb-8">
          <h1 className="text-6xl font-bold text-techmania-blue dark:text-blue-400">{t('title')}</h1>
        </div>
        <h2 className="mb-4 text-2xl font-semibold text-gray-900 dark:text-white">
          {t('subtitle')}
        </h2>
        <p className="mb-8 text-gray-600 dark:text-gray-400">
          {t('description')}
        </p>
        <div className="space-x-4">
          <button
            onClick={reset}
            className="rounded-md bg-techmania-blue dark:bg-blue-600 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-blue-700 dark:hover:bg-blue-500"
          >
            {t('tryAgain')}
          </button>
          <a
            href="/"
            className="rounded-md bg-gray-200 dark:bg-gray-700 px-4 py-2 text-sm font-semibold text-gray-900 dark:text-gray-100 shadow-sm hover:bg-gray-300 dark:hover:bg-gray-600"
          >
            {t('goHome')}
          </a>
        </div>
      </div>
    </div>
  );
}

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  const [messages, setMessages] = useState<any>(null);
  const [locale, setLocale] = useState<'cs' | 'en'>('cs');

  useEffect(() => {
    // Load saved locale from localStorage
    const savedSettings = localStorage.getItem('techmania-settings');
    if (savedSettings) {
      try {
        const settings = JSON.parse(savedSettings);
        if (settings.language) {
          setLocale(settings.language);
        }
      } catch (error) {
        console.error('Error loading locale:', error);
      }
    }
  }, []);

  useEffect(() => {
    // Load messages for current locale
    import(`../../locale/${locale}.json`)
      .then((module) => setMessages(module.default))
      .catch((error) => console.error('Error loading messages:', error));
  }, [locale]);

  if (!messages) {
    return null;
  }

  return (
    <NextIntlClientProvider locale={locale} messages={messages}>
      <ErrorContent error={error} reset={reset} />
    </NextIntlClientProvider>
  );
}
