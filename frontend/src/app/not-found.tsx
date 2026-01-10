'use client';

import { useTranslations } from '@/lib/i18n';

export default function NotFound() {
  const t = useTranslations('errors');
  
  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-50 dark:bg-gray-900 px-4">
      <div className="max-w-md text-center">
        <div className="mb-8">
          <h1 className="text-9xl font-bold text-techmania-blue dark:text-blue-400">404</h1>
        </div>
        <h2 className="mb-4 text-2xl font-semibold text-gray-900 dark:text-white">
          {t('notFoundTitle')}
        </h2>
        <p className="mb-8 text-gray-600 dark:text-gray-400">
          {t('notFoundDescription')}
        </p>
        <a
          href="/"
          className="rounded-md bg-techmania-blue dark:bg-blue-600 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-blue-700 dark:hover:bg-blue-500"
        >
          {t('backToHome')}
        </a>
      </div>
    </div>
  );
}
