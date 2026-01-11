'use client';

import Sidebar from '@/components/Sidebar';
import Header from '@/components/Header';
import EventsCalendar from '@/components/EventsCalendar';
import { useTranslations } from '@/lib/i18n';

export default function CalendarPage() {
  const t = useTranslations('calendar');

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <Sidebar />
      
      <main className="lg:pl-72">
        <Header />
        <div className="px-4 py-10 sm:px-6 lg:px-8">
          {/* Header */}
          <div className="mb-8">
            <h1 className="text-3xl font-bold tracking-tight text-gray-900 dark:text-white">
              {t('pageTitle') || 'Kalendář událostí'}
            </h1>
            <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">
              {t('pageSubtitle') || 'Přehled svátků, prázdnin a predikované návštěvnosti'}
            </p>
          </div>

          {/* Calendar */}
          <div className="max-w-5xl">
            <EventsCalendar />
          </div>
        </div>
      </main>
    </div>
  );
}
