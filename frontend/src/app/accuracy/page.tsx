/**
 * Accuracy Page - Přesnost predikcí
 */

'use client';

import Sidebar from '@/components/Sidebar';
import Header from '@/components/Header';
import PredictionAccuracy from '@/components/PredictionAccuracy';
import { useTranslations } from '@/lib/i18n';

export default function AccuracyPage() {
  const t = useTranslations('accuracy');

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <Sidebar />
      
      <main className="lg:pl-72">
        <Header />
        <div className="px-4 py-10 sm:px-6 lg:px-8">
          {/* Header */}
          <div className="mb-8">
            <h1 className="text-3xl font-bold tracking-tight text-gray-900 dark:text-white">
              Přesnost predikcí
            </h1>
            <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">
              Porovnání predikovaných a skutečných hodnot návštěvnosti
            </p>
          </div>

          {/* Prediction Accuracy Component */}
          <PredictionAccuracy />
        </div>
      </main>
    </div>
  );
}
