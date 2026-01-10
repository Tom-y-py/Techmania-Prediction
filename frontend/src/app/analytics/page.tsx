'use client';

import { useState } from 'react';
import Sidebar from '@/components/Sidebar';
import Header from '@/components/Header';
import CalendarHeatmap from '@/components/CalendarHeatmap';
import SeasonalityChart from '@/components/SeasonalityChart';
import CorrelationAnalysis from '@/components/CorrelationAnalysis';
import TrendAnalysis from '@/components/TrendAnalysis';
import { useTranslations } from '@/lib/i18n';

export default function AnalyticsPage() {
  const t = useTranslations('analytics');
  const [selectedYear, setSelectedYear] = useState(new Date().getFullYear());

  const currentYear = new Date().getFullYear();
  const availableYears = [currentYear - 2, currentYear - 1, currentYear];

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <Sidebar />
      
      <main className="lg:pl-72">
        <Header />
        <div className="px-4 py-10 sm:px-6 lg:px-8">
          {/* Header */}
          <div className="mb-8">
            <div>
              <h1 className="text-3xl font-bold tracking-tight text-gray-900 dark:text-white">
                {t('title') || 'Datová analytika'}
              </h1>
              <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">
                {t('subtitle') || 'Podrobná analýza historických dat a trendů návštěvnosti'}
              </p>
            </div>
          </div>

          {/* Year Selector */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Vyberte rok pro analýzu
            </label>
            <div className="flex gap-2">
              {availableYears.map(year => (
                <button
                  key={year}
                  onClick={() => setSelectedYear(year)}
                  className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
                    selectedYear === year
                      ? 'bg-techmania-blue text-white'
                      : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 border border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-700'
                  }`}
                >
                  {year}
                </button>
              ))}
            </div>
          </div>

          {/* Trend Analysis */}
          <div className="mb-8">
            <TrendAnalysis />
          </div>

          {/* Correlation Analysis */}
          <div className="mb-8">
            <CorrelationAnalysis />
          </div>

          {/* Seasonality Charts */}
          <div className="mb-8">
            <SeasonalityChart />
          </div>

          {/* Calendar Heatmap */}
          <div className="mb-8">
            <CalendarHeatmap year={selectedYear} />
          </div>

          {/* Additional Insights */}
          <div className="bg-white dark:bg-gray-800 shadow-sm ring-1 ring-gray-900/5 dark:ring-gray-700 sm:rounded-xl p-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              📊 Klíčová zjištění
            </h3>
            
            <div className="space-y-4">
              <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4 border-l-4 border-blue-500">
                <h4 className="text-sm font-semibold text-gray-900 dark:text-white mb-1">
                  Nejvytíženější období
                </h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Největší návštěvnost je obvykle o víkendech a státních svátcích, 
                  zejména v letních měsících (červen-srpen).
                </p>
              </div>

              <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4 border-l-4 border-green-500">
                <h4 className="text-sm font-semibold text-gray-900 dark:text-white mb-1">
                  Vliv počasí
                </h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Pěkné počasí má pozitivní vliv na návštěvnost. 
                  Teploty mezi 20-25°C jsou optimální pro vyšší počet návštěvníků.
                </p>
              </div>

              <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4 border-l-4 border-purple-500">
                <h4 className="text-sm font-semibold text-gray-900 dark:text-white mb-1">
                  Sezónní vzorce
                </h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Jasné sezónní vzorce ukazují zvýšenou návštěvnost během školních prázdnin 
                  a pokles v zimních měsících.
                </p>
              </div>

              <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-4 border-l-4 border-orange-500">
                <h4 className="text-sm font-semibold text-gray-900 dark:text-white mb-1">
                  Doporučení pro plánování
                </h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Pro optimální využití kapacity doporučujeme zvýšení personálu o víkendech 
                  a svátcích, zejména v letním období. V zimních měsících lze plánovat údržbu 
                  a servisní práce.
                </p>
              </div>
            </div>
          </div>

          {/* Data Quality Info */}
          <div className="mt-6 bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
            <h4 className="text-sm font-semibold text-gray-900 dark:text-white mb-2">
              ℹ️ O datech a metodologii
            </h4>
            <ul className="text-xs text-gray-600 dark:text-gray-400 space-y-1">
              <li>• Data zahrnují návštěvnost Techmanie od roku 2021</li>
              <li>• Korelace jsou počítány pomocí Pearsonova korelačního koeficientu</li>
              <li>• Klouzavé průměry jsou vypočítány z 7-denního okna</li>
              <li>• Predikční model využívá ensemble více algoritmů strojového učení</li>
              <li>• Všechny hodnoty jsou aktualizovány denně</li>
            </ul>
          </div>
        </div>
      </main>
    </div>
  );
}
