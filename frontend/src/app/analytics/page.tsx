'use client';

import Sidebar from '@/components/Sidebar';
import Header from '@/components/Header';
import CalendarHeatmap from '@/components/CalendarHeatmap';
import SeasonalityChart from '@/components/SeasonalityChart';
import CorrelationAnalysis from '@/components/CorrelationAnalysis';
import TrendAnalysis from '@/components/TrendAnalysis';
import { useTranslations } from '@/lib/i18n';

export default function AnalyticsPage() {
  const t = useTranslations('analytics');

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
                {t('title')}
              </h1>
              <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">
                {t('subtitle')}
              </p>
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
            <CalendarHeatmap />
          </div>

          {/* Additional Insights */}
          <div className="bg-white dark:bg-gray-800 shadow-sm ring-1 ring-gray-900/5 dark:ring-gray-700 sm:rounded-xl p-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              📊 {t('keyInsights.title')}
            </h3>
            
            <div className="space-y-4">
              <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4 border-l-4 border-blue-500">
                <h4 className="text-sm font-semibold text-gray-900 dark:text-white mb-1">
                  {t('keyInsights.busiestPeriod')}
                </h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  {t('keyInsights.busiestDesc')}
                </p>
              </div>

              <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4 border-l-4 border-green-500">
                <h4 className="text-sm font-semibold text-gray-900 dark:text-white mb-1">
                  {t('keyInsights.weatherInfluence')}
                </h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  {t('keyInsights.weatherDesc')}
                </p>
              </div>

              <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4 border-l-4 border-purple-500">
                <h4 className="text-sm font-semibold text-gray-900 dark:text-white mb-1">
                  {t('keyInsights.seasonalPatterns')}
                </h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  {t('keyInsights.seasonalDesc')}
                </p>
              </div>

              <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-4 border-l-4 border-orange-500">
                <h4 className="text-sm font-semibold text-gray-900 dark:text-white mb-1">
                  {t('keyInsights.planningRecommendation')}
                </h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  {t('keyInsights.planningDesc')}
                </p>
              </div>
            </div>
          </div>

          {/* Data Quality Info */}
          <div className="mt-6 bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
            <h4 className="text-sm font-semibold text-gray-900 dark:text-white mb-2">
              ℹ️ {t('dataInfo.title')}
            </h4>
            <ul className="text-xs text-gray-600 dark:text-gray-400 space-y-1">
              <li>• {t('dataInfo.dataRange')}</li>
              <li>• {t('dataInfo.correlationMethod')}</li>
              <li>• {t('dataInfo.movingAverage')}</li>
              <li>• {t('dataInfo.ensembleModel')}</li>
              <li>• {t('dataInfo.dailyUpdate')}</li>
            </ul>
          </div>
        </div>
      </main>
    </div>
  );
}
