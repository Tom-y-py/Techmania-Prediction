'use client';

import { useEffect, useState } from 'react';
import { api } from '@/lib/api';
import type { CorrelationData } from '@/types/api';
import { 
  CloudIcon, 
  SunIcon,
  CalendarIcon,
  CalendarDaysIcon 
} from '@heroicons/react/24/outline';
import { useTranslations } from '@/lib/i18n';

export default function CorrelationAnalysis() {
  const t = useTranslations('analytics.correlation');
  const [data, setData] = useState<CorrelationData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const correlationData = await api.getCorrelationData();
        setData(correlationData);
      } catch (err: any) {
        console.error('Failed to fetch correlation data:', err);
        setError(err.message || 'Nepodařilo se načíst data');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6">
        <div className="animate-pulse">
          <div className="h-6 bg-gray-200 dark:bg-gray-700 rounded w-1/3 mb-4"></div>
          <div className="space-y-3">
            <div className="h-20 bg-gray-100 dark:bg-gray-700 rounded"></div>
            <div className="h-20 bg-gray-100 dark:bg-gray-700 rounded"></div>
          </div>
        </div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6">
        <div className="rounded-lg bg-yellow-50 dark:bg-yellow-900/20 p-4">
          <p className="text-sm text-yellow-800 dark:text-yellow-300">
            {error || 'Žádná data k zobrazení'}
          </p>
        </div>
      </div>
    );
  }

  const getCorrelationStrength = (value: number) => {
    const absValue = Math.abs(value);
    if (absValue >= 0.7) return { text: t('strong'), color: 'text-green-600' };
    if (absValue >= 0.4) return { text: t('medium'), color: 'text-yellow-600' };
    return { text: t('weak'), color: 'text-gray-600' };
  };

  const getImpactColor = (value: number) => {
    if (value >= 1.3) return 'text-green-600';
    if (value >= 1.1) return 'text-yellow-600';
    return 'text-gray-600';
  };

  const getImpactLevel = (value: number) => {
    if (value >= 1.3) return t('high');
    if (value >= 1.1) return t('medium');
    return t('low');
  };

  return (
    <div className="bg-white dark:bg-gray-800 shadow-sm ring-1 ring-gray-900/5 dark:ring-gray-700 sm:rounded-xl p-6">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">
        {t('title')}
      </h3>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Weather Correlation */}
        <div className="bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 rounded-lg p-4 border border-blue-200 dark:border-blue-700">
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-blue-200 dark:bg-blue-700 rounded-lg">
                <CloudIcon className="w-6 h-6 text-blue-700 dark:text-blue-300" />
              </div>
              <div>
                <div className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  {t('weatherCorrelation')}
                </div>
                <div className="text-2xl font-bold text-gray-900 dark:text-white">
                  {(data.weather_correlation * 100).toFixed(0)}%
                </div>
              </div>
            </div>
            <div className={`text-sm font-medium ${getCorrelationStrength(data.weather_correlation).color}`}>
              {getCorrelationStrength(data.weather_correlation).text}
            </div>
          </div>
          <div className="mt-2 text-xs text-gray-600 dark:text-gray-400">
            {data.weather_correlation > 0 ? t('positiveInfluence') : t('negativeInfluence')}
          </div>
        </div>

        {/* Temperature Correlation */}
        <div className="bg-gradient-to-br from-orange-50 to-orange-100 dark:from-orange-900/20 dark:to-orange-800/20 rounded-lg p-4 border border-orange-200 dark:border-orange-700">
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-orange-200 dark:bg-orange-700 rounded-lg">
                <SunIcon className="w-6 h-6 text-orange-700 dark:text-orange-300" />
              </div>
              <div>
                <div className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  {t('temperatureCorrelation')}
                </div>
                <div className="text-2xl font-bold text-gray-900 dark:text-white">
                  {(data.temperature_correlation * 100).toFixed(0)}%
                </div>
              </div>
            </div>
            <div className={`text-sm font-medium ${getCorrelationStrength(data.temperature_correlation).color}`}>
              {getCorrelationStrength(data.temperature_correlation).text}
            </div>
          </div>
          <div className="mt-2 text-xs text-gray-600 dark:text-gray-400">
            {t('influencesAttendance')} {getCorrelationStrength(data.temperature_correlation).text.toLowerCase()}
          </div>
        </div>

        {/* Holiday Impact */}
        <div className="bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20 rounded-lg p-4 border border-purple-200 dark:border-purple-700">
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-purple-200 dark:bg-purple-700 rounded-lg">
                <CalendarIcon className="w-6 h-6 text-purple-700 dark:text-purple-300" />
              </div>
              <div>
                <div className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  {t('holidayImpact')}
                </div>
                <div className="text-2xl font-bold text-gray-900 dark:text-white">
                  {data.holiday_impact.toFixed(2)}x
                </div>
              </div>
            </div>
            <div className={`text-sm font-medium ${getImpactColor(data.holiday_impact)}`}>
              {getImpactLevel(data.holiday_impact)}
            </div>
          </div>
          <div className="mt-2 text-xs text-gray-600 dark:text-gray-400">
            {t('holidaysHigher')} {((data.holiday_impact - 1) * 100).toFixed(0)}% {t('higher')}
          </div>
        </div>

        {/* Weekend Impact */}
        <div className="bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 rounded-lg p-4 border border-green-200 dark:border-green-700">
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-green-200 dark:bg-green-700 rounded-lg">
                <CalendarDaysIcon className="w-6 h-6 text-green-700 dark:text-green-300" />
              </div>
              <div>
                <div className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  {t('weekendImpact')}
                </div>
                <div className="text-2xl font-bold text-gray-900 dark:text-white">
                  {data.weekend_impact.toFixed(2)}x
                </div>
              </div>
            </div>
            <div className={`text-sm font-medium ${getImpactColor(data.weekend_impact)}`}>
              {getImpactLevel(data.weekend_impact)}
            </div>
          </div>
          <div className="mt-2 text-xs text-gray-600 dark:text-gray-400">
            {t('weekendsHigher')} {((data.weekend_impact - 1) * 100).toFixed(0)}% {t('higher')}
          </div>
        </div>
      </div>

      {/* Info Box */}
      <div className="mt-6 bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
        <h4 className="text-sm font-semibold text-gray-900 dark:text-white mb-2">
          ℹ️ {t('interpretationTitle')}
        </h4>
        <ul className="text-xs text-gray-600 dark:text-gray-400 space-y-1">
          <li>• <strong>{t('strongCorrelation')}</strong>: {t('strongDesc')}</li>
          <li>• <strong>{t('mediumCorrelation')}</strong>: {t('mediumDesc')}</li>
          <li>• <strong>{t('weakCorrelation')}</strong>: {t('weakDesc')}</li>
          <li>• <strong>{t('impactMultiplier')}</strong>: {t('impactDesc')}</li>
        </ul>
      </div>
    </div>
  );
}
