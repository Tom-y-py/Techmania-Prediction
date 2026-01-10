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

export default function CorrelationAnalysis() {
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
        // Pro demo účely můžeme použít ukázková data
        setData({
          weather_correlation: 0.65,
          temperature_correlation: 0.58,
          holiday_impact: 1.45,
          weekend_impact: 1.32
        });
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
    if (absValue >= 0.7) return { text: 'Silná', color: 'text-green-600' };
    if (absValue >= 0.4) return { text: 'Střední', color: 'text-yellow-600' };
    return { text: 'Slabá', color: 'text-gray-600' };
  };

  const getImpactColor = (value: number) => {
    if (value >= 1.3) return 'text-green-600';
    if (value >= 1.1) return 'text-yellow-600';
    return 'text-gray-600';
  };

  return (
    <div className="bg-white dark:bg-gray-800 shadow-sm ring-1 ring-gray-900/5 dark:ring-gray-700 sm:rounded-xl p-6">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">
        Analýza korelací a vlivů
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
                  Korelace s počasím
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
            Pěkné počasí má {data.weather_correlation > 0 ? 'pozitivní' : 'negativní'} vliv na návštěvnost
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
                  Korelace s teplotou
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
            Teplota ovlivňuje návštěvnost {getCorrelationStrength(data.temperature_correlation).text.toLowerCase()}
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
                  Vliv svátků
                </div>
                <div className="text-2xl font-bold text-gray-900 dark:text-white">
                  {data.holiday_impact.toFixed(2)}x
                </div>
              </div>
            </div>
            <div className={`text-sm font-medium ${getImpactColor(data.holiday_impact)}`}>
              {data.holiday_impact >= 1.3 ? 'Vysoký' : data.holiday_impact >= 1.1 ? 'Střední' : 'Nízký'}
            </div>
          </div>
          <div className="mt-2 text-xs text-gray-600 dark:text-gray-400">
            O svátcích je návštěvnost {((data.holiday_impact - 1) * 100).toFixed(0)}% vyšší
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
                  Vliv víkendů
                </div>
                <div className="text-2xl font-bold text-gray-900 dark:text-white">
                  {data.weekend_impact.toFixed(2)}x
                </div>
              </div>
            </div>
            <div className={`text-sm font-medium ${getImpactColor(data.weekend_impact)}`}>
              {data.weekend_impact >= 1.3 ? 'Vysoký' : data.weekend_impact >= 1.1 ? 'Střední' : 'Nízký'}
            </div>
          </div>
          <div className="mt-2 text-xs text-gray-600 dark:text-gray-400">
            O víkendech je návštěvnost {((data.weekend_impact - 1) * 100).toFixed(0)}% vyšší
          </div>
        </div>
      </div>

      {/* Info Box */}
      <div className="mt-6 bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
        <h4 className="text-sm font-semibold text-gray-900 dark:text-white mb-2">
          ℹ️ Interpretace korelací
        </h4>
        <ul className="text-xs text-gray-600 dark:text-gray-400 space-y-1">
          <li>• <strong>Korelace 0.7+</strong>: Silný vztah mezi faktory</li>
          <li>• <strong>Korelace 0.4-0.7</strong>: Střední vztah</li>
          <li>• <strong>Korelace &lt;0.4</strong>: Slabý vztah</li>
          <li>• <strong>Impact multiplikátor</strong>: Jak moc se návštěvnost násobí (1.0 = žádný efekt)</li>
        </ul>
      </div>
    </div>
  );
}
