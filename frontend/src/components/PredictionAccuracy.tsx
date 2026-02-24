/**
 * Prediction Accuracy Component
 * Zobrazuje historii predikcí a jejich přesnost oproti skutečným hodnotám
 */

'use client';

import React, { useState, useEffect } from 'react';
import { useAnalytics } from '@/hooks/useApi';
import type { PredictionHistoryResponse } from '@/types/api';

export default function PredictionAccuracy() {
  const { data, loading, error, fetchPredictionHistory } = useAnalytics();
  const [days, setDays] = useState(30);
  const [includeFuture, setIncludeFuture] = useState(true);
  const [expandedDates, setExpandedDates] = useState<Set<string>>(new Set());

  useEffect(() => {
    loadData();
  }, [days, includeFuture]);

  const loadData = async () => {
    try {
      await fetchPredictionHistory(days, includeFuture);
    } catch (err) {
      console.error('Failed to load prediction history:', err);
    }
  };

  const toggleDate = (date: string) => {
    const newExpanded = new Set(expandedDates);
    if (newExpanded.has(date)) {
      newExpanded.delete(date);
    } else {
      newExpanded.add(date);
    }
    setExpandedDates(newExpanded);
  };

  const historyData = data as PredictionHistoryResponse | null;

  if (loading) {
    return (
      <div className="flex justify-center items-center p-12">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 dark:border-blue-400"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-700 dark:text-red-300 px-4 py-3 rounded-lg">
        {error}
      </div>
    );
  }

  if (!historyData) return null;

  const { history, summary } = historyData;

  // Seskupit predikce podle data a seřadit podle verze (nejnovější první)
  const groupedByDate = history.reduce((acc, item) => {
    if (!acc[item.date]) {
      acc[item.date] = [];
    }
    acc[item.date].push(item);
    return acc;
  }, {} as Record<string, typeof history>);

  // Seřadit verze u každého data (nejvyšší verze první)
  Object.keys(groupedByDate).forEach(date => {
    groupedByDate[date].sort((a, b) => b.version - a.version);
  });

  // Seřadit data sestupně (nejnovější datum první)
  const sortedDates = Object.keys(groupedByDate).sort((a, b) => 
    new Date(b).getTime() - new Date(a).getTime()
  );

  const renderPredictionRow = (item: typeof history[0], isLatest: boolean, showDate: boolean) => {
    const errorPercent = item.error_percent;
    const isAccurate = errorPercent !== null && Math.abs(errorPercent) <= 10;
    const isGood = errorPercent !== null && Math.abs(errorPercent) <= 20;
    
    return (
      <tr 
        key={`${item.date}-${item.version}`} 
        className={`
          ${item.is_future ? 'bg-gray-50 dark:bg-gray-900/30' : 'hover:bg-gray-50 dark:hover:bg-gray-900/30'} 
          ${!isLatest ? 'bg-gray-100/50 dark:bg-gray-800/50' : ''} 
          transition-colors
        `}
      >
        <td className="px-4 py-3 text-sm text-gray-900 dark:text-gray-100">
          {showDate && (
            <>
              <div className="font-medium">
                {new Date(item.date).toLocaleDateString('cs-CZ')}
              </div>
              {item.is_future && (
                <span className="text-xs text-gray-500 dark:text-gray-400">Budoucí</span>
              )}
            </>
          )}
          {!isLatest && (
            <span className="text-xs text-gray-500 dark:text-gray-400 ml-4">
              ↳ Starší verze
            </span>
          )}
        </td>
        
        <td className="px-4 py-3 text-sm text-right font-medium text-gray-900 dark:text-gray-100">
          {item.predicted.toLocaleString('cs-CZ')}
        </td>
        
        <td className="px-4 py-3 text-sm text-right">
          {item.actual !== null ? (
            <span className="font-medium text-gray-900 dark:text-gray-100">
              {item.actual.toLocaleString('cs-CZ')}
            </span>
          ) : (
            <span className="text-gray-400 dark:text-gray-500">-</span>
          )}
        </td>
        
        <td className="px-4 py-3 text-sm text-right">
          {item.error !== null ? (
            <div>
              <div
                className={`font-medium ${
                  isAccurate
                    ? 'text-green-600 dark:text-green-400'
                    : isGood
                    ? 'text-yellow-600 dark:text-yellow-400'
                    : 'text-red-600 dark:text-red-400'
                }`}
              >
                {item.error > 0 ? '+' : ''}
                {item.error.toLocaleString('cs-CZ')}
              </div>
              <div className="text-xs text-gray-500 dark:text-gray-400">
                {errorPercent !== null && (
                  <>
                    {errorPercent > 0 ? '+' : ''}
                    {errorPercent.toFixed(1)}%
                  </>
                )}
              </div>
            </div>
          ) : (
            <span className="text-gray-400 dark:text-gray-500">-</span>
          )}
        </td>
        
        <td className="px-4 py-3 text-center text-sm">
          {item.confidence_lower !== null && item.confidence_upper !== null ? (
            <div>
              <div className="text-xs text-gray-500 dark:text-gray-400">
                {item.confidence_lower.toLocaleString('cs-CZ')} -{' '}
                {item.confidence_upper.toLocaleString('cs-CZ')}
              </div>
              {item.actual !== null && (
                <div className="mt-1">
                  {item.within_confidence ? (
                    <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-400">
                      ✓ V rozsahu
                    </span>
                  ) : (
                    <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-red-100 dark:bg-red-900/30 text-red-800 dark:text-red-400">
                      ✗ Mimo rozsah
                    </span>
                  )}
                </div>
              )}
            </div>
          ) : (
            <span className="text-gray-400 dark:text-gray-500">-</span>
          )}
        </td>
        
        <td className="px-4 py-3 text-sm text-right">
          <div className="flex items-center justify-end gap-2">
            <span className={`${isLatest ? 'font-medium text-gray-700 dark:text-gray-300' : 'text-gray-500 dark:text-gray-400'}`}>
              v{item.version}
            </span>
            {isLatest && groupedByDate[item.date].length > 1 && (
              <button
                onClick={() => toggleDate(item.date)}
                className="text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 focus:outline-none"
                title={expandedDates.has(item.date) ? 'Skrýt starší verze' : 'Zobrazit starší verze'}
              >
                <svg 
                  className={`w-4 h-4 transition-transform ${expandedDates.has(item.date) ? 'rotate-180' : ''}`} 
                  fill="none" 
                  viewBox="0 0 24 24" 
                  stroke="currentColor"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>
            )}
          </div>
        </td>
      </tr>
    );
  };

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="bg-white dark:bg-gray-800 shadow-sm ring-1 ring-gray-900/5 dark:ring-gray-700 sm:rounded-xl p-6">
        <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center gap-4">
          <div className="flex gap-4 items-center flex-wrap">
            <select
              value={days}
              onChange={(e) => setDays(Number(e.target.value))}
              className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 dark:focus:ring-blue-400"
            >
              <option value={7}>Posledních 7 dní</option>
              <option value={14}>Posledních 14 dní</option>
              <option value={30}>Posledních 30 dní</option>
              <option value={60}>Posledních 60 dní</option>
              <option value={90}>Posledních 90 dní</option>
            </select>

            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={includeFuture}
                onChange={(e) => setIncludeFuture(e.target.checked)}
                className="rounded border-gray-300 dark:border-gray-600 text-blue-600 focus:ring-blue-500"
              />
              <span className="text-sm text-gray-700 dark:text-gray-300">Zahrnout budoucí predikce</span>
            </label>
          </div>
        </div>
      </div>

      {/* Summary Cards */}
      {summary.valid_comparisons > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <div className="bg-white dark:bg-gray-800 shadow-sm ring-1 ring-gray-900/5 dark:ring-gray-700 rounded-xl p-6 hover:shadow-lg transition-shadow">
            <div className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-2">Průměrná chyba</div>
            <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
              {summary.avg_error?.toFixed(0)} návštěvníků
            </div>
            <div className="text-sm text-gray-500 dark:text-gray-400 mt-1">
              {summary.avg_error_percent?.toFixed(1)}%
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 shadow-sm ring-1 ring-gray-900/5 dark:ring-gray-700 rounded-xl p-6 hover:shadow-lg transition-shadow">
            <div className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-2">Přesnost ±10%</div>
            <div className="text-2xl font-bold text-green-600 dark:text-green-400">
              {summary.accuracy_10_percent?.toFixed(1)}%
            </div>
            <div className="text-sm text-gray-500 dark:text-gray-400 mt-1">
              {summary.predictions_within_10_percent} z {summary.valid_comparisons}
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 shadow-sm ring-1 ring-gray-900/5 dark:ring-gray-700 rounded-xl p-6 hover:shadow-lg transition-shadow">
            <div className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-2">Přesnost ±20%</div>
            <div className="text-2xl font-bold text-yellow-600 dark:text-yellow-400">
              {summary.accuracy_20_percent?.toFixed(1)}%
            </div>
            <div className="text-sm text-gray-500 dark:text-gray-400 mt-1">
              {summary.predictions_within_20_percent} z {summary.valid_comparisons}
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 shadow-sm ring-1 ring-gray-900/5 dark:ring-gray-700 rounded-xl p-6 hover:shadow-lg transition-shadow">
            <div className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-2">Celkem porovnání</div>
            <div className="text-2xl font-bold text-gray-700 dark:text-gray-300">
              {summary.valid_comparisons}
            </div>
            <div className="text-sm text-gray-500 dark:text-gray-400 mt-1">
              z {summary.total_predictions} predikcí
            </div>
          </div>
        </div>
      )}

      {/* History Table */}
      <div className="bg-white dark:bg-gray-800 shadow-sm ring-1 ring-gray-900/5 dark:ring-gray-700 sm:rounded-xl overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50 dark:bg-gray-900/50">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Datum
                </th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Predikce
                </th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Skutečnost
                </th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Chyba
                </th>
                <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Confidence
                </th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Verze
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
              {sortedDates.map((date) => {
                const predictions = groupedByDate[date];
                const latestPrediction = predictions[0];
                const olderPredictions = predictions.slice(1);
                const isExpanded = expandedDates.has(date);

                return (
                  <React.Fragment key={date}>
                    {/* Nejnovější verze pro dané datum */}
                    {renderPredictionRow(latestPrediction, true, true)}
                    
                    {/* Starší verze (pokud jsou rozbalené) */}
                    {isExpanded && olderPredictions.map((prediction) => 
                      renderPredictionRow(prediction, false, false)
                    )}
                  </React.Fragment>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Legend */}
      <div className="flex gap-6 text-sm text-gray-600 dark:text-gray-400">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-green-600 dark:bg-green-400"></div>
          <span>Chyba ≤10%</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-yellow-600 dark:bg-yellow-400"></div>
          <span>Chyba ≤20%</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-red-600 dark:bg-red-400"></div>
          <span>Chyba &gt;20%</span>
        </div>
      </div>
    </div>
  );
}
