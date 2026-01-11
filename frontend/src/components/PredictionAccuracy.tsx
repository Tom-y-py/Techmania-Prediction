/**
 * Prediction Accuracy Component
 * Zobrazuje historii predikcí a jejich přesnost oproti skutečným hodnotám
 */

'use client';

import { useState, useEffect } from 'react';
import { useAnalytics } from '@/hooks/useApi';
import type { PredictionHistoryResponse } from '@/types/api';

export default function PredictionAccuracy() {
  const { data, loading, error, fetchPredictionHistory } = useAnalytics();
  const [days, setDays] = useState(30);
  const [includeFuture, setIncludeFuture] = useState(true);

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

  const historyData = data as PredictionHistoryResponse | null;

  if (loading) {
    return (
      <div className="flex justify-center items-center p-8">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg">
        {error}
      </div>
    );
  }

  if (!historyData) return null;

  const { history, summary } = historyData;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold">Přesnost predikcí</h2>
        
        <div className="flex gap-4 items-center">
          <select
            value={days}
            onChange={(e) => setDays(Number(e.target.value))}
            className="px-3 py-2 border rounded-lg"
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
              className="rounded"
            />
            <span className="text-sm">Zahrnout budoucí predikce</span>
          </label>
        </div>
      </div>

      {/* Summary Cards */}
      {summary.valid_comparisons > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-white rounded-lg shadow p-4">
            <div className="text-sm text-gray-600 mb-1">Průměrná chyba</div>
            <div className="text-2xl font-bold text-blue-600">
              {summary.avg_error?.toFixed(0)} návštěvníků
            </div>
            <div className="text-sm text-gray-500 mt-1">
              {summary.avg_error_percent?.toFixed(1)}%
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-4">
            <div className="text-sm text-gray-600 mb-1">Přesnost ±10%</div>
            <div className="text-2xl font-bold text-green-600">
              {summary.accuracy_10_percent?.toFixed(1)}%
            </div>
            <div className="text-sm text-gray-500 mt-1">
              {summary.predictions_within_10_percent} z {summary.valid_comparisons}
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-4">
            <div className="text-sm text-gray-600 mb-1">Přesnost ±20%</div>
            <div className="text-2xl font-bold text-yellow-600">
              {summary.accuracy_20_percent?.toFixed(1)}%
            </div>
            <div className="text-sm text-gray-500 mt-1">
              {summary.predictions_within_20_percent} z {summary.valid_comparisons}
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-4">
            <div className="text-sm text-gray-600 mb-1">Celkem porovnání</div>
            <div className="text-2xl font-bold text-gray-700">
              {summary.valid_comparisons}
            </div>
            <div className="text-sm text-gray-500 mt-1">
              z {summary.total_predictions} predikcí
            </div>
          </div>
        </div>
      )}

      {/* History Table */}
      <div className="bg-white rounded-lg shadow overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Datum
                </th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">
                  Predikce
                </th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">
                  Skutečnost
                </th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">
                  Chyba
                </th>
                <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase">
                  Confidence
                </th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">
                  Verze
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {history.map((item, index) => {
                const errorPercent = item.error_percent;
                const isAccurate = errorPercent !== null && Math.abs(errorPercent) <= 10;
                const isGood = errorPercent !== null && Math.abs(errorPercent) <= 20;
                
                return (
                  <tr key={index} className={item.is_future ? 'bg-gray-50' : ''}>
                    <td className="px-4 py-3 text-sm">
                      <div className="font-medium">
                        {new Date(item.date).toLocaleDateString('cs-CZ')}
                      </div>
                      {item.is_future && (
                        <span className="text-xs text-gray-500">Budoucí</span>
                      )}
                    </td>
                    
                    <td className="px-4 py-3 text-sm text-right font-medium">
                      {item.predicted.toLocaleString('cs-CZ')}
                    </td>
                    
                    <td className="px-4 py-3 text-sm text-right">
                      {item.actual !== null ? (
                        <span className="font-medium">
                          {item.actual.toLocaleString('cs-CZ')}
                        </span>
                      ) : (
                        <span className="text-gray-400">-</span>
                      )}
                    </td>
                    
                    <td className="px-4 py-3 text-sm text-right">
                      {item.error !== null ? (
                        <div>
                          <div
                            className={`font-medium ${
                              isAccurate
                                ? 'text-green-600'
                                : isGood
                                ? 'text-yellow-600'
                                : 'text-red-600'
                            }`}
                          >
                            {item.error > 0 ? '+' : ''}
                            {item.error.toLocaleString('cs-CZ')}
                          </div>
                          <div className="text-xs text-gray-500">
                            {errorPercent !== null && (
                              <>
                                {errorPercent > 0 ? '+' : ''}
                                {errorPercent.toFixed(1)}%
                              </>
                            )}
                          </div>
                        </div>
                      ) : (
                        <span className="text-gray-400">-</span>
                      )}
                    </td>
                    
                    <td className="px-4 py-3 text-center text-sm">
                      {item.confidence_lower !== null && item.confidence_upper !== null ? (
                        <div>
                          <div className="text-xs text-gray-500">
                            {item.confidence_lower.toLocaleString('cs-CZ')} -{' '}
                            {item.confidence_upper.toLocaleString('cs-CZ')}
                          </div>
                          {item.actual !== null && (
                            <div className="mt-1">
                              {item.within_confidence ? (
                                <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-green-100 text-green-800">
                                  ✓ V rozsahu
                                </span>
                              ) : (
                                <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-red-100 text-red-800">
                                  ✗ Mimo rozsah
                                </span>
                              )}
                            </div>
                          )}
                        </div>
                      ) : (
                        <span className="text-gray-400">-</span>
                      )}
                    </td>
                    
                    <td className="px-4 py-3 text-sm text-right text-gray-500">
                      v{item.version}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Legend */}
      <div className="flex gap-4 text-sm">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-green-600"></div>
          <span>Chyba ≤10%</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-yellow-600"></div>
          <span>Chyba ≤20%</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-red-600"></div>
          <span>Chyba >20%</span>
        </div>
      </div>
    </div>
  );
}
