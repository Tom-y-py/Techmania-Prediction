'use client';

import { useState } from 'react';
import { format } from 'date-fns';
import { cs } from 'date-fns/locale';
import { api } from '@/lib/api';
import type { RangePredictionResponse } from '@/types/api';
import ExportButton from './ExportButton';

export default function RangePredictionForm() {
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<RangePredictionResponse | null>(null);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setResult(null);

    try {
      const prediction = await api.predictRange({
        start_date: startDate,
        end_date: endDate,
      });
      setResult(prediction);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Chyba při načítání predikce');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white shadow-sm ring-1 ring-gray-900/5 sm:rounded-xl">
      <div className="px-4 py-6 sm:p-8">
        <div className="max-w-4xl">
          <h2 className="text-base font-semibold leading-7 text-gray-900">
            Predikce pro rozsah dat
          </h2>
          <p className="mt-1 text-sm leading-6 text-gray-600">
            Získejte predikci pro více dní najednou
          </p>

          <form onSubmit={handleSubmit} className="mt-8 grid grid-cols-1 gap-6 sm:grid-cols-2">
            <div>
              <label htmlFor="startDate" className="block text-sm font-medium leading-6 text-gray-900">
                Datum od
              </label>
              <input
                type="date"
                id="startDate"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                required
                className="mt-2 block w-full rounded-md border-0 py-1.5 px-3 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-techmania-blue sm:text-sm sm:leading-6"
              />
            </div>

            <div>
              <label htmlFor="endDate" className="block text-sm font-medium leading-6 text-gray-900">
                Datum do
              </label>
              <input
                type="date"
                id="endDate"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                required
                className="mt-2 block w-full rounded-md border-0 py-1.5 px-3 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-techmania-blue sm:text-sm sm:leading-6"
              />
            </div>

            <div className="sm:col-span-2">
              <button
                type="submit"
                disabled={loading}
                className="w-full rounded-md bg-techmania-blue px-3.5 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-blue-700 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-techmania-blue disabled:opacity-50"
              >
                {loading ? 'Načítání...' : 'Spustit predikci'}
              </button>
            </div>
          </form>

          {error && (
            <div className="mt-6 rounded-md bg-red-50 p-4">
              <p className="text-sm text-red-800">{error}</p>
            </div>
          )}

          {result && (
            <div className="mt-6">
              <div className="rounded-lg bg-gradient-to-r from-blue-50 to-green-50 p-6 mb-6">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900 mb-2">
                      Celková predikce
                    </h3>
                    <p className="text-3xl font-bold text-techmania-blue">
                      {Math.round(result.total_predicted).toLocaleString('cs-CZ')} návštěvníků
                    </p>
                    <p className="text-sm text-gray-600 mt-1">
                      Pro období {result.predictions.length} dní
                    </p>
                  </div>
                  <ExportButton 
                    data={result.predictions} 
                    filename={`techmania_predikce_${startDate}_${endDate}`}
                  />
                </div>
              </div>

              <div className="overflow-hidden shadow ring-1 ring-black ring-opacity-5 sm:rounded-lg">
                <table className="min-w-full divide-y divide-gray-300">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="py-3.5 pl-4 pr-3 text-left text-sm font-semibold text-gray-900 sm:pl-6">
                        Datum
                      </th>
                      <th className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900">
                        Predikovaný počet návštěvníků
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200 bg-white">
                    {result.predictions.map((prediction, idx) => (
                      <tr key={idx}>
                        <td className="whitespace-nowrap py-4 pl-4 pr-3 text-sm font-medium text-gray-900 sm:pl-6">
                          {format(new Date(prediction.date), 'PPP', { locale: cs })}
                        </td>
                        <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-900 font-semibold">
                          {Math.round(prediction.predicted_visitors).toLocaleString('cs-CZ')}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
