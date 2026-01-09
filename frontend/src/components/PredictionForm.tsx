'use client';

import { useState } from 'react';
import { format } from 'date-fns';
import { cs } from 'date-fns/locale';
import { api } from '@/lib/api';
import type { PredictionResponse } from '@/types/api';

export default function PredictionForm() {
  const [date, setDate] = useState('');
  const [isHoliday, setIsHoliday] = useState(false);
  const [openingHours, setOpeningHours] = useState('09:00-17:00');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setResult(null);

    try {
      const prediction = await api.predict({
        date,
        is_holiday: isHoliday,
        opening_hours: openingHours,
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
        <div className="max-w-2xl">
          <h2 className="text-base font-semibold leading-7 text-gray-900">
            Predikce návštěvnosti
          </h2>
          <p className="mt-1 text-sm leading-6 text-gray-600">
            Zadejte parametry pro predikci počtu návštěvníků
          </p>

          <form onSubmit={handleSubmit} className="mt-8 space-y-6">
            <div>
              <label htmlFor="date" className="block text-sm font-medium leading-6 text-gray-900">
                Datum
              </label>
              <input
                type="date"
                id="date"
                value={date}
                onChange={(e) => setDate(e.target.value)}
                required
                className="mt-2 block w-full rounded-md border-0 py-1.5 px-3 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-techmania-blue sm:text-sm sm:leading-6"
              />
            </div>

            <div className="flex items-center">
              <input
                type="checkbox"
                id="isHoliday"
                checked={isHoliday}
                onChange={(e) => setIsHoliday(e.target.checked)}
                className="h-4 w-4 rounded border-gray-300 text-techmania-blue focus:ring-techmania-blue"
              />
              <label htmlFor="isHoliday" className="ml-3 block text-sm font-medium leading-6 text-gray-900">
                Státní svátek / prázdniny
              </label>
            </div>

            <div>
              <label htmlFor="openingHours" className="block text-sm font-medium leading-6 text-gray-900">
                Otevírací doba
              </label>
              <select
                id="openingHours"
                value={openingHours}
                onChange={(e) => setOpeningHours(e.target.value)}
                className="mt-2 block w-full rounded-md border-0 py-1.5 px-3 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 focus:ring-2 focus:ring-inset focus:ring-techmania-blue sm:text-sm sm:leading-6"
              >
                <option value="09:00-17:00">09:00 - 17:00</option>
                <option value="09:00-18:00">09:00 - 18:00</option>
                <option value="10:00-18:00">10:00 - 18:00</option>
                <option value="10:00-19:00">10:00 - 19:00</option>
              </select>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full rounded-md bg-techmania-blue px-3.5 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-blue-700 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-techmania-blue disabled:opacity-50"
            >
              {loading ? 'Načítání...' : 'Spustit predikci'}
            </button>
          </form>

          {error && (
            <div className="mt-6 rounded-md bg-red-50 p-4">
              <p className="text-sm text-red-800">{error}</p>
            </div>
          )}

          {result && (
            <div className="mt-6 rounded-lg bg-gradient-to-r from-blue-50 to-green-50 p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Výsledek predikce
              </h3>
              <dl className="grid grid-cols-1 gap-4">
                <div>
                  <dt className="text-sm font-medium text-gray-600">Datum</dt>
                  <dd className="mt-1 text-lg font-semibold text-gray-900">
                    {format(new Date(result.date), 'PPP', { locale: cs })}
                  </dd>
                </div>
                <div>
                  <dt className="text-sm font-medium text-gray-600">Predikovaný počet návštěvníků</dt>
                  <dd className="mt-1 text-3xl font-bold text-techmania-blue">
                    {Math.round(result.predicted_visitors).toLocaleString('cs-CZ')}
                  </dd>
                </div>
                <div>
                  <dt className="text-sm font-medium text-gray-600">Interval spolehlivosti</dt>
                  <dd className="mt-1 text-sm text-gray-700">
                    {Math.round(result.confidence_interval.lower).toLocaleString('cs-CZ')} -{' '}
                    {Math.round(result.confidence_interval.upper).toLocaleString('cs-CZ')} návštěvníků
                  </dd>
                </div>
              </dl>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
