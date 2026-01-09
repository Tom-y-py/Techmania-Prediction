import React, { useState } from 'react';
import { api } from '../api/client';
import type { PredictionResponse } from '../types/api';

export const PredictionForm: React.FC = () => {
  const [date, setDate] = useState<string>(new Date().toISOString().split('T')[0]);
  const [isHoliday, setIsHoliday] = useState<boolean>(false);
  const [openingHours, setOpeningHours] = useState<string>('9-17');
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await api.predict({
        date,
        is_holiday: isHoliday,
        opening_hours: openingHours,
      });
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Něco se pokazilo');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-br from-gray-50 to-gray-100 rounded-2xl p-8 border-l-4 border-primary-500 shadow-lg">
        <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-2">
          <span>📅</span>
          <span>Predikce pro konkrétní datum</span>
        </h2>
        
        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label htmlFor="date" className="block text-sm font-semibold text-gray-700 mb-2">
              Vyberte datum:
            </label>
            <input
              type="date"
              id="date"
              value={date}
              onChange={(e) => setDate(e.target.value)}
              required
              className="w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:border-primary-500 focus:ring-2 focus:ring-primary-500 focus:outline-none transition-all"
            />
          </div>

          <div>
            <label htmlFor="isHoliday" className="block text-sm font-semibold text-gray-700 mb-2">
              Je svátek?
            </label>
            <select
              id="isHoliday"
              value={isHoliday ? 'true' : 'false'}
              onChange={(e) => setIsHoliday(e.target.value === 'true')}
              className="w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:border-primary-500 focus:ring-2 focus:ring-primary-500 focus:outline-none transition-all"
            >
              <option value="false">Ne</option>
              <option value="true">Ano</option>
            </select>
          </div>

          <div>
            <label htmlFor="openingHours" className="block text-sm font-semibold text-gray-700 mb-2">
              Otevírací doba:
            </label>
            <input
              type="text"
              id="openingHours"
              value={openingHours}
              onChange={(e) => setOpeningHours(e.target.value)}
              placeholder="např. 9-17"
              className="w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:border-primary-500 focus:ring-2 focus:ring-primary-500 focus:outline-none transition-all"
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-gradient-to-r from-primary-500 to-secondary-500 text-white font-bold py-4 px-8 rounded-lg hover:shadow-xl hover:-translate-y-0.5 transition-all disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:translate-y-0"
          >
            {loading ? (
              <span className="flex items-center justify-center gap-2">
                <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                Zpracovávám...
              </span>
            ) : (
              '🔮 Předpovědět návštěvnost'
            )}
          </button>
        </form>
      </div>

      {error && (
        <div className="bg-red-50 border-l-4 border-red-500 rounded-lg p-4">
          <p className="text-red-800 font-medium">❌ {error}</p>
        </div>
      )}

      {result && (
        <div className="bg-gradient-to-r from-primary-500 to-secondary-500 rounded-2xl p-8 text-white shadow-xl">
          <h2 className="text-2xl font-bold mb-6">📊 Výsledek predikce</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6">
              <div className="text-sm opacity-90 mb-2">Datum</div>
              <div className="text-3xl font-bold">{result.date}</div>
            </div>
            
            <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6">
              <div className="text-sm opacity-90 mb-2">Predikovaná návštěvnost</div>
              <div className="text-3xl font-bold">{result.predicted_visitors}</div>
              <div className="text-sm opacity-75 mt-1">návštěvníků</div>
            </div>
            
            <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6">
              <div className="text-sm opacity-90 mb-2">Interval spolehlivosti</div>
              <div className="text-2xl font-bold">
                {result.confidence_interval.lower} - {result.confidence_interval.upper}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
