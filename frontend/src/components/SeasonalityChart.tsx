'use client';

import { useEffect, useState } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Bar } from 'react-chartjs-2';
import { api } from '@/lib/api';
import type { SeasonalityData } from '@/types/api';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

export default function SeasonalityChart() {
  const [data, setData] = useState<SeasonalityData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [view, setView] = useState<'weekday' | 'month'>('weekday');

  useEffect(() => {
    const fetchData = async () => {
      try {
        const seasonalityData = await api.getSeasonalityData();
        setData(seasonalityData);
      } catch (err: any) {
        console.error('Failed to fetch seasonality data:', err);
        setError(err.message);
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
          <div className="h-64 bg-gray-100 dark:bg-gray-700 rounded"></div>
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

  const weekdayOrder = ['Pondělí', 'Úterý', 'Středa', 'Čtvrtek', 'Pátek', 'Sobota', 'Neděle'];
  const monthOrder = ['Leden', 'Únor', 'Březen', 'Duben', 'Květen', 'Červen', 
                      'Červenec', 'Srpen', 'Září', 'Říjen', 'Listopad', 'Prosinec'];

  const chartData = view === 'weekday' 
    ? {
        labels: weekdayOrder.filter(day => data.by_weekday[day]),
        datasets: [{
          label: 'Průměrná návštěvnost',
          data: weekdayOrder.filter(day => data.by_weekday[day]).map(day => data.by_weekday[day]),
          backgroundColor: weekdayOrder.filter(day => data.by_weekday[day]).map((day, idx) => 
            ['Sobota', 'Neděle'].includes(day) 
              ? 'rgba(255, 99, 132, 0.6)' 
              : 'rgba(0, 102, 204, 0.6)'
          ),
          borderColor: weekdayOrder.filter(day => data.by_weekday[day]).map((day, idx) => 
            ['Sobota', 'Neděle'].includes(day) 
              ? 'rgb(255, 99, 132)' 
              : 'rgb(0, 102, 204)'
          ),
          borderWidth: 2,
        }]
      }
    : {
        labels: monthOrder.filter(month => data.by_month[month]),
        datasets: [{
          label: 'Průměrná návštěvnost',
          data: monthOrder.filter(month => data.by_month[month]).map(month => data.by_month[month]),
          backgroundColor: 'rgba(0, 102, 204, 0.6)',
          borderColor: 'rgb(0, 102, 204)',
          borderWidth: 2,
        }]
      };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
      title: {
        display: true,
        text: view === 'weekday' ? 'Návštěvnost podle dne v týdnu' : 'Návštěvnost podle měsíce',
        font: {
          size: 16,
          weight: 'bold' as const,
        },
      },
      tooltip: {
        callbacks: {
          label: function(context: any) {
            return `${context.parsed.y.toLocaleString('cs-CZ')} návštěvníků`;
          }
        }
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        ticks: {
          callback: function(value: any) {
            return value.toLocaleString('cs-CZ');
          }
        }
      }
    }
  };

  return (
    <div className="bg-white dark:bg-gray-800 shadow-sm ring-1 ring-gray-900/5 dark:ring-gray-700 sm:rounded-xl p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
          Sezonalita návštěvnosti
        </h3>
        <div className="flex gap-2">
          <button
            onClick={() => setView('weekday')}
            className={`px-3 py-1.5 text-sm font-medium rounded-lg transition-colors ${
              view === 'weekday'
                ? 'bg-techmania-blue text-white'
                : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
            }`}
          >
            Den v týdnu
          </button>
          <button
            onClick={() => setView('month')}
            className={`px-3 py-1.5 text-sm font-medium rounded-lg transition-colors ${
              view === 'month'
                ? 'bg-techmania-blue text-white'
                : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
            }`}
          >
            Měsíc
          </button>
        </div>
      </div>

      <div style={{ height: '300px' }}>
        <Bar options={options} data={chartData} />
      </div>

      {/* Holiday vs Regular Stats */}
      {data.holiday_vs_regular.holiday_avg > 0 && (
        <div className="mt-6 grid grid-cols-3 gap-4">
          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
            <div className="text-sm text-gray-600 dark:text-gray-400">Svátky</div>
            <div className="text-xl font-bold text-gray-900 dark:text-white">
              {data.holiday_vs_regular.holiday_avg.toLocaleString('cs-CZ', { maximumFractionDigits: 0 })}
            </div>
          </div>
          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
            <div className="text-sm text-gray-600 dark:text-gray-400">Běžný den</div>
            <div className="text-xl font-bold text-gray-900 dark:text-white">
              {data.holiday_vs_regular.regular_avg.toLocaleString('cs-CZ', { maximumFractionDigits: 0 })}
            </div>
          </div>
          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
            <div className="text-sm text-gray-600 dark:text-gray-400">Rozdíl</div>
            <div className={`text-xl font-bold ${
              data.holiday_vs_regular.difference > 0 ? 'text-green-600' : 'text-red-600'
            }`}>
              {data.holiday_vs_regular.difference > 0 ? '+' : ''}
              {data.holiday_vs_regular.difference.toLocaleString('cs-CZ', { maximumFractionDigits: 0 })}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
