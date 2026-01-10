'use client';

import { useEffect, useState } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import { api } from '@/lib/api';
import type { HistoricalDataResponse } from '@/types/api';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

export default function TrendAnalysis() {
  const [data, setData] = useState<HistoricalDataResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showMovingAverage, setShowMovingAverage] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const historicalData = await api.getHistoricalData();
        setData(historicalData);
      } catch (err: any) {
        console.error('Failed to fetch historical data:', err);
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

  // Výpočet klouzavého průměru (7 dní)
  const calculateMovingAverage = (data: number[], window: number) => {
    const result: (number | null)[] = [];
    for (let i = 0; i < data.length; i++) {
      if (i < window - 1) {
        result.push(null);
      } else {
        const sum = data.slice(i - window + 1, i + 1).reduce((a, b) => a + b, 0);
        result.push(sum / window);
      }
    }
    return result;
  };

  const visitors = data.data.map(d => d.visitors);
  const movingAvg = calculateMovingAverage(visitors, 7);

  const labels = data.data.map(d => {
    const date = new Date(d.date);
    return `${date.getDate()}.${date.getMonth() + 1}`;
  });

  const chartData = {
    labels,
    datasets: [
      {
        label: 'Denní návštěvnost',
        data: visitors,
        borderColor: 'rgba(0, 102, 204, 0.4)',
        backgroundColor: 'rgba(0, 102, 204, 0.1)',
        fill: false,
        tension: 0.1,
        borderWidth: 1,
        pointRadius: 0,
      },
      ...(showMovingAverage ? [{
        label: '7-denní klouzavý průměr',
        data: movingAvg,
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.1)',
        fill: false,
        tension: 0.4,
        borderWidth: 3,
        pointRadius: 0,
      }] : [])
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index' as const,
      intersect: false,
    },
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: true,
        text: 'Dlouhodobý trend návštěvnosti',
        font: {
          size: 16,
          weight: 'bold' as const,
        },
      },
      tooltip: {
        callbacks: {
          label: function(context: any) {
            return `${context.dataset.label}: ${context.parsed.y.toLocaleString('cs-CZ')}`;
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

  // Výpočet statistik trendu
  const avgVisitors = visitors.reduce((a, b) => a + b, 0) / visitors.length;
  const maxVisitors = Math.max(...visitors);
  const minVisitors = Math.min(...visitors);
  
  // Jednoduchý lineární trend (poslední vs první měsíc)
  const firstMonthAvg = visitors.slice(0, 30).reduce((a, b) => a + b, 0) / 30;
  const lastMonthAvg = visitors.slice(-30).reduce((a, b) => a + b, 0) / 30;
  const trendPercentage = ((lastMonthAvg - firstMonthAvg) / firstMonthAvg) * 100;

  return (
    <div className="bg-white dark:bg-gray-800 shadow-sm ring-1 ring-gray-900/5 dark:ring-gray-700 sm:rounded-xl p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
          Analýza trendu
        </h3>
        <button
          onClick={() => setShowMovingAverage(!showMovingAverage)}
          className="px-3 py-1.5 text-sm font-medium rounded-lg bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
        >
          {showMovingAverage ? 'Skrýt' : 'Zobrazit'} klouzavý průměr
        </button>
      </div>

      <div style={{ height: '300px' }}>
        <Line options={options} data={chartData} />
      </div>

      {/* Statistiky */}
      <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-3">
          <div className="text-xs text-gray-600 dark:text-gray-400">Průměr</div>
          <div className="text-lg font-bold text-gray-900 dark:text-white">
            {avgVisitors.toLocaleString('cs-CZ', { maximumFractionDigits: 0 })}
          </div>
        </div>
        <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-3">
          <div className="text-xs text-gray-600 dark:text-gray-400">Maximum</div>
          <div className="text-lg font-bold text-gray-900 dark:text-white">
            {maxVisitors.toLocaleString('cs-CZ')}
          </div>
        </div>
        <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-3">
          <div className="text-xs text-gray-600 dark:text-gray-400">Minimum</div>
          <div className="text-lg font-bold text-gray-900 dark:text-white">
            {minVisitors.toLocaleString('cs-CZ')}
          </div>
        </div>
        <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-3">
          <div className="text-xs text-gray-600 dark:text-gray-400">Trend</div>
          <div className={`text-lg font-bold ${trendPercentage > 0 ? 'text-green-600' : 'text-red-600'}`}>
            {trendPercentage > 0 ? '+' : ''}{trendPercentage.toFixed(1)}%
          </div>
        </div>
      </div>
    </div>
  );
}
