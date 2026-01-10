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
import type { HistoricalDataResponse, TodayVisitorsResponse, TimeRange, RangePredictionResponse } from '@/types/api';
import { useTranslations } from '@/lib/i18n';

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

export default function VisitorChart() {
  const t = useTranslations('chart');
  const [historicalData, setHistoricalData] = useState<HistoricalDataResponse | null>(null);
  const [todayData, setTodayData] = useState<TodayVisitorsResponse | null>(null);
  const [futureData, setFutureData] = useState<RangePredictionResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [timeRange, setTimeRange] = useState<'week' | 'month' | 'quarter' | 'year'>('month');

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      setError(null);
      try {
        // Určíme kolik dní zpátky pro historii a predikci
        const historyDaysMap = {
          week: 7,
          month: 30,
          quarter: 90,
          year: 365
        };
        
        // Načteme historická reálná data
        const historical = await api.getHistoricalData(historyDaysMap[timeRange]);
        setHistoricalData(historical);
        
        // Pokusíme se načíst dnešní data (může selhat, není kritické)
        try {
          const today = await api.getTodayVisitors();
          setTodayData(today);
        } catch (todayErr) {
          console.log('Today data not available:', todayErr);
          setTodayData(null);
        }
        
        // Načteme predikci pouze od dneška dopředu
        // Pro minulost používáme skutečná historická data (už načtená výše)
        // Weather forecast API nepodporuje predikce pro minulé dny
        try {
          const now = new Date();
          const end = new Date(now);
          end.setDate(end.getDate() + 10); // 10 dní dopředu
          
          // Formátujeme datumy ve formátu YYYY-MM-DD (stejně jako v RangePredictionForm)
          const formatDateString = (date: Date): string => {
            const year = date.getFullYear();
            const month = String(date.getMonth() + 1).padStart(2, '0');
            const day = String(date.getDate()).padStart(2, '0');
            return `${year}-${month}-${day}`;
          };
          
          const startDate = formatDateString(now);
          const endDate = formatDateString(end);
          
          const prediction = await api.predictRange({
            start_date: startDate,
            end_date: endDate,
          });
          setFutureData(prediction);
        } catch (futureErr) {
          console.log('Future predictions not available:', futureErr);
          setFutureData(null);
        }
        
      } catch (err: any) {
        console.error('Failed to fetch historical data:', err);
        setError(err.message || t('error'));
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [timeRange]);

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
        labels: {
          usePointStyle: true,
          padding: 15,
          font: {
            size: 12
          }
        }
      },
      title: {
        display: true,
        text: t('title'),
        font: {
          size: 18,
          weight: 'bold' as const,
        },
        padding: {
          bottom: 20
        }
      },
      tooltip: {
        callbacks: {
          label: function(context: any) {
            let label = context.dataset.label || '';
            if (label) {
              label += ': ';
            }
            label += context.parsed.y.toLocaleString('cs-CZ');
            return label;
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
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.05)'
        }
      },
      x: {
        grid: {
          display: false
        }
      }
    }
  };

  if (loading) {
    return (
      <div className="bg-white dark:bg-gray-800 shadow-sm ring-1 ring-gray-900/5 dark:ring-gray-700 sm:rounded-xl p-6">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-200 dark:bg-gray-700 rounded w-1/3 mb-4"></div>
          <div style={{ height: '400px' }} className="bg-gray-100 dark:bg-gray-700 rounded"></div>
        </div>
      </div>
    );
  }

  if (error || !historicalData) {
    return (
      <div className="bg-white dark:bg-gray-800 shadow-sm ring-1 ring-gray-900/5 dark:ring-gray-700 sm:rounded-xl p-6">
        <div className="rounded-lg bg-red-50 dark:bg-red-900/20 p-4">
          <p className="text-sm text-red-800 dark:text-red-300">
            {error || t('notAvailable')}
          </p>
        </div>
      </div>
    );
  }

  // Připravíme data pro graf - spojíme všechna data podle datumu
  const dateMap = new Map<string, {
    historical?: number;
    predicted?: number;
    lower?: number;
    upper?: number;
  }>();

  // Přidáme historická data
  historicalData.data.forEach(d => {
    dateMap.set(d.date, { historical: d.visitors });
  });

  // Přidáme predikce - mohou se překrývat s historickými daty
  futureData?.predictions.forEach(p => {
    const existing = dateMap.get(p.date) || {};
    dateMap.set(p.date, {
      ...existing,
      predicted: p.predicted_visitors,
      lower: p.confidence_interval.lower,
      upper: p.confidence_interval.upper
    });
  });

  // Seřadíme datumy a vytvoříme pole pro graf
  const sortedDates = Array.from(dateMap.keys()).sort();
  
  const labels = sortedDates.map(date => {
    const d = new Date(date);
    return `${d.getDate()}.${d.getMonth() + 1}`;
  });

  const historicalVisitors = sortedDates.map(date => 
    dateMap.get(date)?.historical ?? null
  );

  const predictedVisitors = sortedDates.map(date => 
    dateMap.get(date)?.predicted ?? null
  );

  const confidenceLower = sortedDates.map(date => 
    dateMap.get(date)?.lower ?? null
  );

  const confidenceUpper = sortedDates.map(date => 
    dateMap.get(date)?.upper ?? null
  );

  // Přidáme dnešní real-time hodnotu pokud je dostupná
  if (todayData) {
    const todayIndex = sortedDates.indexOf(todayData.date);
    if (todayIndex !== -1) {
      historicalVisitors[todayIndex] = todayData.current_visitors;
    }
  }

  const datasets = [
    {
      label: t('actualVisitors') || 'Skutečné návštěvy',
      data: historicalVisitors,
      borderColor: 'rgb(0, 102, 204)',
      backgroundColor: 'rgba(0, 102, 204, 0.1)',
      fill: false,
      tension: 0.4,
      borderWidth: 2,
      pointRadius: 3,
      pointHoverRadius: 5,
    },
    {
      label: t('predictedVisitors') || 'Předpověď',
      data: predictedVisitors,
      borderColor: 'rgb(34, 197, 94)',
      backgroundColor: 'rgba(34, 197, 94, 0.1)',
      borderDash: [5, 5],
      fill: false,
      tension: 0.4,
      borderWidth: 2,
      pointRadius: 3,
      pointHoverRadius: 5,
    },
    {
      label: t('confidenceUpper') || 'Horní interval',
      data: confidenceUpper,
      borderColor: 'rgba(34, 197, 94, 0.3)',
      backgroundColor: 'rgba(34, 197, 94, 0.05)',
      fill: '+1',
      tension: 0.4,
      borderWidth: 1,
      pointRadius: 0,
    },
    {
      label: t('confidenceLower') || 'Dolní interval',
      data: confidenceLower,
      borderColor: 'rgba(34, 197, 94, 0.3)',
      backgroundColor: 'rgba(34, 197, 94, 0.05)',
      fill: false,
      tension: 0.4,
      borderWidth: 1,
      pointRadius: 0,
    }
  ];

  const data = {
    labels,
    datasets
  };

  return (
    <div className="bg-white dark:bg-gray-800 shadow-sm ring-1 ring-gray-900/5 dark:ring-gray-700 sm:rounded-xl p-6">
      {/* Time Range Selector */}
      <div className="mb-6 flex items-center justify-between">
        <div className="flex gap-2">
          <button
            onClick={() => setTimeRange('week')}
            className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
              timeRange === 'week'
                ? 'bg-techmania-blue text-white'
                : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
            }`}
          >
            Týden
          </button>
          <button
            onClick={() => setTimeRange('month')}
            className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
              timeRange === 'month'
                ? 'bg-techmania-blue text-white'
                : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
            }`}
          >
            Měsíc
          </button>
          <button
            onClick={() => setTimeRange('quarter')}
            className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
              timeRange === 'quarter'
                ? 'bg-techmania-blue text-white'
                : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
            }`}
          >
            Kvartál
          </button>
          <button
            onClick={() => setTimeRange('year')}
            className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
              timeRange === 'year'
                ? 'bg-techmania-blue text-white'
                : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
            }`}
          >
            Rok
          </button>
        </div>

        {/* Today's Stats */}
        {todayData && (
          <div className="text-right">
            <div className="text-sm text-gray-600 dark:text-gray-400">Dnes aktuálně</div>
            <div className="text-2xl font-bold text-gray-900 dark:text-white">
              {todayData.current_visitors.toLocaleString('cs-CZ')}
            </div>
            <div className={`text-xs ${
              todayData.current_visitors > todayData.predicted_visitors 
                ? 'text-green-600' 
                : 'text-orange-600'
            }`}>
              Předpověď: {todayData.predicted_visitors.toLocaleString('cs-CZ')}
            </div>
          </div>
        )}
      </div>

      <div style={{ height: '400px' }}>
        <Line options={options} data={data} />
      </div>
    </div>
  );
}
