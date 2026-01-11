'use client';

import { useEffect, useState } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import { Chart } from 'react-chartjs-2';
import { api } from '@/lib/api';
import type { HistoricalDataResponse, TodayVisitorsResponse, TimeRange, RangePredictionResponse, PredictionHistoryResponse } from '@/types/api';
import { useTranslations } from '@/lib/i18n';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
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
  const [predictionHistory, setPredictionHistory] = useState<PredictionHistoryResponse | null>(null);
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
        
        // Načteme historii predikcí - porovnání predikce vs. realita
        try {
          const history = await api.getPredictionHistory(historyDaysMap[timeRange], true);
          console.log('Prediction history loaded:', history);
          console.log('Non-future predictions:', history.history.filter(p => !p.is_future).length);
          setPredictionHistory(history);
        } catch (historyErr) {
          console.log('Prediction history not available:', historyErr);
          setPredictionHistory(null);
        }
        
        // Načteme predikci pouze od dneška dopředu
        // DŮLEŽITÉ: Používáme uložené predikce z databáze, NEvoláme novou predikci!
        // Nové predikce se vytváří pouze přes RangePredictionForm
        try {
          const savedPredictions = await api.getLatestPredictions(15);
          
          // Filtrovat pouze budoucí predikce (od dneška)
          const now = new Date();
          now.setHours(0, 0, 0, 0);
          
          const futurePredictions = savedPredictions.predictions
            .filter((pred: any) => {
              const predDate = new Date(pred.date);
              predDate.setHours(0, 0, 0, 0);
              return predDate >= now;
            })
            .map((pred: any) => ({
              date: pred.date,
              predicted_visitors: pred.predicted_visitors,
              confidence_interval: pred.confidence_interval,
              day_of_week: new Date(pred.date).toLocaleDateString('cs-CZ', { weekday: 'long' }),
              is_weekend: [0, 6].includes(new Date(pred.date).getDay()),
              weather_info: {
                temperature_mean: pred.temperature_mean || 0,
                precipitation: pred.precipitation || 0,
                is_nice_weather: pred.is_nice_weather || false,
                weather_description: ''
              },
              holiday_info: {
                is_holiday: false,
                holiday_name: null
              }
            }));
          
          if (futurePredictions.length > 0) {
            setFutureData({
              predictions: futurePredictions,
              total_predicted: futurePredictions.reduce((sum: number, p: any) => sum + p.predicted_visitors, 0),
              average_daily: futurePredictions.reduce((sum: number, p: any) => sum + p.predicted_visitors, 0) / futurePredictions.length
            });
          } else {
            setFutureData(null);
          }
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
            if (context.parsed.y === null || context.parsed.y === undefined) {
              return null; // Skryje tooltip pro null hodnoty
            }
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
    historicalPrediction?: number; // Historická predikce pro porovnání
    historicalPredLower?: number; // Dolní interval historické predikce
    historicalPredUpper?: number; // Horní interval historické predikce
    lower?: number;
    upper?: number;
  }>();

  // Přidáme historická data
  historicalData.data.forEach(d => {
    dateMap.set(d.date, { historical: d.visitors });
  });

  // Přidáme historické predikce (predikce pro datumy, kde máme i skutečná data)
  predictionHistory?.history.forEach(p => {
    const existing = dateMap.get(p.date);
    // Zobrazíme predikci pokud:
    // 1. Není to budoucnost (skutečná historická predikce)
    // 2. NEBO pokud máme pro tento datum skutečná data (můžeme porovnat)
    if (!p.is_future || existing?.historical !== undefined) {
      dateMap.set(p.date, {
        ...existing,
        historicalPrediction: p.predicted,
        historicalPredLower: p.confidence_lower ?? undefined,
        historicalPredUpper: p.confidence_upper ?? undefined
      });
    }
  });

  // Debug - kolik historických predikcí máme
  const historicalPredCount = Array.from(dateMap.values()).filter(v => v.historicalPrediction !== undefined).length;
  console.log('Historical predictions in dateMap:', historicalPredCount);

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

  const historicalPredictions = sortedDates.map(date => 
    dateMap.get(date)?.historicalPrediction ?? null
  );

  const historicalPredLower = sortedDates.map(date => 
    dateMap.get(date)?.historicalPredLower ?? null
  );

  const historicalPredUpper = sortedDates.map(date => 
    dateMap.get(date)?.historicalPredUpper ?? null
  );

  const confidenceLower = sortedDates.map(date => 
    dateMap.get(date)?.lower ?? null
  );

  const confidenceUpper = sortedDates.map(date => 
    dateMap.get(date)?.upper ?? null
  );

  // Přidáme dnešní real-time hodnotu pouze pokud jsou to skutečná data (ne predikce)
  // todayData obsahuje is_historical informaci z API, ale ta se ztratí při transformaci
  // Proto kontrolujeme, zda datum existuje v historických datech
  if (todayData && dateMap.has(todayData.date) && dateMap.get(todayData.date)?.historical !== undefined) {
    const todayIndex = sortedDates.indexOf(todayData.date);
    if (todayIndex !== -1) {
      historicalVisitors[todayIndex] = todayData.current_visitors;
    }
  }

  const datasets = [
    {
      type: 'bar' as const,
      label: t('actualVisitors') || 'Skutečné návštěvy',
      data: historicalVisitors,
      borderColor: 'rgb(0, 102, 204)',
      backgroundColor: 'rgba(0, 102, 204, 0.6)',
      borderWidth: 1,
      borderRadius: 4,
      order: 2, // Sloupce se vykreslí za liniemi
    },
    {
      type: 'line' as const,
      label: t('historicalPrediction') || 'Historická predikce',
      data: historicalPredictions,
      borderColor: 'rgb(249, 115, 22)', // Oranžová barva
      backgroundColor: 'rgba(249, 115, 22, 0.1)',
      borderDash: [3, 3],
      fill: false,
      tension: 0.4,
      borderWidth: 2,
      pointRadius: 3,
      pointHoverRadius: 5,
      order: 0,
    },
    {
      type: 'line' as const,
      label: 'Hist. predikce - horní interval',
      data: historicalPredUpper,
      borderColor: 'rgba(249, 115, 22, 0.3)',
      backgroundColor: 'rgba(249, 115, 22, 0.05)',
      fill: '+1',
      tension: 0.4,
      borderWidth: 1,
      pointRadius: 0,
      order: 1,
    },
    {
      type: 'line' as const,
      label: 'Hist. predikce - dolní interval',
      data: historicalPredLower,
      borderColor: 'rgba(249, 115, 22, 0.3)',
      backgroundColor: 'rgba(249, 115, 22, 0.05)',
      fill: false,
      tension: 0.4,
      borderWidth: 1,
      pointRadius: 0,
      order: 1,
    },
    {
      type: 'line' as const,
      label: t('predictedVisitors') || 'Budoucí předpověď',
      data: predictedVisitors,
      borderColor: 'rgb(34, 197, 94)',
      backgroundColor: 'rgba(34, 197, 94, 0.1)',
      borderDash: [5, 5],
      fill: false,
      tension: 0.4,
      borderWidth: 2,
      pointRadius: 3,
      pointHoverRadius: 5,
      order: 0,
    },
    {
      type: 'line' as const,
      label: t('confidenceUpper') || 'Horní interval',
      data: confidenceUpper,
      borderColor: 'rgba(34, 197, 94, 0.3)',
      backgroundColor: 'rgba(34, 197, 94, 0.05)',
      fill: '+1',
      tension: 0.4,
      borderWidth: 1,
      pointRadius: 0,
      order: 1,
    },
    {
      type: 'line' as const,
      label: t('confidenceLower') || 'Dolní interval',
      data: confidenceLower,
      borderColor: 'rgba(34, 197, 94, 0.3)',
      backgroundColor: 'rgba(34, 197, 94, 0.05)',
      fill: false,
      tension: 0.4,
      borderWidth: 1,
      pointRadius: 0,
      order: 1,
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

        {/* Today's Stats - zobrazíme pouze pokud máme skutečná data pro dnešek */}
        {todayData && historicalData.data.some(d => d.date === todayData.date) ? (
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
        ) : (
          <div className="text-right">
            <div className="text-sm text-gray-600 dark:text-gray-400">Dnes</div>
            <div className="text-sm text-gray-500 dark:text-gray-400">
              Žádná data
            </div>
          </div>
        )}
      </div>

      <div style={{ height: '400px' }}>
        <Chart type="bar" options={options} data={data} />
      </div>

      {/* Prediction Accuracy Stats */}
      {predictionHistory && predictionHistory.summary && predictionHistory.summary.valid_comparisons > 0 && (
        <div className="mt-6 pt-4 border-t border-gray-200 dark:border-gray-700">
          <div className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            {t('predictionAccuracy') || 'Přesnost predikcí'}
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-3">
              <div className="text-xs text-gray-500 dark:text-gray-400">
                {t('accuracyLabel') || 'Přesnost (±10%)'}
              </div>
              <div className="text-lg font-semibold text-green-600 dark:text-green-400">
                {predictionHistory.summary.accuracy_10_percent?.toFixed(1) || 0}%
              </div>
            </div>
            <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-3">
              <div className="text-xs text-gray-500 dark:text-gray-400">
                Přesnost (±20%)
              </div>
              <div className="text-lg font-semibold text-blue-600 dark:text-blue-400">
                {predictionHistory.summary.accuracy_20_percent?.toFixed(1) || 0}%
              </div>
            </div>
            <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-3">
              <div className="text-xs text-gray-500 dark:text-gray-400">
                {t('avgError') || 'Průměrná odchylka'}
              </div>
              <div className="text-lg font-semibold text-gray-900 dark:text-white">
                {predictionHistory.summary.avg_error_percent?.toFixed(1) || 0}%
              </div>
            </div>
            <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-3">
              <div className="text-xs text-gray-500 dark:text-gray-400">
                Porovnáno dnů
              </div>
              <div className="text-lg font-semibold text-gray-900 dark:text-white">
                {predictionHistory.summary.valid_comparisons}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
