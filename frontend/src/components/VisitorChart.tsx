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

export default function VisitorChart() {
  const [historicalData, setHistoricalData] = useState<HistoricalDataResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchHistoricalData = async () => {
      try {
        const data = await api.getHistoricalData(30);
        setHistoricalData(data);
      } catch (err: any) {
        console.error('Failed to fetch historical data:', err);
        setError(err.response?.data?.detail || 'Nepodařilo se načíst historická data');
      } finally {
        setLoading(false);
      }
    };

    fetchHistoricalData();
  }, []);

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: true,
        text: 'Návštěvnost za poslední měsíc',
        font: {
          size: 16,
          weight: 'bold' as const,
        },
      },
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

  if (loading) {
    return (
      <div className="bg-white shadow-sm ring-1 ring-gray-900/5 sm:rounded-xl p-6">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-200 rounded w-1/3 mb-4"></div>
          <div style={{ height: '400px' }} className="bg-gray-100 rounded"></div>
        </div>
      </div>
    );
  }

  if (error || !historicalData) {
    return (
      <div className="bg-white shadow-sm ring-1 ring-gray-900/5 sm:rounded-xl p-6">
        <div className="rounded-lg bg-red-50 p-4">
          <p className="text-sm text-red-800">
            {error || 'Historická data nejsou dostupná'}
          </p>
        </div>
      </div>
    );
  }

  const labels = historicalData.data.map(d => {
    const date = new Date(d.date);
    return `${date.getDate()}.${date.getMonth() + 1}`;
  });

  const visitors = historicalData.data.map(d => d.visitors);

  const data = {
    labels,
    datasets: [
      {
        label: 'Skutečná návštěvnost',
        data: visitors,
        borderColor: 'rgb(0, 102, 204)',
        backgroundColor: 'rgba(0, 102, 204, 0.1)',
        fill: true,
        tension: 0.4,
      },
    ],
  };

  return (
    <div className="bg-white shadow-sm ring-1 ring-gray-900/5 sm:rounded-xl p-6">
      <div style={{ height: '400px' }}>
        <Line options={options} data={data} />
      </div>
    </div>
  );
}
