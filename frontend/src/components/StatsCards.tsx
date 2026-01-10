'use client';

import { useEffect, useState } from 'react';
import { 
  UsersIcon, 
  CalendarDaysIcon, 
  ChartBarSquareIcon,
  ArrowTrendingUpIcon 
} from '@heroicons/react/24/outline';
import { api } from '@/lib/api';
import type { StatsResponse } from '@/types/api';
import { useTranslations } from '@/lib/i18n';

export default function StatsCards() {
  const t = useTranslations('stats');
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const data = await api.getStats();
        setStats(data);
      } catch (err: any) {
        console.error('Failed to fetch stats:', err);
        setError(err.response?.data?.detail || t('error'));
      } finally {
        setLoading(false);
      }
    };

    fetchStats();
  }, []);

  if (loading) {
    return (
      <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="overflow-hidden rounded-lg bg-white shadow animate-pulse">
            <div className="p-6">
              <div className="h-20 bg-gray-200 rounded"></div>
            </div>
          </div>
        ))}
      </div>
    );
  }

  if (error || !stats) {
    return (
      <div className="rounded-lg bg-red-50 dark:bg-red-900/20 p-4">
        <p className="text-sm text-red-800 dark:text-red-300">
          {error || t('notAvailable')}
        </p>
      </div>
    );
  }

  const cards = [
    {
      name: t('totalVisitors'),
      value: stats.total_visitors.toLocaleString('cs-CZ'),
      icon: UsersIcon,
      change: `${t('period')}: ${new Date(stats.data_start_date).toLocaleDateString('cs-CZ')} - ${new Date(stats.data_end_date).toLocaleDateString('cs-CZ')}`,
      changeType: 'neutral' as const,
    },
    {
      name: t('avgDaily'),
      value: Math.round(stats.avg_daily_visitors).toLocaleString('cs-CZ'),
      icon: ChartBarSquareIcon,
      change: `${stats.trend > 0 ? '+' : ''}${stats.trend}% ${t('trend')}`,
      changeType: stats.trend > 0 ? 'positive' as const : stats.trend < 0 ? 'negative' as const : 'neutral' as const,
    },
    {
      name: t('peakDay'),
      value: stats.peak_day,
      icon: CalendarDaysIcon,
      change: `${stats.peak_visitors.toLocaleString('cs-CZ')} ${t('visitors')}`,
      changeType: 'neutral' as const,
    },
    {
      name: t('monthlyTrend'),
      value: `${stats.trend > 0 ? '+' : ''}${stats.trend}%`,
      icon: ArrowTrendingUpIcon,
      change: stats.trend > 0 ? t('rising') : stats.trend < 0 ? t('falling') : t('stable'),
      changeType: stats.trend > 0 ? 'positive' as const : stats.trend < 0 ? 'negative' as const : 'neutral' as const,
    },
  ];

  return (
    <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
      {cards.map((card) => (
        <div
          key={card.name}
          className="overflow-hidden rounded-lg bg-white dark:bg-gray-800 shadow hover:shadow-lg transition-shadow"
        >
          <div className="p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <card.icon className="h-8 w-8 text-techmania-blue" aria-hidden="true" />
              </div>
              <div className="ml-5 w-0 flex-1">
                <dl>
                  <dt className="truncate text-sm font-medium text-gray-500 dark:text-gray-400">{card.name}</dt>
                  <dd>
                    <div className="text-2xl font-bold text-gray-900 dark:text-white">{card.value}</div>
                  </dd>
                </dl>
              </div>
            </div>
          </div>
          <div className="bg-gray-50 dark:bg-gray-900 px-6 py-3">
            <div className="text-sm">
              <span
                className={
                  card.changeType === 'positive'
                    ? 'font-medium text-green-600'
                    : card.changeType === 'negative'
                    ? 'font-medium text-red-600'
                    : 'font-medium text-gray-600'
                }
              >
                {card.change}
              </span>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
