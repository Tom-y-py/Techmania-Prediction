'use client';

import { useEffect, useState } from 'react';
import { 
  UsersIcon, 
  CalendarDaysIcon, 
  ChartBarSquareIcon,
  ArrowTrendingUpIcon 
} from '@heroicons/react/24/outline';

interface StatsData {
  totalVisitors: number;
  avgDailyVisitors: number;
  peakDay: string;
  trend: number;
}

export default function StatsCards() {
  const [stats, setStats] = useState<StatsData>({
    totalVisitors: 0,
    avgDailyVisitors: 0,
    peakDay: '-',
    trend: 0,
  });

  useEffect(() => {
    // Mock data - v reálné aplikaci by se načetla z API
    setStats({
      totalVisitors: 125430,
      avgDailyVisitors: 3421,
      peakDay: '15. prosince 2025',
      trend: 12.5,
    });
  }, []);

  const cards = [
    {
      name: 'Celkový počet návštěvníků',
      value: stats.totalVisitors.toLocaleString('cs-CZ'),
      icon: UsersIcon,
      change: `+${stats.trend}%`,
      changeType: 'positive',
    },
    {
      name: 'Průměr návštěvníků/den',
      value: stats.avgDailyVisitors.toLocaleString('cs-CZ'),
      icon: ChartBarSquareIcon,
      change: '+4.2%',
      changeType: 'positive',
    },
    {
      name: 'Den s nejvyšší návštěvností',
      value: stats.peakDay,
      icon: CalendarDaysIcon,
      change: '8,542 návštěvníků',
      changeType: 'neutral',
    },
    {
      name: 'Měsíční trend',
      value: `+${stats.trend}%`,
      icon: ArrowTrendingUpIcon,
      change: 'Rostoucí tendence',
      changeType: 'positive',
    },
  ];

  return (
    <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
      {cards.map((card) => (
        <div
          key={card.name}
          className="overflow-hidden rounded-lg bg-white shadow hover:shadow-lg transition-shadow"
        >
          <div className="p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <card.icon className="h-8 w-8 text-techmania-blue" aria-hidden="true" />
              </div>
              <div className="ml-5 w-0 flex-1">
                <dl>
                  <dt className="truncate text-sm font-medium text-gray-500">{card.name}</dt>
                  <dd>
                    <div className="text-2xl font-bold text-gray-900">{card.value}</div>
                  </dd>
                </dl>
              </div>
            </div>
          </div>
          <div className="bg-gray-50 px-6 py-3">
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
