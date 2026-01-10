'use client';

import { useState, useEffect } from 'react';
import { CalendarDaysIcon, ArrowRightIcon } from '@heroicons/react/24/outline';
import { api } from '@/lib/api';
import { useTranslations } from '@/lib/i18n';
import Link from 'next/link';

interface UpcomingEvent {
  date: string;
  name: string;
  type: 'holiday' | 'vacation' | 'high_traffic';
  visitors: number;
  dayOfWeek: string;
}

export default function UpcomingEvents() {
  const t = useTranslations('upcomingEvents');
  const [events, setEvents] = useState<UpcomingEvent[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadEvents();
  }, []);

  const loadEvents = async () => {
    try {
      const today = new Date();
      const endDate = new Date(today);
      endDate.setDate(endDate.getDate() + 14);

      const response = await api.predictRange({
        start_date: today.toISOString().split('T')[0],
        end_date: endDate.toISOString().split('T')[0]
      });

      const newEvents: UpcomingEvent[] = [];

      response.predictions.forEach(pred => {
        // Svátek
        if (pred.holiday_info?.is_holiday && pred.holiday_info?.holiday_name) {
          newEvents.push({
            date: pred.date,
            name: pred.holiday_info.holiday_name,
            type: 'holiday',
            visitors: pred.predicted_visitors,
            dayOfWeek: pred.day_of_week
          });
        }

        // Vysoká návštěvnost (> 600) - ale jen pokud to není už svátek
        if (pred.predicted_visitors > 600 && !pred.holiday_info?.is_holiday) {
          newEvents.push({
            date: pred.date,
            name: t('highTraffic') || 'Vysoká návštěvnost',
            type: 'high_traffic',
            visitors: pred.predicted_visitors,
            dayOfWeek: pred.day_of_week
          });
        }
      });

      // Seřadit podle data a omezit na 4 události
      setEvents(newEvents.slice(0, 4));
    } catch (error) {
      console.error('Error loading events:', error);
    }
    setLoading(false);
  };

  const getEventColor = (type: string) => {
    switch (type) {
      case 'holiday':
        return 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300 border-red-200 dark:border-red-800';
      case 'vacation':
        return 'bg-orange-100 dark:bg-orange-900/30 text-orange-700 dark:text-orange-300 border-orange-200 dark:border-orange-800';
      case 'high_traffic':
        return 'bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300 border-amber-200 dark:border-amber-800';
      default:
        return 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 border-blue-200 dark:border-blue-800';
    }
  };

  const getEventIcon = (type: string) => {
    switch (type) {
      case 'holiday':
        return '🎉';
      case 'vacation':
        return '🏖️';
      case 'high_traffic':
        return '📈';
      default:
        return '📅';
    }
  };

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const eventDate = new Date(date);
    eventDate.setHours(0, 0, 0, 0);
    
    const diffDays = Math.floor((eventDate.getTime() - today.getTime()) / (1000 * 60 * 60 * 24));
    
    if (diffDays === 0) return t('today') || 'Dnes';
    if (diffDays === 1) return t('tomorrow') || 'Zítra';
    if (diffDays < 7) return `${t('in') || 'Za'} ${diffDays} ${t('days') || 'dní'}`;
    
    return date.toLocaleDateString('cs-CZ', { day: 'numeric', month: 'short' });
  };

  if (loading) {
    return (
      <div className="bg-white dark:bg-gray-800 shadow-sm ring-1 ring-gray-900/5 dark:ring-gray-700 sm:rounded-xl p-6">
        <div className="animate-pulse">
          <div className="h-6 bg-gray-200 dark:bg-gray-700 rounded w-1/3 mb-4"></div>
          <div className="space-y-3">
            <div className="h-16 bg-gray-100 dark:bg-gray-700 rounded"></div>
            <div className="h-16 bg-gray-100 dark:bg-gray-700 rounded"></div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-gray-800 shadow-sm ring-1 ring-gray-900/5 dark:ring-gray-700 sm:rounded-xl p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center gap-2">
          <CalendarDaysIcon className="h-5 w-5 text-blue-500" />
          {t('title') || 'Nadcházející události'}
        </h3>
        <Link 
          href="/calendar" 
          className="text-sm text-blue-600 hover:text-blue-800 dark:text-blue-400 flex items-center gap-1"
        >
          {t('viewAll') || 'Zobrazit vše'}
          <ArrowRightIcon className="h-4 w-4" />
        </Link>
      </div>

      {events.length > 0 ? (
        <div className="space-y-3">
          {events.map((event) => (
            <div
              key={event.date + event.type}
              className={`flex items-center gap-4 p-3 rounded-lg border ${getEventColor(event.type)}`}
            >
              <span className="text-2xl">{getEventIcon(event.type)}</span>
              <div className="flex-1 min-w-0">
                <p className="font-medium truncate">{event.name}</p>
                <p className="text-sm opacity-80">
                  {event.dayOfWeek} • {event.visitors.toLocaleString('cs-CZ')} {t('visitors') || 'návštěvníků'}
                </p>
              </div>
              <div className="text-right">
                <p className="text-sm font-medium">{formatDate(event.date)}</p>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <p className="text-sm text-gray-500 dark:text-gray-400 text-center py-4">
          {t('noEvents') || 'Žádné nadcházející události'}
        </p>
      )}
    </div>
  );
}
