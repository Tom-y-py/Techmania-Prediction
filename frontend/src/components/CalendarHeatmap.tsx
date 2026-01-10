'use client';

import { useEffect, useState } from 'react';
import { api } from '@/lib/api';
import type { CalendarHeatmapData } from '@/types/api';
import { useTranslations } from '@/lib/i18n';

interface CalendarHeatmapProps {
  defaultYear?: number;
}

export default function CalendarHeatmap({ defaultYear = 2025 }: CalendarHeatmapProps) {
  const t = useTranslations('analytics.calendarHeatmap');
  const [data, setData] = useState<CalendarHeatmapData[]>([]);
  const [availableYears, setAvailableYears] = useState<number[]>([]);
  const [selectedYear, setSelectedYear] = useState<number>(defaultYear);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Načíst data bez specifikace roku, abychom dostali všechny roky
        const response = await fetch('http://localhost:8000/analytics/heatmap');
        const result = await response.json();
        
        setData(result.data || []);
        const years = result.available_years || [];
        setAvailableYears(years);
        
        // Nastavit poslední dostupný rok jako výchozí
        if (years.length > 0 && !defaultYear) {
          setSelectedYear(years[years.length - 1]);
        }
      } catch (err: any) {
        console.error('Failed to fetch heatmap data:', err);
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
          <div className="h-6 bg-gray-200 dark:bg-gray-700 rounded w-1/4 mb-4"></div>
          <div className="h-40 bg-gray-100 dark:bg-gray-700 rounded"></div>
        </div>
      </div>
    );
  }

  if (error || data.length === 0) {
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

  // Filtrovat data podle vybraného roku
  const yearData = data.filter(d => new Date(d.date).getFullYear() === selectedYear);
  
  // Seskupení dat podle měsíců
  const monthsData: { [key: string]: CalendarHeatmapData[] } = {};
  yearData.forEach(d => {
    const date = new Date(d.date);
    const monthKey = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
    if (!monthsData[monthKey]) {
      monthsData[monthKey] = [];
    }
    monthsData[monthKey].push(d);
  });

  // Výpočet rozsahu pro barevnou škálu
  const visitors = yearData.map(d => d.visitors);
  const maxVisitors = visitors.length > 0 ? Math.max(...visitors) : 0;
  const minVisitors = visitors.length > 0 ? Math.min(...visitors) : 0;

  const getColor = (visitors: number) => {
    const normalized = (visitors - minVisitors) / (maxVisitors - minVisitors || 1);
    
    if (normalized < 0.2) return 'bg-blue-100 dark:bg-blue-900/20';
    if (normalized < 0.4) return 'bg-blue-200 dark:bg-blue-800/40';
    if (normalized < 0.6) return 'bg-blue-400 dark:bg-blue-600/60';
    if (normalized < 0.8) return 'bg-blue-600 dark:bg-blue-500/80';
    return 'bg-blue-800 dark:bg-blue-400';
  };

  const monthNames = ['Led', 'Úno', 'Bře', 'Dub', 'Kvě', 'Čvn', 'Čvc', 'Srp', 'Zář', 'Říj', 'Lis', 'Pro'];
  const dayNames = ['Po', 'Út', 'St', 'Čt', 'Pá', 'So', 'Ne'];

  return (
    <div className="bg-white dark:bg-gray-800 shadow-sm ring-1 ring-gray-900/5 dark:ring-gray-700 sm:rounded-xl p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
          {t('title')}
        </h3>
        
        {/* Year selector */}
        <div className="flex items-center gap-3">
          <label className="text-sm text-gray-600 dark:text-gray-400">
            {t('year')}:
          </label>
          <select
            value={selectedYear}
            onChange={(e) => setSelectedYear(Number(e.target.value))}
            className="px-4 py-2 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg text-sm font-medium text-gray-900 dark:text-white hover:bg-gray-50 dark:hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-techmania-blue"
          >
            {availableYears.map(year => (
              <option key={year} value={year}>
                {year}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6 overflow-x-auto">
        {Object.entries(monthsData).map(([monthKey, monthData]) => {
          const [yearStr, monthStr] = monthKey.split('-');
          const monthIndex = parseInt(monthStr) - 1;
          
          // Vytvoříme 2D pole pro týdny a dny
          const firstDay = new Date(parseInt(yearStr), monthIndex, 1);
          const lastDay = new Date(parseInt(yearStr), monthIndex + 1, 0);
          const startDayOfWeek = (firstDay.getDay() + 6) % 7; // Pondělí = 0
          
          const weeks: (CalendarHeatmapData | null)[][] = [[]];
          let weekIndex = 0;
          
          // Přidáme prázdné buňky na začátku
          for (let i = 0; i < startDayOfWeek; i++) {
            weeks[weekIndex].push(null);
          }
          
          // Přidáme data
          for (let day = 1; day <= lastDay.getDate(); day++) {
            const dateStr = `${yearStr}-${monthStr}-${String(day).padStart(2, '0')}`;
            const dayData = monthData.find(d => d.date === dateStr) || null;
            
            weeks[weekIndex].push(dayData);
            
            if (weeks[weekIndex].length === 7 && day < lastDay.getDate()) {
              weekIndex++;
              weeks[weekIndex] = [];
            }
          }
          
          return (
            <div key={monthKey} className="min-w-max">
              <div className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                {monthNames[monthIndex]}
              </div>
              <div className="space-y-1">
                {weeks.map((week, weekIdx) => (
                  <div key={weekIdx} className="flex gap-1">
                    {week.map((day, dayIdx) => (
                      <div
                        key={dayIdx}
                        className={`w-8 h-8 rounded ${
                          day 
                            ? `${getColor(day.visitors)} border border-gray-300 dark:border-gray-600 cursor-pointer hover:ring-2 hover:ring-techmania-blue`
                            : 'bg-gray-50 dark:bg-gray-900'
                        }`}
                        title={day ? `${day.date}: ${day.visitors.toLocaleString('cs-CZ')} ${t('visitorsCount')}` : ''}
                      >
                        {day && (
                          <div className="flex items-center justify-center h-full text-xs font-medium text-gray-700 dark:text-gray-300">
                            {new Date(day.date).getDate()}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>

      {/* Legenda */}
      <div className="mt-6 flex items-center gap-2 text-xs text-gray-600 dark:text-gray-400">
        <span>{t('less')}</span>
        <div className="flex gap-1">
          <div className="w-4 h-4 bg-blue-100 dark:bg-blue-900/20 rounded"></div>
          <div className="w-4 h-4 bg-blue-200 dark:bg-blue-800/40 rounded"></div>
          <div className="w-4 h-4 bg-blue-400 dark:bg-blue-600/60 rounded"></div>
          <div className="w-4 h-4 bg-blue-600 dark:bg-blue-500/80 rounded"></div>
          <div className="w-4 h-4 bg-blue-800 dark:bg-blue-400 rounded"></div>
        </div>
        <span>{t('more')}</span>
      </div>
    </div>
  );
}
