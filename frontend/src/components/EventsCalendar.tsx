'use client';

import { useState, useEffect } from 'react';
import { CalendarIcon, ChevronLeftIcon, ChevronRightIcon } from '@heroicons/react/24/outline';
import { api } from '@/lib/api';
import { useTranslations } from '@/lib/i18n';

interface Event {
  date: string;
  name: string;
  type: 'holiday' | 'vacation' | 'event' | 'weekend';
}

interface DayInfo {
  date: Date;
  day: number;
  isCurrentMonth: boolean;
  isToday: boolean;
  isWeekend: boolean;
  events: Event[];
}

export default function EventsCalendar() {
  const t = useTranslations('calendar');
  const [currentDate, setCurrentDate] = useState(new Date());
  const [events, setEvents] = useState<Event[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedDay, setSelectedDay] = useState<DayInfo | null>(null);

  useEffect(() => {
    loadData();
  }, [currentDate]);

  const loadData = async () => {
    setLoading(true);
    const year = currentDate.getFullYear();
    const month = currentDate.getMonth() + 1; // API očekává 1-12

    try {
      // Načíst události z API
      const calendarData = await api.getCalendarEvents(month, year);
      
      // Zpracovat události z API
      const newEvents: Event[] = calendarData.events.map(e => ({
        date: e.date,
        name: e.name,
        type: e.type as 'holiday' | 'vacation' | 'event' | 'weekend'
      }));

      setEvents(newEvents);
    } catch (error) {
      console.error('Error loading calendar data:', error);
      setEvents([]);
    }

    setLoading(false);
  };

  const getEventTypeColor = (type: string) => {
    switch (type) {
      case 'holiday':
        return 'bg-red-500';
      case 'vacation':
        return 'bg-orange-500';
      case 'event':
        return 'bg-purple-500';
      default:
        return 'bg-blue-500';
    }
  };

  const getDaysInMonth = (): DayInfo[] => {
    const year = currentDate.getFullYear();
    const month = currentDate.getMonth();
    const today = new Date();
    today.setHours(0, 0, 0, 0);

    const firstDay = new Date(year, month, 1);
    const lastDay = new Date(year, month + 1, 0);
    
    // Začít od pondělí (1 = pondělí, 0 = neděle)
    let startOffset = firstDay.getDay() - 1;
    if (startOffset < 0) startOffset = 6;

    const days: DayInfo[] = [];

    // Dny z předchozího měsíce
    const prevMonthLastDay = new Date(year, month, 0).getDate();
    for (let i = startOffset - 1; i >= 0; i--) {
      const date = new Date(year, month - 1, prevMonthLastDay - i);
      const dateStr = date.toISOString().split('T')[0];
      days.push({
        date,
        day: prevMonthLastDay - i,
        isCurrentMonth: false,
        isToday: date.getTime() === today.getTime(),
        isWeekend: date.getDay() === 0 || date.getDay() === 6,
        events: events.filter(e => e.date === dateStr)
      });
    }

    // Dny aktuálního měsíce
    for (let day = 1; day <= lastDay.getDate(); day++) {
      const date = new Date(year, month, day);
      const dateStr = date.toISOString().split('T')[0];
      days.push({
        date,
        day,
        isCurrentMonth: true,
        isToday: date.getTime() === today.getTime(),
        isWeekend: date.getDay() === 0 || date.getDay() === 6,
        events: events.filter(e => e.date === dateStr)
      });
    }

    // Dny z následujícího měsíce
    const remainingDays = 42 - days.length; // 6 řádků × 7 dní
    for (let i = 1; i <= remainingDays; i++) {
      const date = new Date(year, month + 1, i);
      const dateStr = date.toISOString().split('T')[0];
      days.push({
        date,
        day: i,
        isCurrentMonth: false,
        isToday: date.getTime() === today.getTime(),
        isWeekend: date.getDay() === 0 || date.getDay() === 6,
        events: events.filter(e => e.date === dateStr)
      });
    }

    return days;
  };

  const prevMonth = () => {
    setCurrentDate(new Date(currentDate.getFullYear(), currentDate.getMonth() - 1, 1));
    setSelectedDay(null);
  };

  const nextMonth = () => {
    setCurrentDate(new Date(currentDate.getFullYear(), currentDate.getMonth() + 1, 1));
    setSelectedDay(null);
  };

  const goToToday = () => {
    setCurrentDate(new Date());
    setSelectedDay(null);
  };

  const monthNames = [
    'Leden', 'Únor', 'Březen', 'Duben', 'Květen', 'Červen',
    'Červenec', 'Srpen', 'Září', 'Říjen', 'Listopad', 'Prosinec'
  ];

  const dayNames = ['Po', 'Út', 'St', 'Čt', 'Pá', 'So', 'Ne'];

  const days = getDaysInMonth();

  return (
    <div className="bg-white dark:bg-gray-800 shadow-sm ring-1 ring-gray-900/5 dark:ring-gray-700 sm:rounded-xl p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center gap-2">
          <CalendarIcon className="h-5 w-5" />
          {t('title') || 'Kalendář událostí'}
        </h3>
        <div className="flex items-center gap-2">
          <button
            onClick={goToToday}
            className="px-2 py-1 text-xs font-medium text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white"
          >
            {t('today') || 'Dnes'}
          </button>
          <button
            onClick={prevMonth}
            className="p-1 rounded-md hover:bg-gray-100 dark:hover:bg-gray-700"
          >
            <ChevronLeftIcon className="h-5 w-5 text-gray-600 dark:text-gray-400" />
          </button>
          <span className="text-sm font-medium text-gray-900 dark:text-white min-w-[120px] text-center">
            {monthNames[currentDate.getMonth()]} {currentDate.getFullYear()}
          </span>
          <button
            onClick={nextMonth}
            className="p-1 rounded-md hover:bg-gray-100 dark:hover:bg-gray-700"
          >
            <ChevronRightIcon className="h-5 w-5 text-gray-600 dark:text-gray-400" />
          </button>
        </div>
      </div>

      {/* Legenda */}
      <div className="flex flex-wrap gap-3 mb-4 text-xs">
        <div className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-red-500"></span>
          <span className="text-gray-600 dark:text-gray-400">{t('holiday') || 'Svátek'}</span>
        </div>
        <div className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-orange-500"></span>
          <span className="text-gray-600 dark:text-gray-400">{t('vacation') || 'Prázdniny'}</span>
        </div>
        <div className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-purple-500"></span>
          <span className="text-gray-600 dark:text-gray-400">{t('event') || 'Akce'}</span>
        </div>
      </div>

      {loading ? (
        <div className="animate-pulse grid grid-cols-7 gap-1">
          {Array.from({ length: 42 }).map((_, i) => (
            <div key={i} className="h-12 bg-gray-100 dark:bg-gray-700 rounded"></div>
          ))}
        </div>
      ) : (
        <>
          {/* Hlavička dnů */}
          <div className="grid grid-cols-7 gap-1 mb-1">
            {dayNames.map((day, i) => (
              <div
                key={day}
                className={`text-xs font-medium text-center py-2 ${
                  i >= 5 ? 'text-red-500 dark:text-red-400' : 'text-gray-500 dark:text-gray-400'
                }`}
              >
                {day}
              </div>
            ))}
          </div>

          {/* Dny */}
          <div className="grid grid-cols-7 gap-1">
            {days.map((dayInfo, i) => (
              <button
                key={i}
                onClick={() => setSelectedDay(dayInfo)}
                className={`
                  relative h-12 p-1 rounded-md text-sm transition-colors
                  ${dayInfo.isCurrentMonth 
                    ? 'text-gray-900 dark:text-white' 
                    : 'text-gray-400 dark:text-gray-600'
                  }
                  ${dayInfo.isToday 
                    ? 'ring-2 ring-blue-500' 
                    : ''
                  }
                  ${dayInfo.isWeekend && dayInfo.isCurrentMonth
                    ? 'text-red-600 dark:text-red-400' 
                    : ''
                  }
                  ${dayInfo.events.length > 0 && dayInfo.isCurrentMonth
                    ? 'bg-blue-50 dark:bg-blue-900/20'
                    : ''
                  }
                  hover:bg-blue-50 dark:hover:bg-blue-900/20
                `}
              >
                <span className="font-medium">{dayInfo.day}</span>
                
                {/* Event indikátory */}
                {dayInfo.events.length > 0 && (
                  <div className="absolute bottom-1 left-1/2 transform -translate-x-1/2 flex gap-0.5">
                    {dayInfo.events.slice(0, 3).map((event, j) => (
                      <span
                        key={j}
                        className={`w-1.5 h-1.5 rounded-full ${getEventTypeColor(event.type)}`}
                      />
                    ))}
                  </div>
                )}
              </button>
            ))}
          </div>
        </>
      )}

      {/* Detail vybraného dne */}
      {selectedDay && selectedDay.isCurrentMonth && (
        <div className="mt-4 p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
          <div className="flex items-center justify-between mb-2">
            <h4 className="font-medium text-gray-900 dark:text-white">
              {selectedDay.date.toLocaleDateString('cs-CZ', { 
                weekday: 'long', 
                day: 'numeric', 
                month: 'long' 
              })}
            </h4>
            <button 
              onClick={() => setSelectedDay(null)}
              className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
            >
              ×
            </button>
          </div>

          {selectedDay.events.length > 0 ? (
            <div className="space-y-2">
              {selectedDay.events.map((event, i) => (
                <div key={i} className="flex items-center gap-2">
                  <span className={`w-2 h-2 rounded-full ${getEventTypeColor(event.type)}`}></span>
                  <span className="text-sm text-gray-700 dark:text-gray-300">{event.name}</span>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-gray-500 dark:text-gray-400">
              {t('noEvents') || 'Žádné události'}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
