'use client';

import { Fragment, useState, useEffect } from 'react';
import { Disclosure, Menu, Transition, Popover } from '@headlessui/react';
import { BellIcon, ExclamationTriangleIcon, CalendarIcon, SunIcon, CheckCircleIcon } from '@heroicons/react/24/outline';
import HealthStatus from './HealthStatus';
import { useTranslations } from '@/lib/i18n';
import { api } from '@/lib/api';

interface Notification {
  id: string;
  type: 'warning' | 'info' | 'success';
  title: string;
  message: string;
  time: string;
}

export default function Header() {
  const t = useTranslations('header');
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [unreadCount, setUnreadCount] = useState(0);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    loadNotifications();
  }, []);

  const loadNotifications = async () => {
    try {
      const today = new Date();
      const endDate = new Date(today);
      endDate.setDate(endDate.getDate() + 7);

      const predictions = await api.predictRange({
        start_date: today.toISOString().split('T')[0],
        end_date: endDate.toISOString().split('T')[0]
      });

      const newNotifications: Notification[] = [];

      predictions.predictions.forEach((pred) => {
        // Vysoká návštěvnost
        if (pred.predicted_visitors > 700) {
          newNotifications.push({
            id: `high-${pred.date}`,
            type: 'warning',
            title: t('highAttendance') || 'Vysoká návštěvnost',
            message: `${new Date(pred.date).toLocaleDateString('cs-CZ')}: ${pred.predicted_visitors} návštěvníků`,
            time: pred.day_of_week
          });
        }

        // Svátek
        if (pred.holiday_info?.is_holiday && pred.holiday_info?.holiday_name) {
          newNotifications.push({
            id: `holiday-${pred.date}`,
            type: 'info',
            title: pred.holiday_info.holiday_name,
            message: new Date(pred.date).toLocaleDateString('cs-CZ'),
            time: pred.day_of_week
          });
        }

        // Pěkné počasí
        if (pred.weather_info?.is_nice_weather && pred.weather_info?.temperature_mean > 18) {
          newNotifications.push({
            id: `weather-${pred.date}`,
            type: 'success',
            title: t('niceWeather') || 'Pěkné počasí',
            message: `${new Date(pred.date).toLocaleDateString('cs-CZ')}: ${pred.weather_info.temperature_mean.toFixed(0)}°C`,
            time: pred.day_of_week
          });
        }
      });

      // Omezit na 5 notifikací
      setNotifications(newNotifications.slice(0, 5));
      setUnreadCount(Math.min(newNotifications.length, 5));
    } catch (error) {
      console.error('Error loading notifications:', error);
    }
    setLoading(false);
  };

  const getIcon = (type: string) => {
    switch (type) {
      case 'warning':
        return <ExclamationTriangleIcon className="h-5 w-5 text-amber-500" />;
      case 'success':
        return <SunIcon className="h-5 w-5 text-green-500" />;
      default:
        return <CalendarIcon className="h-5 w-5 text-blue-500" />;
    }
  };

  const markAllAsRead = () => {
    setUnreadCount(0);
  };
  
  return (
    <div className="sticky top-0 z-40 flex h-16 shrink-0 items-center gap-x-4 border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 px-4 shadow-sm sm:gap-x-6 sm:px-6 lg:px-8">
      <div className="flex flex-1 gap-x-4 self-stretch lg:gap-x-6">
        <div className="relative flex flex-1 items-center">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
            {t('title')}
          </h2>
        </div>
        <div className="flex items-center gap-x-4 lg:gap-x-6">
          <HealthStatus />
          
          {/* Notifications Dropdown */}
          <Popover className="relative">
            <Popover.Button
              className="-m-2.5 p-2.5 text-gray-400 hover:text-gray-500 dark:text-gray-300 dark:hover:text-gray-200 relative"
            >
              <span className="sr-only">{t('notifications')}</span>
              <BellIcon className="h-6 w-6" aria-hidden="true" />
              {unreadCount > 0 && (
                <span className="absolute top-0 right-0 flex h-4 w-4 items-center justify-center rounded-full bg-red-500 text-[10px] font-bold text-white">
                  {unreadCount}
                </span>
              )}
            </Popover.Button>

            <Transition
              as={Fragment}
              enter="transition ease-out duration-200"
              enterFrom="opacity-0 translate-y-1"
              enterTo="opacity-100 translate-y-0"
              leave="transition ease-in duration-150"
              leaveFrom="opacity-100 translate-y-0"
              leaveTo="opacity-0 translate-y-1"
            >
              <Popover.Panel className="absolute right-0 z-10 mt-2 w-80 origin-top-right rounded-xl bg-white dark:bg-gray-800 shadow-lg ring-1 ring-gray-900/5 dark:ring-gray-700 focus:outline-none">
                <div className="p-4">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-sm font-semibold text-gray-900 dark:text-white">
                      {t('notificationsTitle') || 'Upozornění'}
                    </h3>
                    {unreadCount > 0 && (
                      <button 
                        onClick={markAllAsRead}
                        className="text-xs text-blue-600 hover:text-blue-800 dark:text-blue-400"
                      >
                        {t('markAllRead') || 'Označit jako přečtené'}
                      </button>
                    )}
                  </div>
                  
                  {loading ? (
                    <div className="animate-pulse space-y-3">
                      <div className="h-12 bg-gray-100 dark:bg-gray-700 rounded"></div>
                      <div className="h-12 bg-gray-100 dark:bg-gray-700 rounded"></div>
                    </div>
                  ) : notifications.length > 0 ? (
                    <div className="space-y-2">
                      {notifications.map((notif) => (
                        <div 
                          key={notif.id}
                          className="flex items-start gap-3 p-2 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer"
                        >
                          {getIcon(notif.type)}
                          <div className="flex-1 min-w-0">
                            <p className="text-sm font-medium text-gray-900 dark:text-white truncate">
                              {notif.title}
                            </p>
                            <p className="text-xs text-gray-500 dark:text-gray-400">
                              {notif.message}
                            </p>
                          </div>
                          <span className="text-xs text-gray-400 dark:text-gray-500 whitespace-nowrap">
                            {notif.time}
                          </span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-sm text-gray-500 dark:text-gray-400 text-center py-4">
                      {t('noNotifications') || 'Žádná upozornění'}
                    </p>
                  )}
                  
                  <div className="mt-3 pt-3 border-t border-gray-100 dark:border-gray-700">
                    <a 
                      href="/calendar" 
                      className="block text-center text-sm text-blue-600 hover:text-blue-800 dark:text-blue-400 font-medium"
                    >
                      {t('viewCalendar') || 'Zobrazit kalendář'} →
                    </a>
                  </div>
                </div>
              </Popover.Panel>
            </Transition>
          </Popover>

          <div
            className="hidden lg:block lg:h-6 lg:w-px lg:bg-gray-200 dark:bg-gray-700"
            aria-hidden="true"
          />

          <Menu as="div" className="relative">
            <Menu.Button className="-m-1.5 flex items-center p-1.5">
              <span className="sr-only">{t('userMenu')}</span>
              <div className="h-8 w-8 rounded-full bg-techmania-blue flex items-center justify-center">
                <span className="text-sm font-medium text-white">TC</span>
              </div>
              <span className="hidden lg:flex lg:items-center">
                <span
                  className="ml-4 text-sm font-semibold leading-6 text-gray-900 dark:text-white"
                  aria-hidden="true"
                >
                  Techmania
                </span>
              </span>
            </Menu.Button>
            <Transition
              as={Fragment}
              enter="transition ease-out duration-100"
              enterFrom="transform opacity-0 scale-95"
              enterTo="transform opacity-100 scale-100"
              leave="transition ease-in duration-75"
              leaveFrom="transform opacity-100 scale-100"
              leaveTo="transform opacity-0 scale-95"
            >
              <Menu.Items className="absolute right-0 z-10 mt-2.5 w-32 origin-top-right rounded-md bg-white py-2 shadow-lg ring-1 ring-gray-900/5 focus:outline-none">
                <Menu.Item>
                  {({ active }) => (
                    <a
                      href="#"
                      className={`${
                        active ? 'bg-gray-50' : ''
                      } block px-3 py-1 text-sm leading-6 text-gray-900`}
                    >
                      {t('profile')}
                    </a>
                  )}
                </Menu.Item>
                <Menu.Item>
                  {({ active }) => (
                    <a
                      href="#"
                      className={`${
                        active ? 'bg-gray-50' : ''
                      } block px-3 py-1 text-sm leading-6 text-gray-900`}
                    >
                      {t('logout')}
                    </a>
                  )}
                </Menu.Item>
              </Menu.Items>
            </Transition>
          </Menu>
        </div>
      </div>
    </div>
  );
}
