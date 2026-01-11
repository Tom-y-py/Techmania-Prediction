'use client';

import { useState, useEffect } from 'react';
import { 
  BellIcon, 
  ExclamationTriangleIcon, 
  CheckCircleIcon,
  InformationCircleIcon,
  XMarkIcon
} from '@heroicons/react/24/outline';
import { api } from '@/lib/api';
import { useTranslations } from '@/lib/i18n';

interface Alert {
  id: string;
  type: 'warning' | 'info' | 'success' | 'error';
  title: string;
  message: string;
  date?: string;
  dismissable?: boolean;
}

export default function AlertsPanel() {
  const t = useTranslations('alerts');
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [loading, setLoading] = useState(true);
  const [dismissedAlerts, setDismissedAlerts] = useState<string[]>([]);

  useEffect(() => {
    generateAlerts();
  }, []);

  const generateAlerts = async () => {
    setLoading(true);
    const newAlerts: Alert[] = [];

    try {
      // Načíst predikce na příštích 14 dní
      const today = new Date();
      const endDate = new Date(today);
      endDate.setDate(endDate.getDate() + 14);

      const predictions = await api.predictRange({
        start_date: today.toISOString().split('T')[0],
        end_date: endDate.toISOString().split('T')[0]
      });

      // Analyzovat predikce a vytvořit alerty
      predictions.predictions.forEach((pred, index) => {
        // Vysoká návštěvnost (> 700)
        if (pred.predicted_visitors > 700) {
          newAlerts.push({
            id: `high-${pred.date}`,
            type: 'warning',
            title: t('highVisitors') || 'Vysoká návštěvnost očekávána',
            message: `${new Date(pred.date).toLocaleDateString('cs-CZ')} (${pred.day_of_week}): očekáváno ${pred.predicted_visitors} návštěvníků`,
            date: pred.date,
            dismissable: true
          });
        }

        // Svátek
        if (pred.holiday_info?.is_holiday && pred.holiday_info?.holiday_name) {
          newAlerts.push({
            id: `holiday-${pred.date}`,
            type: 'info',
            title: t('upcomingHoliday') || 'Nadcházející svátek',
            message: `${new Date(pred.date).toLocaleDateString('cs-CZ')}: ${pred.holiday_info.holiday_name}`,
            date: pred.date,
            dismissable: true
          });
        }

        // Pěkné počasí (může zvýšit návštěvnost venkovních aktivit nebo snížit indoor)
        if (pred.weather_info?.is_nice_weather && pred.weather_info?.temperature_mean > 20) {
          newAlerts.push({
            id: `weather-${pred.date}`,
            type: 'info',
            title: t('niceWeather') || 'Pěkné počasí',
            message: `${new Date(pred.date).toLocaleDateString('cs-CZ')}: ${pred.weather_info.temperature_mean.toFixed(1)}°C - možný vliv na návštěvnost`,
            date: pred.date,
            dismissable: true
          });
        }
      });

      // Načíst historii predikcí pro kontrolu přesnosti
      try {
        const history = await api.getPredictionHistory(7, false);
        if (history.summary.valid_comparisons > 0) {
          const accuracy = history.summary.accuracy_10_percent;
          if (accuracy !== null && accuracy >= 80) {
            newAlerts.push({
              id: 'accuracy-good',
              type: 'success',
              title: t('predictionAccurate') || 'Predikce přesné',
              message: `${t('last7Days') || 'Za posledních 7 dní'}: ${accuracy.toFixed(0)}% ${t('accuracyWithin10') || 'predikcí bylo v rozmezí ±10%'}`,
              dismissable: true
            });
          } else if (accuracy !== null && accuracy < 60) {
            newAlerts.push({
              id: 'accuracy-low',
              type: 'warning',
              title: t('predictionInaccurate') || 'Nižší přesnost predikcí',
              message: `${t('last7Days') || 'Za posledních 7 dní'}: pouze ${accuracy.toFixed(0)}% ${t('accuracyWithin10') || 'predikcí bylo v rozmezí ±10%'}`,
              dismissable: true
            });
          }
        }
      } catch (err) {
        console.log('Prediction history not available');
      }

      // Víkend alert
      const nextWeekend = predictions.predictions.find(p => p.is_weekend);
      if (nextWeekend) {
        const weekendPredictions = predictions.predictions.filter(p => p.is_weekend);
        const totalWeekend = weekendPredictions.reduce((sum, p) => sum + p.predicted_visitors, 0);
        newAlerts.push({
          id: 'weekend-forecast',
          type: 'info',
          title: t('weekendForecast') || 'Víkendová předpověď',
          message: `${t('expectedTotal') || 'Očekávaný celkový počet'}: ${totalWeekend.toLocaleString('cs-CZ')} ${t('visitors') || 'návštěvníků'}`,
          dismissable: true
        });
      }

    } catch (error) {
      console.error('Error generating alerts:', error);
      newAlerts.push({
        id: 'error',
        type: 'error',
        title: t('errorLoading') || 'Chyba při načítání',
        message: t('errorLoadingAlerts') || 'Nepodařilo se načíst upozornění',
        dismissable: false
      });
    }

    // Omezit na max 5 alertů a seřadit podle priority
    const priorityOrder = { error: 0, warning: 1, success: 2, info: 3 };
    const sortedAlerts = newAlerts
      .sort((a, b) => priorityOrder[a.type] - priorityOrder[b.type])
      .slice(0, 5);

    setAlerts(sortedAlerts);
    setLoading(false);
  };

  const dismissAlert = (id: string) => {
    setDismissedAlerts(prev => [...prev, id]);
  };

  const getIcon = (type: string) => {
    switch (type) {
      case 'warning':
        return <ExclamationTriangleIcon className="h-5 w-5 text-amber-500" />;
      case 'success':
        return <CheckCircleIcon className="h-5 w-5 text-green-500" />;
      case 'error':
        return <ExclamationTriangleIcon className="h-5 w-5 text-red-500" />;
      default:
        return <InformationCircleIcon className="h-5 w-5 text-blue-500" />;
    }
  };

  const getBgColor = (type: string) => {
    switch (type) {
      case 'warning':
        return 'bg-amber-50 dark:bg-amber-900/20 border-amber-200 dark:border-amber-800';
      case 'success':
        return 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800';
      case 'error':
        return 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800';
      default:
        return 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800';
    }
  };

  const visibleAlerts = alerts.filter(alert => !dismissedAlerts.includes(alert.id));

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

  if (visibleAlerts.length === 0) {
    return (
      <div className="bg-white dark:bg-gray-800 shadow-sm ring-1 ring-gray-900/5 dark:ring-gray-700 sm:rounded-xl p-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
          <BellIcon className="h-5 w-5" />
          {t('title') || 'Upozornění a Insights'}
        </h3>
        <p className="text-sm text-gray-500 dark:text-gray-400">
          {t('noAlerts') || 'Žádná nová upozornění'}
        </p>
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-gray-800 shadow-sm ring-1 ring-gray-900/5 dark:ring-gray-700 sm:rounded-xl p-6">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
        <BellIcon className="h-5 w-5" />
        {t('title') || 'Upozornění a Insights'}
        <span className="ml-auto text-xs font-normal text-gray-500 dark:text-gray-400">
          {visibleAlerts.length} {t('active') || 'aktivních'}
        </span>
      </h3>
      <div className="space-y-3">
        {visibleAlerts.map((alert) => (
          <div
            key={alert.id}
            className={`p-4 rounded-lg border ${getBgColor(alert.type)} relative`}
          >
            <div className="flex items-start gap-3">
              {getIcon(alert.type)}
              <div className="flex-1 min-w-0">
                <p className="font-medium text-gray-900 dark:text-white text-sm">
                  {alert.title}
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                  {alert.message}
                </p>
              </div>
              {alert.dismissable && (
                <button
                  onClick={() => dismissAlert(alert.id)}
                  className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                >
                  <XMarkIcon className="h-4 w-4" />
                </button>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
