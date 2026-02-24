'use client';

import { useState, useEffect } from 'react';
import Sidebar from '@/components/Sidebar';
import Header from '@/components/Header';
import StatsCards from '@/components/StatsCards';
import VisitorChart from '@/components/VisitorChart';
import RangePredictionForm from '@/components/RangePredictionForm';
import AlertsPanel from '@/components/AlertsPanel';
import EventsCalendar from '@/components/EventsCalendar';
import { useTranslations } from '@/lib/i18n';

export default function Home() {
  const t = useTranslations('dashboard');
  // ODSTRANĚNO: refreshKey způsoboval remountování všech komponent každých 30s
  // což vedlo k burst desítek API požadavků najednou -> 503 rate limit errors
  // Každá komponenta má vlastní useEffect a auto-refresh hook
  const [settings, setSettings] = useState({
    autoRefresh: true,
    refreshInterval: '30'
  });

  // Načíst nastavení z localStorage
  useEffect(() => {
    const savedSettings = localStorage.getItem('techmania-settings');
    if (savedSettings) {
      try {
        const parsed = JSON.parse(savedSettings);
        setSettings({
          autoRefresh: parsed.autoRefresh ?? true,
          refreshInterval: parsed.refreshInterval || '30'
        });
      } catch (error) {
        console.error('Chyba při načítání nastavení:', error);
      }
    }

    // Naslouchat změnám v nastavení
    const handleStorageChange = () => {
      const savedSettings = localStorage.getItem('techmania-settings');
      if (savedSettings) {
        try {
          const parsed = JSON.parse(savedSettings);
          setSettings({
            autoRefresh: parsed.autoRefresh ?? true,
            refreshInterval: parsed.refreshInterval || '30'
          });
        } catch (error) {
          console.error('Chyba při načítání nastavení:', error);
        }
      }
    };

    window.addEventListener('storage', handleStorageChange);
    return () => window.removeEventListener('storage', handleStorageChange);
  }, []);

  // Automatická aktualizace dat - VYPNUTO
  // Každá komponenta má vlastní useEffect a refresh logiku (např. usePredictionUpdates hook)
  // Globální refresh způsoboval remountování všech komponent a burst API calls

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <Sidebar />
      
      <main className="lg:pl-72">
        <Header />
        <div className="px-4 py-10 sm:px-6 lg:px-8">
          {/* Header */}
          <div className="mb-8">
            <h1 className="text-3xl font-bold tracking-tight text-gray-900 dark:text-white">
              {t('title')}
            </h1>
            <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">
              {t('subtitle')}
            </p>
          </div>

          {/* Stats Cards */}
          <div className="mb-8">
            <StatsCards />
          </div>

          {/* Chart */}
          <div id="analytics" className="mb-8">
            <VisitorChart />
          </div>

          {/* Alerts & Calendar Row */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <AlertsPanel />
            <EventsCalendar />
          </div>

          {/* Predictions Section */}
          <div id="predictions" className="space-y-8">
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
                {t('predictionsTitle')}
              </h2>
              <RangePredictionForm />
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
