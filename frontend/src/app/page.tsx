'use client';

import { useState, useEffect } from 'react';
import Sidebar from '@/components/Sidebar';
import Header from '@/components/Header';
import StatsCards from '@/components/StatsCards';
import VisitorChart from '@/components/VisitorChart';
import RangePredictionForm from '@/components/RangePredictionForm';
import { useTranslations } from '@/lib/i18n';

export default function Home() {
  const t = useTranslations('dashboard');
  const [refreshKey, setRefreshKey] = useState(0);
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

  // Automatická aktualizace dat
  useEffect(() => {
    if (!settings.autoRefresh) return;

    const interval = setInterval(() => {
      setRefreshKey(prev => prev + 1);
      console.log('Automatická aktualizace dat...');
    }, parseInt(settings.refreshInterval) * 1000);

    return () => clearInterval(interval);
  }, [settings.autoRefresh, settings.refreshInterval]);

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
            <StatsCards key={`stats-${refreshKey}`} />
          </div>

          {/* Chart */}
          <div id="analytics" className="mb-8">
            <VisitorChart key={`chart-${refreshKey}`} />
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
