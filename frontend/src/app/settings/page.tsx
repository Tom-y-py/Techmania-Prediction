'use client';

import { useState, useEffect } from 'react';
import { useTheme } from 'next-themes';
import { useLocale } from '@/components/LocaleProvider';
import Sidebar from '@/components/Sidebar';
import Header from '@/components/Header';
import { useTranslations } from '@/lib/i18n';
import { 
  MoonIcon,
  SunIcon,
  BellIcon,
  ChartBarIcon,
  GlobeAltIcon
} from '@heroicons/react/24/outline';

export default function SettingsPage() {
  const { theme, setTheme } = useTheme();
  const { locale, setLocale } = useLocale();
  const t = useTranslations('settings');
  const [notifications, setNotifications] = useState(true);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState('30');
  const [saved, setSaved] = useState(false);
  const [mounted, setMounted] = useState(false);

  // Prevent hydration mismatch
  useEffect(() => {
    setMounted(true);
  }, []);

  // Načíst nastavení z localStorage při mountu
  useEffect(() => {
    if (!mounted) return;
    
    const savedSettings = localStorage.getItem('techmania-settings');
    if (savedSettings) {
      try {
        const settings = JSON.parse(savedSettings);
        setNotifications(settings.notifications ?? true);
        setAutoRefresh(settings.autoRefresh ?? true);
        setRefreshInterval(settings.refreshInterval || '30');
      } catch (error) {
        console.error('Error loading settings:', error);
      }
    }
  }, [mounted]);

  // Synchronizovat theme s localStorage
  useEffect(() => {
    if (!mounted) return;
    
    const savedSettings = localStorage.getItem('techmania-settings');
    if (savedSettings) {
      try {
        const settings = JSON.parse(savedSettings);
        const savedTheme = settings.darkMode ? 'dark' : 'light';
        if (theme !== savedTheme) {
          setTheme(savedTheme);
        }
      } catch (error) {
        console.error('Error syncing theme:', error);
      }
    }
  }, [mounted, setTheme]);

  const handleDarkModeToggle = () => {
    const newTheme = theme === 'dark' ? 'light' : 'dark';
    setTheme(newTheme);
    
    // Okamžitě uložit změnu
    const savedSettings = localStorage.getItem('techmania-settings');
    let settings = { notifications, autoRefresh, refreshInterval, language: locale, darkMode: newTheme === 'dark' };
    
    if (savedSettings) {
      try {
        const existing = JSON.parse(savedSettings);
        settings = { ...existing, darkMode: newTheme === 'dark' };
      } catch (error) {
        console.error('Error parsing settings:', error);
      }
    }
    
    localStorage.setItem('techmania-settings', JSON.stringify(settings));
    window.dispatchEvent(new Event('storage'));
  };

  const handleSave = () => {
    const settings = {
      darkMode: theme === 'dark',
      notifications,
      autoRefresh,
      refreshInterval,
      language: locale,
    };
    localStorage.setItem('techmania-settings', JSON.stringify(settings));
    setSaved(true);
    setTimeout(() => setSaved(false), 3000);
    
    // Trigger storage event pro ostatní záložky/komponenty
    window.dispatchEvent(new Event('storage'));
  };

  if (!mounted) {
    return null;
  }

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

          {/* Notifikace o uložení */}
          {saved && (
            <div className="mb-6 rounded-md bg-green-50 dark:bg-green-900/20 p-4">
              <div className="flex">
                <div className="flex-shrink-0">
                  <svg className="h-5 w-5 text-green-400" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.857-9.809a.75.75 0 00-1.214-.882l-3.483 4.79-1.88-1.88a.75.75 0 10-1.06 1.061l2.5 2.5a.75.75 0 001.137-.089l4-5.5z" clipRule="evenodd" />
                  </svg>
                </div>
                <div className="ml-3">
                  <p className="text-sm font-medium text-green-800 dark:text-green-200">
                    {t('saved')}
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Vzhled */}
          <div className="mb-8 bg-white dark:bg-gray-800 shadow-sm ring-1 ring-gray-900/5 dark:ring-gray-700 sm:rounded-xl">
            <div className="px-4 py-6 sm:p-8">
              <div className="flex items-center mb-6">
                {theme === 'dark' ? (
                  <MoonIcon className="h-6 w-6 text-techmania-blue mr-2" />
                ) : (
                  <SunIcon className="h-6 w-6 text-techmania-blue mr-2" />
                )}
                <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                  {t('appearance')}
                </h2>
              </div>

              <div className="space-y-6">
                {/* Dark Mode Toggle */}
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-sm font-medium text-gray-900 dark:text-white">
                      {t('darkMode')}
                    </h3>
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                      {t('darkModeDescription')}
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={handleDarkModeToggle}
                    className={`${
                      theme === 'dark' ? 'bg-techmania-blue' : 'bg-gray-200'
                    } relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-techmania-blue focus:ring-offset-2`}
                  >
                    <span
                      className={`${
                        theme === 'dark' ? 'translate-x-5' : 'translate-x-0'
                      } pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out`}
                    />
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Notifikace */}
          <div className="mb-8 bg-white dark:bg-gray-800 shadow-sm ring-1 ring-gray-900/5 dark:ring-gray-700 sm:rounded-xl">
            <div className="px-4 py-6 sm:p-8">
              <div className="flex items-center mb-6">
                <BellIcon className="h-6 w-6 text-techmania-blue mr-2" />
                <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                  {t('notifications')}
                </h2>
              </div>

              <div className="space-y-6">
                {/* Povolit notifikace */}
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-sm font-medium text-gray-900 dark:text-white">
                      {t('enableNotifications')}
                    </h3>
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                      {t('notificationsDescription')}
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={() => setNotifications(!notifications)}
                    className={`${
                      notifications ? 'bg-techmania-blue' : 'bg-gray-200'
                    } relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-techmania-blue focus:ring-offset-2`}
                  >
                    <span
                      className={`${
                        notifications ? 'translate-x-5' : 'translate-x-0'
                      } pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out`}
                    />
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Data a aktualizace */}
          <div className="mb-8 bg-white dark:bg-gray-800 shadow-sm ring-1 ring-gray-900/5 dark:ring-gray-700 sm:rounded-xl">
            <div className="px-4 py-6 sm:p-8">
              <div className="flex items-center mb-6">
                <ChartBarIcon className="h-6 w-6 text-techmania-blue mr-2" />
                <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                  {t('dataAndUpdates')}
                </h2>
              </div>

              <div className="space-y-6">
                {/* Auto-refresh */}
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-sm font-medium text-gray-900 dark:text-white">
                      {t('autoRefresh')}
                    </h3>
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                      {t('autoRefreshDescription')}
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={() => setAutoRefresh(!autoRefresh)}
                    className={`${
                      autoRefresh ? 'bg-techmania-blue' : 'bg-gray-200'
                    } relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-techmania-blue focus:ring-offset-2`}
                  >
                    <span
                      className={`${
                        autoRefresh ? 'translate-x-5' : 'translate-x-0'
                      } pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out`}
                    />
                  </button>
                </div>

                {/* Refresh interval */}
                {autoRefresh && (
                  <div>
                    <label htmlFor="refresh-interval" className="block text-sm font-medium text-gray-900 dark:text-white">
                      {t('refreshInterval')}
                    </label>
                    <select
                      id="refresh-interval"
                      value={refreshInterval}
                      onChange={(e) => setRefreshInterval(e.target.value)}
                      className="mt-2 block w-full rounded-md border-0 py-1.5 pl-3 pr-10 text-gray-900 dark:text-white dark:bg-gray-700 ring-1 ring-inset ring-gray-300 dark:ring-gray-600 focus:ring-2 focus:ring-techmania-blue sm:text-sm sm:leading-6"
                    >
                      <option value="10">{t('10seconds')}</option>
                      <option value="30">{t('30seconds')}</option>
                      <option value="60">{t('1minute')}</option>
                      <option value="300">{t('5minutes')}</option>
                    </select>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Jazyk a region */}
          <div className="mb-8 bg-white dark:bg-gray-800 shadow-sm ring-1 ring-gray-900/5 dark:ring-gray-700 sm:rounded-xl">
            <div className="px-4 py-6 sm:p-8">
              <div className="flex items-center mb-6">
                <GlobeAltIcon className="h-6 w-6 text-techmania-blue mr-2" />
                <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                  {t('languageRegion')}
                </h2>
              </div>

              <div>
                <label htmlFor="language" className="block text-sm font-medium text-gray-900 dark:text-white">
                  {t('language')}
                </label>
                <select
                  id="language"
                  value={locale}
                  onChange={(e) => setLocale(e.target.value as 'cs' | 'en')}
                  className="mt-2 block w-full rounded-md border-0 py-1.5 pl-3 pr-10 text-gray-900 dark:text-white dark:bg-gray-700 ring-1 ring-inset ring-gray-300 dark:ring-gray-600 focus:ring-2 focus:ring-techmania-blue sm:text-sm sm:leading-6"
                >
                  <option value="cs">{t('czech')}</option>
                  <option value="en">{t('english')}</option>
                </select>
              </div>
            </div>
          </div>

          {/* Tlačítko Uložit */}
          <div className="flex justify-end">
            <button
              onClick={handleSave}
              className="rounded-md bg-techmania-blue px-6 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-blue-700 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-techmania-blue"
            >
              {t('saveSettings')}
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}
