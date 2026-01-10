'use client';

import { createContext, useContext, useState, useEffect, ReactNode } from 'react';

type Locale = 'cs' | 'en';

interface LocaleContextType {
  locale: Locale;
  setLocale: (locale: Locale) => void;
}

const LocaleContext = createContext<LocaleContextType | undefined>(undefined);

export function LocaleProvider({ children }: { children: ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>('cs');

  useEffect(() => {
    // Load saved locale from localStorage only on client side
    const savedSettings = localStorage.getItem('techmania-settings');
    if (savedSettings) {
      try {
        const settings = JSON.parse(savedSettings);
        if (settings.language) {
          setLocaleState(settings.language);
        }
      } catch (error) {
        console.error('Error loading locale:', error);
      }
    }
  }, []);

  const setLocale = (newLocale: Locale) => {
    setLocaleState(newLocale);
    const savedSettings = localStorage.getItem('techmania-settings');
    let settings = { language: newLocale };
    
    if (savedSettings) {
      try {
        const existing = JSON.parse(savedSettings);
        settings = { ...existing, language: newLocale };
      } catch (error) {
        console.error('Error parsing settings:', error);
      }
    }
    
    localStorage.setItem('techmania-settings', JSON.stringify(settings));
    window.dispatchEvent(new Event('storage'));
  };

  return (
    <LocaleContext.Provider value={{ locale, setLocale }}>
      {children}
    </LocaleContext.Provider>
  );
}

export function useLocale() {
  const context = useContext(LocaleContext);
  if (context === undefined) {
    throw new Error('useLocale must be used within a LocaleProvider');
  }
  return context;
}
