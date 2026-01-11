'use client';

import { useLocale } from '@/components/LocaleProvider';
import cs from '../../locale/cs.json';
import en from '../../locale/en.json';

const messages = { cs, en };

export function useTranslations(namespace?: string) {
  const { locale } = useLocale();
  
  return (key: string) => {
    const keys = namespace ? `${namespace}.${key}` : key;
    const parts = keys.split('.');
    
    let value: any = messages[locale];
    for (const part of parts) {
      value = value?.[part];
    }
    
    return value || key;
  };
}
