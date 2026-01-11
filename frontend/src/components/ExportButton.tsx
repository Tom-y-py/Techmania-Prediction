'use client';

import { useState } from 'react';
import { Menu } from '@headlessui/react';
import { 
  ArrowDownTrayIcon, 
  ChevronDownIcon 
} from '@heroicons/react/24/outline';
import { useTranslations } from '@/lib/i18n';

interface ExportData {
  date: string;
  predicted_visitors: number;
}

interface ExportButtonProps {
  data: ExportData[];
  filename?: string;
}

export default function ExportButton({ 
  data, 
  filename = 'techmania_predikce' 
}: ExportButtonProps) {
  const t = useTranslations('export');
  const [exporting, setExporting] = useState(false);

  const exportToCSV = () => {
    setExporting(true);
    try {
      const headers = [t('date'), t('predictedVisitors')];
      const csvContent = [
        headers.join(','),
        ...data.map(row => [row.date, row.predicted_visitors].join(','))
      ].join('\n');

      const blob = new Blob(['\ufeff' + csvContent], { 
        type: 'text/csv;charset=utf-8;' 
      });
      const link = document.createElement('a');
      const url = URL.createObjectURL(blob);
      
      link.setAttribute('href', url);
      link.setAttribute('download', `${filename}.csv`);
      link.style.visibility = 'hidden';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (error) {
      console.error('Export failed:', error);
    } finally {
      setExporting(false);
    }
  };

  const exportToJSON = () => {
    setExporting(true);
    try {
      const jsonContent = JSON.stringify(data, null, 2);
      const blob = new Blob([jsonContent], { type: 'application/json' });
      const link = document.createElement('a');
      const url = URL.createObjectURL(blob);
      
      link.setAttribute('href', url);
      link.setAttribute('download', `${filename}.json`);
      link.style.visibility = 'hidden';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (error) {
      console.error('Export failed:', error);
    } finally {
      setExporting(false);
    }
  };

  if (!data || data.length === 0) {
    return null;
  }

  return (
    <Menu as="div" className="relative inline-block text-left">
      <Menu.Button className="inline-flex items-center gap-x-1.5 rounded-md bg-white dark:bg-gray-700 px-3 py-2 text-sm font-semibold text-gray-900 dark:text-gray-100 shadow-sm ring-1 ring-inset ring-gray-300 dark:ring-gray-600 hover:bg-gray-50 dark:hover:bg-gray-600 disabled:opacity-50">
        <ArrowDownTrayIcon className="-ml-0.5 h-5 w-5" aria-hidden="true" />
        {exporting ? t('exporting') : t('button')}
        <ChevronDownIcon className="-mr-1 h-5 w-5" aria-hidden="true" />
      </Menu.Button>

      <Menu.Items className="absolute right-0 z-10 mt-2 w-36 origin-top-right rounded-md bg-white dark:bg-gray-800 shadow-lg ring-1 ring-black dark:ring-gray-700 ring-opacity-5 focus:outline-none">
        <div className="py-1">
          <Menu.Item>
            {({ active }) => (
              <button
                onClick={exportToCSV}
                className={`${
                  active ? 'bg-gray-100 dark:bg-gray-700 text-gray-900 dark:text-gray-100' : 'text-gray-700 dark:text-gray-300'
                } block w-full px-4 py-2 text-left text-sm`}
              >
                {t('csv')}
              </button>
            )}
          </Menu.Item>
          <Menu.Item>
            {({ active }) => (
              <button
                onClick={exportToJSON}
                className={`${
                  active ? 'bg-gray-100 dark:bg-gray-700 text-gray-900 dark:text-gray-100' : 'text-gray-700 dark:text-gray-300'
                } block w-full px-4 py-2 text-left text-sm`}
              >
                {t('json')}
              </button>
            )}
          </Menu.Item>
        </div>
      </Menu.Items>
    </Menu>
  );
}
