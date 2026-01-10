'use client';

import Sidebar from '@/components/Sidebar';
import Header from '@/components/Header';
import { useTranslations } from '@/lib/i18n';
import { 
  ChartBarIcon, 
  CpuChipIcon, 
  BeakerIcon,
  CloudIcon,
  CalendarIcon,
  ChartPieIcon
} from '@heroicons/react/24/outline';

export default function InfoPage() {
  const t = useTranslations('info');
  
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

          {/* O projektu */}
          <div className="mb-8 bg-white dark:bg-gray-800 shadow-sm ring-1 ring-gray-900/5 dark:ring-gray-700 sm:rounded-xl">
            <div className="px-4 py-6 sm:p-8">
              <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
                {t('aboutProject')}
              </h2>
              <div className="prose max-w-none text-gray-600 dark:text-gray-400">
                <p>
                  {t('aboutDescription')}
                </p>
              </div>
            </div>
          </div>

          {/* Model Info */}
          <div className="mb-8 bg-white dark:bg-gray-800 shadow-sm ring-1 ring-gray-900/5 dark:ring-gray-700 sm:rounded-xl">
            <div className="px-4 py-6 sm:p-8">
              <div className="flex items-center mb-4">
                <CpuChipIcon className="h-6 w-6 text-techmania-blue mr-2" />
                <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                  {t('ensembleModel')}
                </h2>
              </div>
              <div className="space-y-4 text-gray-600 dark:text-gray-400">
                <p>
                  {t('ensembleDescription')}
                </p>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
                  <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                    <div className="flex items-center mb-2">
                      <ChartBarIcon className="h-5 w-5 text-blue-600 mr-2" />
                      <h3 className="font-semibold text-gray-900 dark:text-white">{t('lightgbm.title')}</h3>
                    </div>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">{t('lightgbm.weight')}</p>
                    <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                      <li>• {t('lightgbm.feature1')}</li>
                      <li>• {t('lightgbm.feature2')}</li>
                      <li>• {t('lightgbm.feature3')}</li>
                    </ul>
                  </div>
                  
                  <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                    <div className="flex items-center mb-2">
                      <CalendarIcon className="h-5 w-5 text-green-600 mr-2" />
                      <h3 className="font-semibold text-gray-900 dark:text-white">{t('prophet.title')}</h3>
                    </div>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">{t('prophet.weight')}</p>
                    <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                      <li>• {t('prophet.feature1')}</li>
                      <li>• {t('prophet.feature2')}</li>
                      <li>• {t('prophet.feature3')}</li>
                    </ul>
                  </div>
                  
                  <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                    <div className="flex items-center mb-2">
                      <BeakerIcon className="h-5 w-5 text-purple-600 mr-2" />
                      <h3 className="font-semibold text-gray-900 dark:text-white">{t('neuralNetwork.title')}</h3>
                    </div>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">{t('neuralNetwork.weight')}</p>
                    <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                      <li>• {t('neuralNetwork.feature1')}</li>
                      <li>• {t('neuralNetwork.feature2')}</li>
                      <li>• {t('neuralNetwork.feature3')}</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Features */}
          <div className="mb-8 bg-white dark:bg-gray-800 shadow-sm ring-1 ring-gray-900/5 dark:ring-gray-700 sm:rounded-xl">
            <div className="px-4 py-6 sm:p-8">
              <div className="flex items-center mb-4">
                <ChartPieIcon className="h-6 w-6 text-techmania-blue mr-2" />
                <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                  {t('inputFactors')}
                </h2>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h3 className="font-semibold text-gray-900 dark:text-white mb-3 flex items-center">
                    <CalendarIcon className="h-5 w-5 text-blue-600 mr-2" />
                    {t('timeFactors')}
                  </h3>
                  <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
                    <li>• {t('timeFactor1')}</li>
                    <li>• {t('timeFactor2')}</li>
                    <li>• {t('timeFactor3')}</li>
                    <li>• {t('timeFactor4')}</li>
                    <li>• {t('timeFactor5')}</li>
                  </ul>
                </div>
                <div>
                  <h3 className="font-semibold text-gray-900 dark:text-white mb-3 flex items-center">
                    <CloudIcon className="h-5 w-5 text-blue-600 mr-2" />
                    {t('weather')}
                  </h3>
                  <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
                    <li>• {t('weatherFactor1')}</li>
                    <li>• {t('weatherFactor2')}</li>
                    <li>• {t('weatherFactor3')}</li>
                    <li>• {t('weatherFactor4')}</li>
                    <li>• {t('weatherFactor5')}</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>

          {/* Technologie */}
          <div className="mb-8 bg-white dark:bg-gray-800 shadow-sm ring-1 ring-gray-900/5 dark:ring-gray-700 sm:rounded-xl">
            <div className="px-4 py-6 sm:p-8">
              <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
                {t('technologies')}
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h3 className="font-semibold text-gray-900 dark:text-white mb-3">{t('backend')}</h3>
                  <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
                    <li>• <strong>Python 3.x</strong> - hlavní jazyk</li>
                    <li>• <strong>Flask</strong> - REST API framework</li>
                    <li>• <strong>scikit-learn</strong> - ML nástroje</li>
                    <li>• <strong>LightGBM</strong> - gradient boosting</li>
                    <li>• <strong>Prophet</strong> - časové řady (Meta)</li>
                    <li>• <strong>TensorFlow/Keras</strong> - neuronové sítě</li>
                    <li>• <strong>pandas & numpy</strong> - data processing</li>
                  </ul>
                </div>
                <div>
                  <h3 className="font-semibold text-gray-900 dark:text-white mb-3">{t('frontend')}</h3>
                  <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
                    <li>• <strong>Next.js 14</strong> - React framework</li>
                    <li>• <strong>TypeScript</strong> - type safety</li>
                    <li>• <strong>Tailwind CSS</strong> - styling</li>
                    <li>• <strong>Recharts</strong> - vizualizace dat</li>
                    <li>• <strong>Headless UI</strong> - komponenty</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>

          {/* Metriky přesnosti */}
          <div className="bg-white dark:bg-gray-800 shadow-sm ring-1 ring-gray-900/5 dark:ring-gray-700 sm:rounded-xl">
            <div className="px-4 py-6 sm:p-8">
              <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
                {t('metrics')}
              </h2>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
                  <div className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-1">RMSE</div>
                  <div className="text-2xl font-bold text-blue-600">~150</div>
                  <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">Root Mean Squared Error</div>
                </div>
                <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
                  <div className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-1">MAE</div>
                  <div className="text-2xl font-bold text-green-600">~100</div>
                  <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">Mean Absolute Error</div>
                </div>
                <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
                  <div className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-1">R²</div>
                  <div className="text-2xl font-bold text-purple-600">~0.85</div>
                  <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">Koeficient determinace</div>
                </div>
              </div>
              <p className="mt-4 text-sm text-gray-600 dark:text-gray-400">
                {t('metricsDescription')}
              </p>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
