'use client';

import Sidebar from '@/components/Sidebar';
import Header from '@/components/Header';
import StatsCards from '@/components/StatsCards';
import VisitorChart from '@/components/VisitorChart';
import RangePredictionForm from '@/components/RangePredictionForm';
import { API_URL, ENVIRONMENT } from '@/lib/api';

export default function Home() {
  return (
    <div className="min-h-screen bg-gray-50">
      <Sidebar />
      
      <main className="lg:pl-72">
        <Header />
        <div className="px-4 py-10 sm:px-6 lg:px-8">
          {/* Header */}
          <div className="mb-8">
            <h1 className="text-3xl font-bold tracking-tight text-gray-900">
              Dashboard Návštěvnosti
            </h1>
            <p className="mt-2 text-sm text-gray-600">
              Analýza a predikce návštěvnosti science centra Techmania
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

          {/* Predictions Section */}
          <div id="predictions" className="space-y-8">
            <div>
              <h2 className="text-2xl font-bold text-gray-900 mb-6">
                Predikce návštěvnosti
              </h2>
              <RangePredictionForm />
            </div>
          </div>

          {/* Settings Section */}
          <div id="settings" className="mt-12">
            <div className="bg-white shadow-sm ring-1 ring-gray-900/5 sm:rounded-xl">
              <div className="px-4 py-6 sm:p-8">
                <h2 className="text-base font-semibold leading-7 text-gray-900">
                  Nastavení aplikace
                </h2>
                <p className="mt-1 text-sm leading-6 text-gray-600 mb-6">
                  Konfigurace parametrů a připojení k API
                </p>
                
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700">
                      API Endpoint
                    </label>
                    <input
                      type="text"
                      value={API_URL}
                      disabled
                      className="mt-1 block w-full rounded-md border-gray-300 bg-gray-50 shadow-sm sm:text-sm px-3 py-2"
                    />
                    <p className="mt-1 text-xs text-gray-500">
                      Prostředí: <span className="font-semibold">{ENVIRONMENT}</span>
                    </p>
                  </div>
                  
                  <div className="rounded-md bg-blue-50 p-4">
                    <div className="flex">
                      <div className="flex-shrink-0">
                        <svg className="h-5 w-5 text-blue-400" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                        </svg>
                      </div>
                      <div className="ml-3">
                        <h3 className="text-sm font-medium text-blue-800">
                          Informace o modelu
                        </h3>
                        <div className="mt-2 text-sm text-blue-700">
                          <p>Používá se ensemble model kombinující:</p>
                          <ul className="list-disc list-inside mt-1 space-y-1">
                            <li>LightGBM</li>
                            <li>XGBoost</li>
                            <li>CatBoost</li>
                          </ul>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
