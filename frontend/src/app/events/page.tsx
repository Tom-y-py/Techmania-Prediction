/**
 * Events Page - Správa událostí v Plzni
 */

'use client';

import { useState } from 'react';
import EventsManagement from '@/components/EventsManagement';
import { api } from '@/lib/api';

export default function EventsPage() {
  const [activeTab, setActiveTab] = useState<'list' | 'scraper'>('list');
  const [scraperLoading, setScraperLoading] = useState(false);
  const [scraperResult, setScraperResult] = useState<any>(null);
  const [scraperError, setScraperError] = useState<string | null>(null);

  // Scraper form state
  const [startDate, setStartDate] = useState(
    new Date().toISOString().split('T')[0]
  );
  const [endDate, setEndDate] = useState(
    new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0]
  );
  const [sources, setSources] = useState<string[]>(['goout', 'plzen.eu']);

  const handleRunScraper = async () => {
    setScraperLoading(true);
    setScraperError(null);
    setScraperResult(null);

    try {
      const result = await api.runEventScraper({
        start_date: startDate,
        end_date: endDate,
        sources,
      });
      setScraperResult(result);
    } catch (err: any) {
      setScraperError(err.response?.data?.detail || 'Chyba při spouštění scraperu');
    } finally {
      setScraperLoading(false);
    }
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-6">Události v Plzni</h1>

      {/* Tabs */}
      <div className="flex gap-2 mb-6 border-b">
        <button
          onClick={() => setActiveTab('list')}
          className={`px-4 py-2 font-medium transition ${
            activeTab === 'list'
              ? 'border-b-2 border-blue-600 text-blue-600'
              : 'text-gray-600 hover:text-gray-900'
          }`}
        >
          Seznam událostí
        </button>
        <button
          onClick={() => setActiveTab('scraper')}
          className={`px-4 py-2 font-medium transition ${
            activeTab === 'scraper'
              ? 'border-b-2 border-blue-600 text-blue-600'
              : 'text-gray-600 hover:text-gray-900'
          }`}
        >
          Scraper
        </button>
      </div>

      {/* Content */}
      {activeTab === 'list' && (
        <EventsManagement
          startDate={startDate}
          endDate={endDate}
        />
      )}

      {activeTab === 'scraper' && (
        <div className="space-y-6">
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-bold mb-4">
              Automatický scraper událostí
            </h2>
            <p className="text-gray-600 mb-6">
              Scraper automaticky stahuje události z GoOut a Plzen.eu a ukládá je
              do databáze.
            </p>

            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium mb-1">
                    Datum od
                  </label>
                  <input
                    type="date"
                    value={startDate}
                    onChange={(e) => setStartDate(e.target.value)}
                    className="w-full px-3 py-2 border rounded-lg"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1">
                    Datum do
                  </label>
                  <input
                    type="date"
                    value={endDate}
                    onChange={(e) => setEndDate(e.target.value)}
                    className="w-full px-3 py-2 border rounded-lg"
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">Zdroje</label>
                <div className="flex gap-4">
                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={sources.includes('goout')}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setSources([...sources, 'goout']);
                        } else {
                          setSources(sources.filter((s) => s !== 'goout'));
                        }
                      }}
                      className="rounded"
                    />
                    <span>GoOut</span>
                  </label>
                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={sources.includes('plzen.eu')}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setSources([...sources, 'plzen.eu']);
                        } else {
                          setSources(sources.filter((s) => s !== 'plzen.eu'));
                        }
                      }}
                      className="rounded"
                    />
                    <span>Plzen.eu</span>
                  </label>
                </div>
              </div>

              <button
                onClick={handleRunScraper}
                disabled={scraperLoading || sources.length === 0}
                className="w-full px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition disabled:opacity-50 font-medium"
              >
                {scraperLoading ? 'Scrapuji...' : 'Spustit scraper'}
              </button>
            </div>

            {/* Scraper Result */}
            {scraperResult && (
              <div className="mt-6 p-4 bg-green-50 border border-green-200 rounded-lg">
                <h3 className="font-semibold text-green-800 mb-2">
                  ✓ Scraping dokončen!
                </h3>
                <div className="text-sm text-green-700 space-y-1">
                  <p>Nalezeno: {scraperResult.events_found} událostí</p>
                  <p>Uloženo: {scraperResult.events_saved} nových událostí</p>
                  <p>Zdroje: {scraperResult.sources_scraped.join(', ')}</p>
                  <p className="text-xs mt-2">{scraperResult.message}</p>
                </div>
              </div>
            )}

            {/* Scraper Error */}
            {scraperError && (
              <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg">
                <h3 className="font-semibold text-red-800 mb-2">✗ Chyba</h3>
                <p className="text-sm text-red-700">{scraperError}</p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
