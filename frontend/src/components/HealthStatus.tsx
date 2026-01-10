'use client';

import { useEffect, useState } from 'react';
import { api } from '@/lib/api';
import type { HealthResponse } from '@/types/api';

export default function HealthStatus() {
  const [status, setStatus] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const health = await api.health();
        setStatus(health);
      } catch (error) {
        console.error('Health check failed:', error);
        setStatus(null);
      } finally {
        setLoading(false);
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 30000); // Check every 30s

    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="flex items-center space-x-2 text-sm text-gray-500">
        <div className="h-2 w-2 rounded-full bg-gray-400 animate-pulse"></div>
        <span>Kontrola připojení...</span>
      </div>
    );
  }

  const isHealthy = status?.status === 'healthy' && 
    status?.models_loaded?.lightgbm && 
    status?.models_loaded?.xgboost && 
    status?.models_loaded?.catboost;

  return (
    <div className="flex items-center space-x-2 text-sm">
      <div
        className={`h-2 w-2 rounded-full ${
          isHealthy ? 'bg-green-500' : 'bg-red-500'
        }`}
      ></div>
      <span className={isHealthy ? 'text-green-700' : 'text-red-700'}>
        {isHealthy ? 'API připojeno' : 'API nedostupné'}
      </span>
      {status?.features_count && (
        <span className="text-xs text-gray-500">
          ({status.features_count} features)
        </span>
      )}
    </div>
  );
}
