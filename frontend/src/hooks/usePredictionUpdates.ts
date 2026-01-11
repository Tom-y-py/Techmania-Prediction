'use client';

import { useEffect, useRef } from 'react';

/**
 * Hook pro automatickou detekci změn v predikcích a trigger refresh callbacku.
 * Používá polling mechanismus pro kontrolu nových predikcí v databázi.
 */
export function usePredictionUpdates(onUpdate: () => void, intervalMs: number = 30000) {
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const lastCheckRef = useRef<Date>(new Date());

  useEffect(() => {
    // Polling funkce - kontroluje změny v predikcích
    const checkForUpdates = async () => {
      try {
        const apiUrl = typeof window !== 'undefined' 
          ? (window as any).NEXT_PUBLIC_API_URL || 'http://localhost:8000'
          : 'http://localhost:8000';
        const response = await fetch(`${apiUrl}/predictions/latest?limit=1`);
        if (!response.ok) return;
        
        const data = await response.json();
        
        if (data.predictions && data.predictions.length > 0) {
          const latestPrediction = data.predictions[0];
          const createdAt = new Date(latestPrediction.created_at);
          
          // Pokud je predikce novější než poslední kontrola, zavoláme callback
          if (createdAt > lastCheckRef.current) {
            lastCheckRef.current = new Date();
            onUpdate();
          }
        }
      } catch (error) {
        console.error('Chyba při kontrole predikcí:', error);
      }
    };

    // Spustit polling
    intervalRef.current = setInterval(checkForUpdates, intervalMs);

    // Cleanup při unmount
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [onUpdate, intervalMs]);
}
