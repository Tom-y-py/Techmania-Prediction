/**
 * React Hook pro práci s API
 * Poskytuje funkce pro volání API s error handlingem a loading states
 */

import { useState, useCallback } from 'react';
import { api, handleApiError } from '@/lib/api';

export interface UseApiState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
}

export function useApi<T = any>() {
  const [state, setState] = useState<UseApiState<T>>({
    data: null,
    loading: false,
    error: null,
  });

  const execute = useCallback(async (apiCall: () => Promise<T>) => {
    setState({ data: null, loading: true, error: null });
    
    try {
      const result = await apiCall();
      setState({ data: result, loading: false, error: null });
      return result;
    } catch (error) {
      const errorMessage = handleApiError(error);
      setState({ data: null, loading: false, error: errorMessage });
      throw error;
    }
  }, []);

  const reset = useCallback(() => {
    setState({ data: null, loading: false, error: null });
  }, []);

  return {
    ...state,
    execute,
    reset,
  };
}

// Specialized hooks pro konkrétní use cases

export function usePrediction() {
  const { execute, ...state } = useApi();
  
  const predictSingleDate = useCallback(
    async (date: string) => {
      return execute(() => api.predictVisitors({ date }));
    },
    [execute]
  );

  const predictRange = useCallback(
    async (start_date: string, end_date: string) => {
      return execute(() => api.predictRange({ start_date, end_date }));
    },
    [execute]
  );

  return {
    ...state,
    predictSingleDate,
    predictRange,
  };
}

export function useEvents() {
  const { execute, ...state } = useApi();
  
  const fetchEvents = useCallback(
    async (params?: Parameters<typeof api.getEvents>[0]) => {
      return execute(() => api.getEvents(params));
    },
    [execute]
  );

  const fetchEventsForDate = useCallback(
    async (date: string) => {
      return execute(() => api.getEventsForDate(date));
    },
    [execute]
  );

  const createEvent = useCallback(
    async (event: Parameters<typeof api.createEvent>[0]) => {
      return execute(() => api.createEvent(event));
    },
    [execute]
  );

  const updateEvent = useCallback(
    async (eventId: number, event: Parameters<typeof api.updateEvent>[1]) => {
      return execute(() => api.updateEvent(eventId, event));
    },
    [execute]
  );

  const deleteEvent = useCallback(
    async (eventId: number) => {
      return execute(() => api.deleteEvent(eventId));
    },
    [execute]
  );

  return {
    ...state,
    fetchEvents,
    fetchEventsForDate,
    createEvent,
    updateEvent,
    deleteEvent,
  };
}

export function useAnalytics() {
  const { execute, ...state } = useApi();
  
  const fetchCorrelation = useCallback(
    async () => {
      return execute(() => api.getCorrelationAnalysis());
    },
    [execute]
  );

  const fetchSeasonality = useCallback(
    async () => {
      return execute(() => api.getSeasonalityAnalysis());
    },
    [execute]
  );

  const fetchHeatmap = useCallback(
    async (year?: number) => {
      return execute(() => api.getHeatmapData(year));
    },
    [execute]
  );

  const fetchPredictionHistory = useCallback(
    async (days: number = 30, includeFuture: boolean = true) => {
      return execute(() => api.getAnalyticsPredictionHistory(days, includeFuture));
    },
    [execute]
  );

  return {
    ...state,
    fetchCorrelation,
    fetchSeasonality,
    fetchHeatmap,
    fetchPredictionHistory,
  };
}

export function useChat() {
  const [messages, setMessages] = useState<Array<{ role: string; content: string }>>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const sendMessage = useCallback(
    async (message: string) => {
      setLoading(true);
      setError(null);

      try {
        const response = await api.chatSync({
          message,
          history: messages,
        });

        setMessages((prev) => [
          ...prev,
          { role: 'user', content: message },
          { role: 'assistant', content: response.response },
        ]);

        return response;
      } catch (err) {
        const errorMessage = handleApiError(err);
        setError(errorMessage);
        throw err;
      } finally {
        setLoading(false);
      }
    },
    [messages]
  );

  const clearMessages = useCallback(() => {
    setMessages([]);
    setError(null);
  }, []);

  return {
    messages,
    loading,
    error,
    sendMessage,
    clearMessages,
  };
}
