import type { PredictionRequest, PredictionResponse, RangePredictionRequest, RangePredictionResponse } from '../types/api';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

export const api = {
  async predict(data: PredictionRequest): Promise<PredictionResponse> {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Chyba při predikci');
    }

    return response.json();
  },

  async predictRange(data: RangePredictionRequest): Promise<RangePredictionResponse> {
    const response = await fetch(`${API_BASE_URL}/predict/range`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Chyba při predikci');
    }

    return response.json();
  },

  async healthCheck(): Promise<{ status: string; model_loaded: boolean }> {
    const response = await fetch(`${API_BASE_URL}/health`);
    return response.json();
  },
};
