import type { 
  PredictionRequest, 
  PredictionResponse, 
  RangePredictionRequest, 
  RangePredictionResponse,
  HealthResponse,
  ModelsInfoResponse,
  StatsResponse,
  HistoricalDataResponse,
  TodayVisitorsResponse,
  TimeRange,
  AggregatedDataPoint,
  SeasonalityData,
  CorrelationData,
  CalendarHeatmapData,
  PredictionHistoryResponse,
  CalendarEventsResponse
} from '../types/api';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000';

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

  async healthCheck(): Promise<HealthResponse> {
    const response = await fetch(`${API_BASE_URL}/health`);
    return response.json();
  },

  async getModelsInfo(): Promise<ModelsInfoResponse> {
    const response = await fetch(`${API_BASE_URL}/models/info`);
    if (!response.ok) {
      throw new Error('Chyba při získávání informací o modelech');
    }
    return response.json();
  },

  async getStats(): Promise<StatsResponse> {
    const response = await fetch(`${API_BASE_URL}/stats`);
    if (!response.ok) {
      throw new Error('Chyba při získávání statistik');
    }
    return response.json();
  },

  async getHistoricalData(days?: number): Promise<HistoricalDataResponse> {
    const url = days 
      ? `${API_BASE_URL}/historical?days=${days}`
      : `${API_BASE_URL}/historical`;
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error('Chyba při získávání historických dat');
    }
    return response.json();
  },

  async getTodayVisitors(): Promise<TodayVisitorsResponse> {
    const response = await fetch(`${API_BASE_URL}/today`);
    if (!response.ok) {
      // Pokud endpoint neexistuje, vrátíme simulovaná data
      const today = new Date().toISOString().split('T')[0];
      const prediction = await this.predict({ date: today });
      return {
        date: today,
        current_visitors: prediction.predicted_visitors,
        predicted_visitors: prediction.predicted_visitors,
        difference: 0,
        percentage_difference: 0,
        last_updated: new Date().toISOString()
      };
    }
    
    // Transformovat odpověď z backendu na očekávaný formát
    const data = await response.json();
    
    // Backend vrací: { date, visitors, is_historical, day_of_week, is_weekend, is_holiday, weather }
    // Frontend očekává: { date, current_visitors, predicted_visitors, difference, percentage_difference, last_updated }
    const currentVisitors = data.visitors || 0;
    const predictedVisitors = data.is_historical ? currentVisitors : data.visitors;
    
    return {
      date: data.date,
      current_visitors: currentVisitors,
      predicted_visitors: predictedVisitors,
      difference: currentVisitors - predictedVisitors,
      percentage_difference: predictedVisitors > 0 
        ? Math.round(((currentVisitors - predictedVisitors) / predictedVisitors) * 100) 
        : 0,
      last_updated: new Date().toISOString()
    };
  },

  async getAggregatedData(timeRange: TimeRange, startDate?: string, endDate?: string): Promise<AggregatedDataPoint[]> {
    const params = new URLSearchParams({ time_range: timeRange });
    if (startDate) params.append('start_date', startDate);
    if (endDate) params.append('end_date', endDate);
    
    const response = await fetch(`${API_BASE_URL}/analytics/aggregated?${params}`);
    if (!response.ok) {
      // Fallback: agregujeme data na frontendu
      const historical = await this.getHistoricalData();
      return this.aggregateDataLocally(historical.data, timeRange);
    }
    return response.json();
  },

  async getSeasonalityData(): Promise<SeasonalityData> {
    const response = await fetch(`${API_BASE_URL}/analytics/seasonality`);
    if (!response.ok) {
      // Fallback: počítáme na frontendu
      const historical = await this.getHistoricalData();
      return this.calculateSeasonalityLocally(historical.data);
    }
    return response.json();
  },

  async getCorrelationData(): Promise<CorrelationData> {
    const response = await fetch(`${API_BASE_URL}/analytics/correlation`);
    if (!response.ok) {
      throw new Error('Chyba při získávání korelačních dat');
    }
    const data = await response.json();
    // API vrací { correlations: {...} }, ale potřebujeme přímo data
    return data.correlations || data;
  },

  async getCalendarHeatmapData(year?: number): Promise<CalendarHeatmapData[]> {
    const url = year 
      ? `${API_BASE_URL}/analytics/heatmap?year=${year}`
      : `${API_BASE_URL}/analytics/heatmap`;
    const response = await fetch(url);
    if (!response.ok) {
      // Fallback: transformujeme historická data
      const historical = await this.getHistoricalData();
      return historical.data.map(d => {
        const date = new Date(d.date);
        return {
          date: d.date,
          visitors: d.visitors,
          day_of_week: date.getDay(),
          week_of_year: this.getWeekNumber(date)
        };
      });
    }
    const result = await response.json();
    // API vrací { year, data, min_visitors, max_visitors }, potřebujeme jen data pole
    return result.data || result;
  },

  // Pomocné funkce pro fallback výpočty
  aggregateDataLocally(data: any[], timeRange: TimeRange): AggregatedDataPoint[] {
    // Jednoduchá agregace dat podle time range
    const grouped: { [key: string]: number[] } = {};
    
    data.forEach(d => {
      const date = new Date(d.date);
      let key = '';
      
      switch (timeRange) {
        case 'daily':
          key = d.date;
          break;
        case 'weekly':
          const weekNum = this.getWeekNumber(date);
          key = `${date.getFullYear()}-W${weekNum}`;
          break;
        case 'monthly':
          key = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
          break;
        case 'quarterly':
          const quarter = Math.floor(date.getMonth() / 3) + 1;
          key = `${date.getFullYear()}-Q${quarter}`;
          break;
        case 'yearly':
          key = `${date.getFullYear()}`;
          break;
      }
      
      if (!grouped[key]) grouped[key] = [];
      grouped[key].push(d.visitors);
    });
    
    return Object.entries(grouped).map(([period, visitors]) => ({
      period,
      visitors: visitors.reduce((a, b) => a + b, 0),
      avg_visitors: visitors.reduce((a, b) => a + b, 0) / visitors.length,
      min_visitors: Math.min(...visitors),
      max_visitors: Math.max(...visitors)
    }));
  },

  calculateSeasonalityLocally(data: any[]): SeasonalityData {
    const byWeekday: { [key: string]: number[] } = {};
    const byMonth: { [key: string]: number[] } = {};
    const weekdays = ['Neděle', 'Pondělí', 'Úterý', 'Středa', 'Čtvrtek', 'Pátek', 'Sobota'];
    const months = ['Leden', 'Únor', 'Březen', 'Duben', 'Květen', 'Červen', 
                    'Červenec', 'Srpen', 'Září', 'Říjen', 'Listopad', 'Prosinec'];
    
    data.forEach(d => {
      const date = new Date(d.date);
      const weekday = weekdays[date.getDay()];
      const month = months[date.getMonth()];
      
      if (!byWeekday[weekday]) byWeekday[weekday] = [];
      if (!byMonth[month]) byMonth[month] = [];
      
      byWeekday[weekday].push(d.visitors);
      byMonth[month].push(d.visitors);
    });
    
    const avgByWeekday: { [key: string]: number } = {};
    const avgByMonth: { [key: string]: number } = {};
    
    Object.entries(byWeekday).forEach(([day, visitors]) => {
      avgByWeekday[day] = visitors.reduce((a, b) => a + b, 0) / visitors.length;
    });
    
    Object.entries(byMonth).forEach(([month, visitors]) => {
      avgByMonth[month] = visitors.reduce((a, b) => a + b, 0) / visitors.length;
    });
    
    return {
      by_weekday: avgByWeekday,
      by_month: avgByMonth,
      holiday_vs_regular: {
        holiday_avg: 0,
        regular_avg: 0,
        difference: 0
      }
    };
  },

  getWeekNumber(date: Date): number {
    const d = new Date(Date.UTC(date.getFullYear(), date.getMonth(), date.getDate()));
    const dayNum = d.getUTCDay() || 7;
    d.setUTCDate(d.getUTCDate() + 4 - dayNum);
    const yearStart = new Date(Date.UTC(d.getUTCFullYear(), 0, 1));
    return Math.ceil((((d.getTime() - yearStart.getTime()) / 86400000) + 1) / 7);
  },

  async getPredictionHistory(days?: number, includeFuture?: boolean): Promise<PredictionHistoryResponse> {
    const params = new URLSearchParams();
    if (days !== undefined) params.append('days', days.toString());
    if (includeFuture !== undefined) params.append('include_future', includeFuture.toString());
    
    const url = `${API_BASE_URL}/analytics/prediction-history${params.toString() ? '?' + params.toString() : ''}`;
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error('Chyba při získávání historie predikcí');
    }
    return response.json();
  },

  async getCalendarEvents(month?: number, year?: number): Promise<CalendarEventsResponse> {
    const params = new URLSearchParams();
    if (month !== undefined) params.append('month', month.toString());
    if (year !== undefined) params.append('year', year.toString());
    
    const url = `${API_BASE_URL}/calendar/events${params.toString() ? '?' + params.toString() : ''}`;
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error('Chyba při získávání událostí kalendáře');
    }
    return response.json();
  },

  // AI Chat - streaming
  async *chatStream(message: string, history?: Array<{role: string, content: string}>): AsyncGenerator<string> {
    const response = await fetch(`${API_BASE_URL}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ message, history }),
    });

    if (!response.ok) {
      throw new Error('Chyba při komunikaci s AI');
    }

    const reader = response.body?.getReader();
    const decoder = new TextDecoder();

    if (!reader) {
      throw new Error('Stream není dostupný');
    }

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') return;
          
          try {
            const parsed = JSON.parse(data);
            if (parsed.content) {
              yield parsed.content;
            }
          } catch {
            // Ignorovat neplatný JSON
          }
        }
      }
    }
  },

  // AI Chat - synchronní
  async chatSync(message: string, history?: Array<{role: string, content: string}>): Promise<string> {
    const response = await fetch(`${API_BASE_URL}/chat/sync`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ message, history }),
    });

    if (!response.ok) {
      throw new Error('Chyba při komunikaci s AI');
    }

    const data = await response.json();
    return data.response;
  }
};