export interface PredictionRequest {
  date: string;
  is_holiday?: boolean | null;
  opening_hours?: string;
}

export interface WeatherInfo {
  temperature_mean: number;
  precipitation: number;
  weather_description: string;
  is_nice_weather: boolean;
}

export interface HolidayInfo {
  is_holiday: boolean;
  holiday_name: string | null;
}

export interface PredictionResponse {
  date: string;
  predicted_visitors: number;
  confidence_interval: {
    lower: number;
    upper: number;
  };
  model_info: {
    type: string;
    models: string[];
    weights: Record<string, number>;
  };
  holiday_info: HolidayInfo;
  weather_info: WeatherInfo;
}

export interface RangePredictionRequest {
  start_date: string;
  end_date: string;
}

export interface DayPrediction {
  date: string;
  predicted_visitors: number;
  confidence_interval: {
    lower: number;
    upper: number;
  };
  holiday_info: HolidayInfo;
  weather_info: WeatherInfo;
  day_of_week: string;
  is_weekend: boolean;
}

export interface RangePredictionResponse {
  predictions: DayPrediction[];
  total_predicted: number;
  average_daily: number;
  period_days: number;
}

export interface HealthResponse {
  status: string;
  models_loaded: {
    lightgbm: boolean;
    xgboost: boolean;
    catboost: boolean;
  };
  features_count: number | null;
}

export interface ModelsInfoResponse {
  models: string[];
  ensemble_weights: Record<string, number>;
  features_count: number;
  feature_sample: string[];
}

export interface StatsResponse {
  total_visitors: number;
  avg_daily_visitors: number;
  peak_day: string;
  peak_visitors: number;
  trend: number;
  data_start_date: string;
  data_end_date: string;
}

export interface HistoricalDataPoint {
  date: string;
  visitors: number;
}

export interface HistoricalDataResponse {
  data: HistoricalDataPoint[];
  start_date: string;
  end_date: string;
  total_days: number;
}
