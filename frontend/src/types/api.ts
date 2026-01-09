export interface PredictionRequest {
  date: string;
  is_holiday: boolean;
  opening_hours: string;
}

export interface PredictionResponse {
  date: string;
  predicted_visitors: number;
  confidence_interval: {
    lower: number;
    upper: number;
  };
}

export interface RangePredictionRequest {
  start_date: string;
  end_date: string;
}

export interface RangePredictionResponse {
  predictions: Array<{
    date: string;
    predicted_visitors: number;
  }>;
  total_predicted: number;
}
