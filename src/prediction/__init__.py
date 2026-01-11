"""
Prediction utilities - refactored prediction logic.

Tento modul obsahuje utility funkce pro:
- Načítání dat (data_loader)
- Zpracování weather dat (weather_processor)
- Zpracování holiday dat (holiday_processor)
- Feature engineering (feature_processor)
- Model predikce (model_predictor)
- Confidence intervaly (confidence)
"""

from .data_loader import (
    load_historical_data,
    load_template_2026,
    combine_historical_and_new
)

from .weather_processor import (
    get_weather_for_date,
    fill_missing_weather_features,
    estimate_precipitation_probability
)

from .holiday_processor import (
    get_holiday_features,
    get_holiday_from_template,
    get_holiday_from_api
)

from .feature_processor import (
    prepare_features_for_prediction,
    handle_missing_features,
    convert_to_numeric,
    add_google_trend_feature
)

from .model_predictor import (
    predict_with_models,
    ensemble_prediction,
    should_use_catboost,
    get_effective_weights
)

from .confidence import (
    calculate_confidence_interval,
    calculate_confidence_intervals_batch
)

__all__ = [
    'load_historical_data',
    'load_template_2026',
    'combine_historical_and_new',
    'get_weather_for_date',
    'fill_missing_weather_features',
    'estimate_precipitation_probability',
    'get_holiday_features',
    'get_holiday_from_template',
    'get_holiday_from_api',
    'prepare_features_for_prediction',
    'handle_missing_features',
    'convert_to_numeric',
    'add_google_trend_feature',
    'predict_with_models',
    'ensemble_prediction',
    'should_use_catboost',
    'get_effective_weights',
    'calculate_confidence_interval',
    'calculate_confidence_intervals_batch',
]
