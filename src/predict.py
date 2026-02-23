"""
Prediction Module - Použití natrénovaných ensemble modelů (REFACTORED)
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime, date as date_type
from pathlib import Path
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Import feature engineering
from feature_engineering_v3 import create_features

# Import refactored prediction utilities
from prediction import (
    load_historical_data,
    load_template_2026,
    combine_historical_and_new,
    get_weather_for_date,
    get_holiday_features,
    convert_to_numeric,
    add_google_trend_feature,
    predict_with_models,
    ensemble_prediction,
    should_use_catboost,
    get_effective_weights,
    calculate_confidence_interval,
    calculate_confidence_intervals_batch
)

TREND_FEATURES = [
    'year', 'month', 'day_of_week', 'week_of_year', 'quarter',
    'is_weekend', 'is_summer_holiday', 'is_winter_holiday', 'is_school_year',
    'is_oct_28', 'is_autumn_break', 'is_summer_weekend_event', 'event_score',
    'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos',
    'week_sin', 'week_cos', 'normalized_time'
]


def load_models():
    """
    Načte všechny natrénované modely.
    
    Returns:
        Dict s modely a pomocnými objekty
    """
    print("📦 Loading models...")
    
    try:
        import os
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        
        # Načíst V3 modely
        models = {
            'lgb': joblib.load(os.path.join(models_dir, 'lightgbm_v3.pkl')),
            'xgb': joblib.load(os.path.join(models_dir, 'xgboost_v3.pkl')),
            'cat': joblib.load(os.path.join(models_dir, 'catboost_v3.pkl')),
            'weights': joblib.load(os.path.join(models_dir, 'ensemble_weights_v3.pkl')),
            'feature_cols': joblib.load(os.path.join(models_dir, 'feature_names_v3.pkl')),
            'google_trend_predictor': joblib.load(os.path.join(models_dir, 'google_trend_predictor_v3.pkl')),
            'historical_mae': joblib.load(os.path.join(models_dir, 'historical_mae_v3.pkl'))
        }
        
        models['ensemble_type'] = 'weighted'
        print(f"✅ Models V3 loaded successfully! (Ensemble: WEIGHTED - 3 models)")
        print(f"   Historical MAE - Weekday: {models['historical_mae']['weekday']:.2f}, Weekend: {models['historical_mae']['weekend']:.2f}")
        
        return models
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("   Please train the V3 models first by running: python src/ensemble_model_v3.py")
        return None


def _ensure_google_trend_feature(
    X_source: pd.DataFrame,
    df_combined: pd.DataFrame,
    pred_dates: pd.Series,
    models_dict: Dict
) -> pd.DataFrame:
    """Zajistí, že predicted_google_trend je připravená bez hardcoded fallbacku."""
    X_prepared = X_source.copy()
    predictor = models_dict.get('google_trend_predictor')
    trend_features = models_dict.get('trend_features', TREND_FEATURES)

    if predictor is not None:
        trend_input = pd.DataFrame(index=X_prepared.index)
        if 'google_trend' in X_prepared.columns:
            trend_input['google_trend'] = X_prepared['google_trend']
        trend_output = add_google_trend_feature(
            trend_input,
            df_combined,
            pred_dates,
            predictor,
            trend_features
        )
        X_prepared['predicted_google_trend'] = trend_output['predicted_google_trend'].values
        return X_prepared

    if 'google_trend' in X_prepared.columns and X_prepared['google_trend'].notna().all():
        X_prepared['predicted_google_trend'] = X_prepared['google_trend']
        return X_prepared

    raise ValueError(
        "Chybí google_trend_predictor a zároveň není dostupná validní "
        "historická feature 'google_trend'."
    )


def _build_feature_matrix(
    df_pred: pd.DataFrame,
    feature_cols: list
) -> pd.DataFrame:
    """Validuje feature kontrakt a vrátí číselnou matici bez NaN."""
    if df_pred.empty:
        raise ValueError("DataFrame pro predikci je prázdný.")

    missing_cols = [col for col in feature_cols if col not in df_pred.columns]
    if missing_cols:
        raise ValueError(f"Chybí požadované featury pro predikci: {missing_cols}")

    X_pred = df_pred[feature_cols].copy()
    X_pred = convert_to_numeric(X_pred)

    nan_cols = X_pred.columns[X_pred.isna().any()].tolist()
    if nan_cols:
        raise ValueError(f"Feature kontrakt porušen: NaN hodnoty ve sloupcích {nan_cols}")

    return X_pred


def predict_single_date(
    date,
    models_dict: Dict,
    historical_df: Optional[pd.DataFrame] = None
) -> Dict:
    """
    Predikuje návštěvnost pro konkrétní datum (REFACTORED).
    
    Args:
        date: datetime nebo string ve formátu 'YYYY-MM-DD'
        models_dict: Dict s natrénovanými modely
        historical_df: DataFrame s historickými daty (pokud není, načte se)
        
    Returns:
        Dict s predikcemi a detaily
    """
    # 1. Načíst data
    if historical_df is None:
        df_historical = load_historical_data()
    else:
        df_historical = historical_df
    
    df_template = load_template_2026()
    
    # Parsovat datum
    if isinstance(date, str):
        date = pd.to_datetime(date)
    pred_date = date.date() if isinstance(date, pd.Timestamp) else date
    pred_date_ts = pd.to_datetime(pred_date)
    
    # 2. Získat weather data
    print(f"\n🔮 Predicting for {pred_date}")
    weather_data = get_weather_for_date(pred_date, df_historical)
    
    # 3. Získat holiday data
    holiday_data = get_holiday_features(pred_date, df_template)
    
    # 4. Vytvořit nový řádek pro predikci
    new_row = pd.DataFrame({
        'date': [pred_date_ts],
        'total_visitors': [np.nan],
        'school_visitors': [np.nan],
        'public_visitors': [np.nan],
        'extra': [None],
        'opening_hours': [None],
        **{k: [v] for k, v in weather_data.items() if k != 'weather_description'},
        **{k: [v] for k, v in holiday_data.items()}
    })
    
    # 5. Feature engineering
    df_combined = combine_historical_and_new(df_historical, new_row)
    df_combined = create_features(df_combined)
    
    # 6. Připravit features s fail-fast validací kontraktu
    df_pred = df_combined[df_combined['date'] == pred_date_ts].copy()
    df_pred = _ensure_google_trend_feature(
        df_pred,
        df_combined,
        pd.Series([pred_date_ts]),
        models_dict
    )
    feature_cols = models_dict['feature_cols']
    X_pred = _build_feature_matrix(df_pred, feature_cols)
    
    # 8. Model predictions
    predictions = predict_with_models(X_pred, models_dict)
    
    # 9. Ensemble
    is_weekend = X_pred['is_weekend'].values[0] == 1
    is_holiday = X_pred['is_holiday'].values[0] == 1
    use_catboost = should_use_catboost(is_weekend, is_holiday)
    
    ensemble_pred = ensemble_prediction(
        predictions,
        models_dict['weights'],
        np.array([is_weekend]),
        np.array([is_holiday]),
        models_dict.get('ensemble_type', 'weighted'),
        models_dict.get('meta_model')
    )[0]
    
    ensemble_pred = int(round(max(ensemble_pred, 0)))
    
    # 10. Confidence interval
    historical_mae = models_dict.get('historical_mae')
    if historical_mae is None:
        raise ValueError(
            "Missing 'historical_mae' in models_dict. "
            "Cannot calculate confidence intervals without historical MAE data."
        )
    
    confidence_lower, confidence_upper = calculate_confidence_interval(
        ensemble_pred,
        is_weekend,
        is_holiday,
        historical_mae
    )
    
    # 11. Formátovat výsledek
    effective_weights = get_effective_weights(models_dict['weights'], use_catboost)
    
    result = {
        'date': date,
        'day_of_week': pred_date.strftime('%A'),
        'ensemble_prediction': ensemble_pred,
        'ensemble_type': models_dict.get('ensemble_type', 'weighted'),
        'confidence_interval': (confidence_lower, confidence_upper),
        'is_holiday': bool(is_holiday),
        'catboost_used': use_catboost,
        'individual_predictions': {
            'lightgbm': int(round(predictions['lightgbm'][0])),
            'xgboost': int(round(predictions['xgboost'][0])),
            'catboost': int(round(predictions['catboost'][0]))
        },
        'model_weights': effective_weights,
        'weather': {
            'description': weather_data.get('weather_description', 'N/A'),
            'temperature': weather_data['temperature_mean'],
            'precipitation': weather_data['precipitation'],
            'rain': weather_data.get('rain', weather_data['precipitation']),
            'snowfall': weather_data.get('snowfall', 0.0),
            'wind_speed_max': weather_data.get('wind_speed'),
            'is_nice_weather': bool(weather_data.get('is_nice_weather', 0)),
        }
    }
    
    return result


def predict_date_range(
    start_date,
    end_date,
    models_dict: Dict
) -> pd.DataFrame:
    """
    Predikuje návštěvnost pro rozsah dat (REFACTORED).
    
    Args:
        start_date: Začátek období
        end_date: Konec období
        models_dict: Dict s natrénovanými modely
        
    Returns:
        DataFrame s predikcemi
    """
    # 1. Načíst data
    df_historical = load_historical_data()
    df_template = load_template_2026()
    
    # Parsovat data
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    print(f"\n🔮 Predicting {len(date_range)} days...")
    print(f"📥 Downloading weather data for {len(date_range)} days...")
    
    # 2. Získat weather + holiday data pro všechny dny
    new_rows = []
    for date in date_range:
        pred_date = date.date()
        
        # Weather
        weather_data = get_weather_for_date(pred_date, df_historical)
        
        # Holiday
        holiday_data = get_holiday_features(pred_date, df_template)
        
        # Kombinovat
        row_data = {
            'date': date,
            'total_visitors': np.nan,
            'school_visitors': np.nan,
            'public_visitors': np.nan,
            'extra': None,
            'opening_hours': None,
            **{k: v for k, v in weather_data.items() if k != 'weather_description'},
            **holiday_data
        }
        
        new_rows.append(row_data)
    
    df_new = pd.DataFrame(new_rows)
    print(f"✅ Weather data downloaded for {len(df_new)} days")
    
    # 3. Feature engineering
    df_combined = combine_historical_and_new(df_historical, df_new)
    df_combined = create_features(df_combined)
    
    # 4. Připravit features s fail-fast validací kontraktu
    df_pred = df_combined[df_combined['date'].isin(date_range)].copy()
    
    # Odstranit duplikáty - vzít pouze poslední řádek pro každé datum
    df_pred = df_pred.drop_duplicates(subset=['date'], keep='last')
    
    df_pred = _ensure_google_trend_feature(
        df_pred,
        df_combined,
        df_pred['date'].reset_index(drop=True),
        models_dict
    )
    feature_cols = models_dict['feature_cols']
    X_pred = _build_feature_matrix(df_pred, feature_cols)
    
    # 6. Model predictions
    print(f"🤖 Running predictions...")
    predictions = predict_with_models(X_pred, models_dict)
    
    # 7. Ensemble
    is_weekend = (X_pred['is_weekend'].values == 1)
    is_holiday = (X_pred['is_holiday'].values == 1)
    
    ensemble_preds = ensemble_prediction(
        predictions,
        models_dict['weights'],
        is_weekend,
        is_holiday,
        models_dict.get('ensemble_type', 'weighted'),
        models_dict.get('meta_model')
    )
    
    ensemble_preds = np.maximum(ensemble_preds, 0)
    ensemble_preds = np.round(ensemble_preds).astype(int)
    
    # 8. Confidence intervals
    historical_mae = models_dict.get('historical_mae')
    if historical_mae is None:
        raise ValueError(
            "Missing 'historical_mae' in models_dict. "
            "Cannot calculate confidence intervals without historical MAE data."
        )
    
    lower_bounds, upper_bounds = calculate_confidence_intervals_batch(
        ensemble_preds,
        is_weekend,
        is_holiday,
        historical_mae
    )
    
    # 9. Sestavit výsledky
    results = pd.DataFrame({
        'date': df_pred['date'].values,
        'day_of_week': df_pred['date'].dt.strftime('%A'),
        'prediction': ensemble_preds,
        'lower_bound': lower_bounds,
        'upper_bound': upper_bounds,
        'lightgbm': np.round(predictions['lightgbm']).astype(int),
        'xgboost': np.round(predictions['xgboost']).astype(int),
        'catboost': np.round(predictions['catboost']).astype(int)
    })
    
    print(f"✅ Predicted {len(results)} days successfully!")
    
    return results


def print_prediction(result: Dict):
    """
    Pěkně vypíše výsledek predikce.
    
    Args:
        result: Dict s predikcí
    """
    print("\n" + "=" * 60)
    print(f"🔮 PREDIKCE PRO {result['date'].strftime('%d.%m.%Y')} ({result['day_of_week']})")
    print("=" * 60)
    
    print(f"\n🎯 ENSEMBLE PREDIKCE: {result['ensemble_prediction']} návštěvníků")
    print(f"   95% Confidence Interval: [{result['confidence_interval'][0]} - {result['confidence_interval'][1]}]")
    
    catboost_status = "ACTIVE" if result.get('catboost_used', True) else "DISABLED (weekday)"
    
    print(f"\n📊 Jednotlivé modely:")
    print(f"   LightGBM (váha {result['model_weights']['lightgbm']:.1%}): {result['individual_predictions']['lightgbm']} návštěvníků")
    print(f"   XGBoost (váha {result['model_weights']['xgboost']:.1%}): {result['individual_predictions']['xgboost']} návštěvníků")
    print(f"   CatBoost (váha {result['model_weights']['catboost']:.1%}, {catboost_status}): {result['individual_predictions']['catboost']} návštěvníků")
    
    print("=" * 60)


def main():
    """
    Demo použití predikčního modulu.
    """
    print("\n" + "=" * 60)
    print("🎯 ENSEMBLE PREDICTION SYSTEM")
    print("=" * 60)
    
    # Načíst modely
    models = load_models()
    
    if models is None:
        return
    
    # Příklad 1: Predikce pro následující den
    print("\n📅 Příklad 1: Predikce pro následující den")
    
    from datetime import date as dt_date, timedelta
    next_day = dt_date.today() + timedelta(days=1)
    next_day_str = next_day.strftime('%Y-%m-%d')
    
    print(f"   Predikuji pro datum: {next_day_str}")
    result = predict_single_date(next_day_str, models)
    print_prediction(result)
    
    # Příklad 2: Predikce pro následujících 7 dní
    print("\n📅 Příklad 2: Predikce pro následujících 7 dní")
    
    start_date = dt_date.today() + timedelta(days=1)
    end_date = start_date + timedelta(days=6)
    
    print(f"   Období: {start_date.strftime('%Y-%m-%d')} až {end_date.strftime('%Y-%m-%d')}")
    
    predictions = predict_date_range(start_date, end_date, models)
    print("\n" + str(predictions))
    
    # Uložit výsledky
    import os
    output_file = os.path.join(os.path.dirname(__file__), '..', 'predictions_next_week.csv')
    predictions.to_csv(output_file, index=False)
    print(f"\n💾 Predictions saved to: {output_file}")
    
    print("\n" + "=" * 60)
    print("✅ PREDICTION COMPLETE!")
    print("=" * 60)


if __name__ == '__main__':
    main()
