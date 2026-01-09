"""
Prediction Module - Použití natrénovaných ensemble modelů
"""

import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from datetime import datetime, date as date_type
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Přidat app do path pro weather service
sys.path.append(str(Path(__file__).parent.parent / 'app'))

from feature_engineering import create_features
try:
    from services import weather_service, holiday_service
    SERVICES_AVAILABLE = True
except ImportError:
    print("⚠️ Weather/Holiday services nedostupné - použijí se průměrné hodnoty")
    SERVICES_AVAILABLE = False


def load_models():
    """
    Načte všechny natrénované modely
    
    Returns:
        Dict s modely a pomocnými objekty
    """
    print("📦 Loading models...")
    
    try:
        import os
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        
        models = {
            'lgb': joblib.load(os.path.join(models_dir, 'lightgbm_model.pkl')),
            'xgb': joblib.load(os.path.join(models_dir, 'xgboost_model.pkl')),
            'cat': joblib.load(os.path.join(models_dir, 'catboost_model.pkl')),
            'weights': joblib.load(os.path.join(models_dir, 'ensemble_weights.pkl')),
            'feature_cols': joblib.load(os.path.join(models_dir, 'feature_columns.pkl'))
        }
        print("✅ Models loaded successfully!")
        return models
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("   Please train the models first by running: python src/ensemble_model.py")
        return None


def predict_single_date(date, models_dict, historical_df=None):
    """
    Predikuje návštěvnost pro konkrétní datum
    
    Args:
        date: datetime nebo string ve formátu 'YYYY-MM-DD'
        models_dict: Dict s natrénovanými modely
        historical_df: DataFrame s historickými daty (pokud není, načte se)
        
    Returns:
        Dict s predikcemi a detaily
    """
    # Načíst historická data S POČASÍM (potřebujeme pro lag features)
    if historical_df is None:
        script_dir = Path(__file__).parent
        data_path = script_dir.parent / 'data' / 'processed' / 'techmania_with_weather.csv'
        
        if not data_path.exists():
            print("⚠️ techmania_with_weather.csv nenalezen, použiji data bez počasí")
            data_path = script_dir.parent / 'data' / 'raw' / 'techmania_cleaned_master.csv'
        
        df = pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['date'])
    else:
        df = historical_df.copy()
    
    # Přidat nový řádek pro predikci
    if isinstance(date, str):
        date = pd.to_datetime(date)
    
    # Získat weather data pro predikované datum
    weather_data = {}
    weather_description = None
    if SERVICES_AVAILABLE:
        try:
            pred_date = date.date() if isinstance(date, pd.Timestamp) else date
            weather_info = weather_service.get_weather(pred_date)
            
            weather_description = weather_info.get('weather_description', 'N/A')
            
            # Základní hodnoty z API
            weather_data = {
                'temperature_max': weather_info.get('temperature_max'),
                'temperature_min': weather_info.get('temperature_min'),
                'temperature_mean': weather_info.get('temperature_mean'),
                'precipitation': weather_info.get('precipitation'),
                'rain': weather_info.get('rain'),
                'snowfall': weather_info.get('snowfall'),
                'precipitation_hours': weather_info.get('precipitation_hours'),
                'weather_code': weather_info.get('weather_code'),
                'wind_speed_max': weather_info.get('wind_speed_max'),
                'wind_gusts_max': weather_info.get('wind_gusts_max'),
                'is_rainy': int(weather_info.get('is_rainy', False)),
                'is_snowy': int(weather_info.get('is_snowy', False)),
                'is_windy': int(weather_info.get('is_windy', False)),
                'is_nice_weather': int(weather_info.get('is_nice_weather', False)),
            }
            
            # Features které API nevrací - dopočítáme z historických dat pokud jsou dostupná
            # Pokud historická data nejsou dostupná, nastavíme value na np.nan (žádné tiché fallbacky)
            pred_month = date.month
            pred_day = date.day

            df_hist = df[df['date'] < date].copy()
            if len(df_hist) > 0:
                df_hist['month'] = df_hist['date'].dt.month
                df_hist['day'] = df_hist['date'].dt.day

                # Najít podobné dny (±15 dní)
                similar = df_hist[
                    ((df_hist['month'] == pred_month) & 
                     (abs(df_hist['day'] - pred_day) <= 15)) |
                    ((pred_month == 1) & (df_hist['month'] == 12) & (df_hist['day'] >= 17)) |
                    ((pred_month == 12) & (df_hist['month'] == 1) & (df_hist['day'] <= 15))
                ]

                if len(similar) < 10:
                    similar = df_hist[df_hist['month'] == pred_month]
                if len(similar) < 5:
                    similar = df_hist

                # Apparent temperature - pokud není v API, zkusíme zhistorických dat
                weather_data['apparent_temp_max'] = similar['apparent_temp_max'].median() if 'apparent_temp_max' in similar and len(similar) > 0 else np.nan
                weather_data['apparent_temp_min'] = similar['apparent_temp_min'].median() if 'apparent_temp_min' in similar and len(similar) > 0 else np.nan
                weather_data['apparent_temp_mean'] = similar['apparent_temp_mean'].median() if 'apparent_temp_mean' in similar and len(similar) > 0 else np.nan

                # Wind direction - medián z podobných dnů
                weather_data['wind_direction'] = similar['wind_direction'].median() if 'wind_direction' in similar and len(similar) > 0 else np.nan

                # Sunshine a daylight - z podobných dnů (pokud nejsou, ponecháme NaN)
                weather_data['sunshine_duration'] = similar['sunshine_duration'].median() if 'sunshine_duration' in similar and len(similar) > 0 else np.nan
                weather_data['daylight_duration'] = similar['daylight_duration'].median() if 'daylight_duration' in similar and len(similar) > 0 else np.nan
                weather_data['sunshine_ratio'] = similar['sunshine_ratio'].median() if 'sunshine_ratio' in similar and len(similar) > 0 else np.nan
            else:
                # Pokud nejsou historická data, žádné tiché výchozí hodnoty - použijeme NaN
                weather_data['apparent_temp_max'] = np.nan
                weather_data['apparent_temp_min'] = np.nan
                weather_data['apparent_temp_mean'] = np.nan
                weather_data['wind_direction'] = np.nan
                weather_data['sunshine_duration'] = np.nan
                weather_data['daylight_duration'] = np.nan
                weather_data['sunshine_ratio'] = np.nan

            # Kontrola, zda API vrátilo validní data (musí existovat nějaká číselná teplota)
            if pd.notna(weather_data.get('temperature_mean')):
                try:
                    print(f"   Weather: {weather_description}, Temp: {weather_data['temperature_mean']:.1f}°C")
                except Exception:
                    print(f"   Weather: {weather_description}, Temp: {weather_data.get('temperature_mean')}")
            else:
                print(f"   ⚠️ Weather API vrátilo neúplná data (ponecháno NaN)")
        except Exception as e:
            print(f"   ⚠️ Weather API error: {e}")
    
    # Pokud weather data nejsou dostupná se pokusíme získat co nejvíce z historických dat
    if not weather_data:
        print(f"   ℹ️  Weather data nedostupná - ponechávám hodnoty jako NaN pro explicitní ošetření")

        pred_month = date.month
        pred_day = date.day

        df_historical = df[df['date'] < date].copy()
        if len(df_historical) > 0:
            df_historical['month'] = df_historical['date'].dt.month
            df_historical['day'] = df_historical['date'].dt.day

            similar_dates = df_historical[
                ((df_historical['month'] == pred_month) & 
                 (abs(df_historical['day'] - pred_day) <= 15)) |
                ((pred_month == 1) & (df_historical['month'] == 12) & (df_historical['day'] >= 17)) |
                ((pred_month == 12) & (df_historical['month'] == 1) & (df_historical['day'] <= 15))
            ]

            if len(similar_dates) < 10:
                similar_dates = df_historical[df_historical['month'] == pred_month]
            if len(similar_dates) < 5:
                similar_dates = df_historical

            # Použijeme mediány kde jsou k dispozici, jinak NaN
            def median_or_nan(dfc, col):
                return dfc[col].median() if col in dfc and len(dfc[col].dropna()) > 0 else np.nan

            weather_data = {
                'temperature_max': median_or_nan(similar_dates, 'temperature_max'),
                'temperature_min': median_or_nan(similar_dates, 'temperature_min'),
                'temperature_mean': median_or_nan(similar_dates, 'temperature_mean'),
                'apparent_temp_max': median_or_nan(similar_dates, 'apparent_temp_max'),
                'apparent_temp_min': median_or_nan(similar_dates, 'apparent_temp_min'),
                'apparent_temp_mean': median_or_nan(similar_dates, 'apparent_temp_mean'),
                'precipitation': median_or_nan(similar_dates, 'precipitation'),
                'rain': median_or_nan(similar_dates, 'rain'),
                'snowfall': median_or_nan(similar_dates, 'snowfall'),
                'precipitation_hours': median_or_nan(similar_dates, 'precipitation_hours'),
                'weather_code': (int(similar_dates['weather_code'].mode().iloc[0]) if 'weather_code' in similar_dates and len(similar_dates['weather_code'].mode()) > 0 else np.nan),
                'wind_speed_max': median_or_nan(similar_dates, 'wind_speed_max'),
                'wind_gusts_max': median_or_nan(similar_dates, 'wind_gusts_max'),
                'wind_direction': median_or_nan(similar_dates, 'wind_direction'),
                'sunshine_duration': median_or_nan(similar_dates, 'sunshine_duration'),
                'daylight_duration': median_or_nan(similar_dates, 'daylight_duration'),
                'is_rainy': (int(similar_dates['is_rainy'].mode().iloc[0]) if 'is_rainy' in similar_dates and len(similar_dates['is_rainy'].mode()) > 0 else np.nan),
                'is_snowy': (int(similar_dates['is_snowy'].mode().iloc[0]) if 'is_snowy' in similar_dates and len(similar_dates['is_snowy'].mode()) > 0 else np.nan),
                'is_windy': (int(similar_dates['is_windy'].mode().iloc[0]) if 'is_windy' in similar_dates and len(similar_dates['is_windy'].mode()) > 0 else np.nan),
                'is_nice_weather': (int(similar_dates['is_nice_weather'].mode().iloc[0]) if 'is_nice_weather' in similar_dates and len(similar_dates['is_nice_weather'].mode()) > 0 else np.nan),
                'sunshine_ratio': median_or_nan(similar_dates, 'sunshine_ratio'),
            }
            print(f"   📊 Použito {len(similar_dates)} podobných historických dnů (neexistující hodnoty jsou NaN)")
        else:
            # Pokud nejsou žádná historická data, necháme všechny hodnoty NaN
            weather_cols = ['temperature_max','temperature_min','temperature_mean','apparent_temp_max','apparent_temp_min','apparent_temp_mean','precipitation','rain','snowfall','precipitation_hours','weather_code','wind_speed_max','wind_gusts_max','wind_direction','sunshine_duration','daylight_duration','is_rainy','is_snowy','is_windy','is_nice_weather','sunshine_ratio']
            weather_data = {c: np.nan for c in weather_cols}
            print(f"   ⚠️  Žádná historická data - všechny weather hodnoty jsou NaN")

    
    new_row = pd.DataFrame({
        'date': [date],
        'total_visitors': [np.nan],
        'school_visitors': [np.nan],
        'public_visitors': [np.nan],
        'extra': [None],
        'opening_hours': [None],
        **{k: [v] for k, v in weather_data.items()}
    })
    
    df = pd.concat([df, new_row], ignore_index=True)
    
    # Feature engineering
    df = create_features(df)
    
    # Vybrat poslední řádek (náš prediction date)
    feature_cols = models_dict['feature_cols']
    
    # Najít společné sloupce
    available_features = [col for col in feature_cols if col in df.columns]
    
    pred_row = df[df['date'] == date]
    
    # Pro chybějící features se nejprve pokusíme o medián z historických dat.
    # Pokud pro některé požadované features není historický ani globální medián, přerušíme predikci a upozorníme uživatele (žádné tiché nahrazování 0).
    X_pred = pred_row[available_features].copy()
    missing_features = []
    for col in available_features:
        if X_pred[col].isna().any():
            # Použij mediánovou hodnotu z posledních 90 dní historických dat
            historical_median = df[df['date'] < date][col].tail(90).median()
            if pd.isna(historical_median):
                # Pokud není k dispozici ani historická hodnota, zkus celkový medián
                historical_median = df[col].median()
                if pd.isna(historical_median):
                    # Explicitně zaznamenat chybějící feature (bez tichého nahrazování)
                    missing_features.append(col)
                else:
                    X_pred[col] = X_pred[col].fillna(historical_median)
            else:
                X_pred[col] = X_pred[col].fillna(historical_median)
    
    if missing_features:
        # Namísto nahrazení 0, vyhodíme chybu aby uživatel poskytl historická data nebo upravil feature engineering
        raise ValueError(f"Chybějící nezbytné feature sloupce pro predikci (bez fallbacku): {', '.join(missing_features)}")
    
    # === Predikce z každého modelu ===
    
    # 1. LightGBM
    lgb_model = models_dict['lgb']
    try:
        lgb_pred = lgb_model.predict(X_pred, num_iteration=lgb_model.best_iteration)[0]
    except:
        lgb_pred = lgb_model.predict(X_pred)[0]
    
    # 2. XGBoost
    xgb_model = models_dict['xgb']
    dmatrix = xgb.DMatrix(X_pred)
    xgb_pred = xgb_model.predict(dmatrix)[0]
    
    # 3. CatBoost
    cat_model = models_dict['cat']
    cat_pred = cat_model.predict(X_pred)[0]
    
    # === Ensemble ===
    weights = models_dict['weights']
    ensemble_pred = (
        weights[0] * lgb_pred +
        weights[1] * xgb_pred +
        weights[2] * cat_pred
    )
    
    # Zaokrouhlit na celé číslo
    ensemble_pred = int(round(max(ensemble_pred, 0)))
    
    # Confidence interval (aproximace z variance modelů)
    model_std = np.std([lgb_pred, xgb_pred, cat_pred])
    confidence_lower = int(max(0, ensemble_pred - 1.96 * model_std))
    confidence_upper = int(ensemble_pred + 1.96 * model_std)
    
    result = {
        'date': date,
        'day_of_week': date.strftime('%A'),
        'ensemble_prediction': ensemble_pred,
        'confidence_interval': (confidence_lower, confidence_upper),
        'individual_predictions': {
            'lightgbm': int(round(lgb_pred)),
            'xgboost': int(round(xgb_pred)),
            'catboost': int(round(cat_pred))
        },
        'model_weights': {
            'lightgbm': float(weights[0]),
            'xgboost': float(weights[1]),
            'catboost': float(weights[2])
        },
        'weather': {
            'description': weather_description or 'N/A',
            'temperature': weather_data.get('temperature_mean', 10.0),
            'precipitation': weather_data.get('precipitation', 0.0),
            'rain': weather_data.get('rain', 0.0),
            'snowfall': weather_data.get('snowfall', 0.0),
        }
    }
    
    return result


def predict_date_range(start_date, end_date, models_dict):
    """
    Predikuje návštěvnost pro rozsah dat
    
    Args:
        start_date: Začátek období
        end_date: Konec období
        models_dict: Dict s natrénovanými modely
        
    Returns:
        DataFrame s predikcemi
    """
    # Načíst historická data S POČASÍM (stejně jako v predict_single_date)
    script_dir = Path(__file__).parent
    data_path = script_dir.parent / 'data' / 'processed' / 'techmania_with_weather.csv'
    
    if not data_path.exists():
        print("⚠️ techmania_with_weather.csv nenalezen, použiji data bez počasí")
        data_path = script_dir.parent / 'data' / 'raw' / 'techmania_cleaned_master.csv'
    
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])

    
    # Vytvořit rozsah dat
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    print(f"\n🔮 Predicting {len(date_range)} days...")
    
    results = []
    for date in date_range:
        try:
            pred = predict_single_date(date, models_dict, df)
            results.append({
                'date': pred['date'],
                'day_of_week': pred['day_of_week'],
                'prediction': pred['ensemble_prediction'],
                'lower_bound': pred['confidence_interval'][0],
                'upper_bound': pred['confidence_interval'][1],
                'lightgbm': pred['individual_predictions']['lightgbm'],
                'xgboost': pred['individual_predictions']['xgboost'],
                'catboost': pred['individual_predictions']['catboost']
            })
        except Exception as e:
            print(f"  ⚠️ Error predicting {date}: {e}")
    
    results_df = pd.DataFrame(results)
    print(f"✅ Predicted {len(results_df)} days successfully!")
    
    return results_df


def print_prediction(result):
    """
    Pěkně vypíše výsledek predikce
    
    Args:
        result: Dict s predikcí
    """
    print("\n" + "=" * 60)
    print(f"🔮 PREDIKCE PRO {result['date'].strftime('%d.%m.%Y')} ({result['day_of_week']})")
    print("=" * 60)
    
    print(f"\n🎯 ENSEMBLE PREDIKCE: {result['ensemble_prediction']} návštěvníků")
    print(f"   95% Confidence Interval: [{result['confidence_interval'][0]} - {result['confidence_interval'][1]}]")
    
    print(f"\n📊 Jednotlivé modely:")
    print(f"   LightGBM (váha {result['model_weights']['lightgbm']:.1%}): {result['individual_predictions']['lightgbm']} návštěvníků")
    print(f"   XGBoost (váha {result['model_weights']['xgboost']:.1%}): {result['individual_predictions']['xgboost']} návštěvníků")
    print(f"   CatBoost (váha {result['model_weights']['catboost']:.1%}): {result['individual_predictions']['catboost']} návštěvníků")
    
    print("=" * 60)


def main():
    """
    Demo použití predikčního modulu
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
