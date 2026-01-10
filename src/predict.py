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
    print("❌ CHYBA: Weather/Holiday services nejsou dostupné!")
    print("   Predikce nelze provést bez reálných dat o počasí.")
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
    if not SERVICES_AVAILABLE:
        raise RuntimeError("Weather services not available. Cannot make prediction without real weather data.")
    
    if SERVICES_AVAILABLE:
        try:
            pred_date = date.date() if isinstance(date, pd.Timestamp) else date
            weather_info = weather_service.get_weather(pred_date)
            
            weather_description = weather_info.get('weather_description', 'N/A')
            
            # Základní hodnoty z API 
            weather_data = {
                'temperature_max': weather_info['temperature_max'],
                'temperature_min': weather_info['temperature_min'],
                'temperature_mean': weather_info['temperature_mean'],
                'precipitation': weather_info['precipitation'],
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
            # Pokud historická data nejsou dostupná, nastavíme value na np.nan 
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
            print(f"   ❌ Weather API error: {e}")
            raise RuntimeError(f"Failed to get weather data for {pred_date}: {e}. Cannot make prediction without real weather data.")
    
    # Ověřit, že máme základní weather data
    if not weather_data or 'temperature_mean' not in weather_data:
        raise RuntimeError(f"Weather data not available for {pred_date}. Cannot make prediction without real weather data.")
    
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
            'temperature': weather_data['temperature_mean'],
            'precipitation': weather_data['precipitation'],
            'rain': weather_data.get('rain', weather_data['precipitation']),
            'snowfall': weather_data.get('snowfall', 0.0) if 'snowfall' in weather_data else 0.0,
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
    if not SERVICES_AVAILABLE:
        raise RuntimeError("Weather services not available. Cannot make predictions without real weather data.")
    
    # Načíst historická data S POČASÍM
    script_dir = Path(__file__).parent
    data_path = script_dir.parent / 'data' / 'processed' / 'techmania_with_weather.csv'
    
    if not data_path.exists():
        print("⚠️ techmania_with_weather.csv nenalezen, použiji data bez počasí")
        data_path = script_dir.parent / 'data' / 'raw' / 'techmania_cleaned_master.csv'
    
    df_historical = pd.read_csv(data_path)
    df_historical['date'] = pd.to_datetime(df_historical['date'])
    
    # Vytvořit rozsah dat
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    print(f"\n🔮 Predicting {len(date_range)} days...")
    print(f"📥 Downloading weather data for {len(date_range)} days...")
    
    # Stáhnout weather data pro všechny dny
    new_rows = []
    for date in date_range:
        try:
            pred_date = date.date() if isinstance(date, pd.Timestamp) else date
            weather_info = weather_service.get_weather(pred_date)
            
            # Základní weather data z API
            weather_data = {
                'date': date,
                'total_visitors': np.nan,
                'school_visitors': np.nan,
                'public_visitors': np.nan,
                'extra': None,
                'opening_hours': None,
                'temperature_max': weather_info['temperature_max'],
                'temperature_min': weather_info['temperature_min'],
                'temperature_mean': weather_info['temperature_mean'],
                'precipitation': weather_info['precipitation'],
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
            
            # Features z historických dat
            pred_month = date.month
            pred_day = date.day
            
            df_hist = df_historical[df_historical['date'] < date].copy()
            if len(df_hist) > 0:
                df_hist['month'] = df_hist['date'].dt.month
                df_hist['day'] = df_hist['date'].dt.day
                
                # Najít podobné dny
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
                
                weather_data['apparent_temp_max'] = similar['apparent_temp_max'].median() if 'apparent_temp_max' in similar and len(similar) > 0 else np.nan
                weather_data['apparent_temp_min'] = similar['apparent_temp_min'].median() if 'apparent_temp_min' in similar and len(similar) > 0 else np.nan
                weather_data['apparent_temp_mean'] = similar['apparent_temp_mean'].median() if 'apparent_temp_mean' in similar and len(similar) > 0 else np.nan
                weather_data['wind_direction'] = similar['wind_direction'].median() if 'wind_direction' in similar and len(similar) > 0 else np.nan
                weather_data['sunshine_duration'] = similar['sunshine_duration'].median() if 'sunshine_duration' in similar and len(similar) > 0 else np.nan
                weather_data['daylight_duration'] = similar['daylight_duration'].median() if 'daylight_duration' in similar and len(similar) > 0 else np.nan
                weather_data['sunshine_ratio'] = similar['sunshine_ratio'].median() if 'sunshine_ratio' in similar and len(similar) > 0 else np.nan
            else:
                weather_data['apparent_temp_max'] = np.nan
                weather_data['apparent_temp_min'] = np.nan
                weather_data['apparent_temp_mean'] = np.nan
                weather_data['wind_direction'] = np.nan
                weather_data['sunshine_duration'] = np.nan
                weather_data['daylight_duration'] = np.nan
                weather_data['sunshine_ratio'] = np.nan
            
            new_rows.append(weather_data)
            
        except Exception as e:
            print(f"  ⚠️ Error getting weather for {date.strftime('%Y-%m-%d')}: {e}")
            raise RuntimeError(f"Failed to get weather data for {date.strftime('%Y-%m-%d')}: {e}")
    
    # Vytvořit DataFrame s novými daty
    df_new = pd.DataFrame(new_rows)
    
    # Spojit historická data s novými daty
    df_combined = pd.concat([df_historical, df_new], ignore_index=True)
    df_combined = df_combined.sort_values('date').reset_index(drop=True)
    
    print(f"✅ Weather data downloaded for {len(df_new)} days")
    
    # Feature engineering na celý dataset
    df_combined = create_features(df_combined)
    
    # Vybrat jen řádky pro predikci
    df_pred = df_combined[df_combined['date'].isin(date_range)].copy()
    
    # Připravit features
    feature_cols = models_dict['feature_cols']
    available_features = [col for col in feature_cols if col in df_pred.columns]
    
    # Doplnit chybějící features mediánem z historických dat
    X_pred = df_pred[available_features].copy()
    for col in available_features:
        if X_pred[col].isna().any():
            historical_median = df_combined[df_combined['date'] < start_date][col].median()
            if pd.isna(historical_median):
                historical_median = df_combined[col].median()
            if not pd.isna(historical_median):
                X_pred[col] = X_pred[col].fillna(historical_median)
    
    # Kontrola chybějících hodnot
    missing_cols = X_pred.columns[X_pred.isna().any()].tolist()
    if missing_cols:
        print(f"  ⚠️ Warning: Chybějící hodnoty v sloupcích: {missing_cols}")
        # Doplnit 0 jako poslední možnost
        X_pred = X_pred.fillna(0)
    
    # === Predikce z každého modelu ===
    print(f"🤖 Running predictions...")
    
    # LightGBM
    lgb_model = models_dict['lgb']
    try:
        lgb_preds = lgb_model.predict(X_pred, num_iteration=lgb_model.best_iteration)
    except:
        lgb_preds = lgb_model.predict(X_pred)
    
    # XGBoost
    xgb_model = models_dict['xgb']
    dmatrix = xgb.DMatrix(X_pred)
    xgb_preds = xgb_model.predict(dmatrix)
    
    # CatBoost
    cat_model = models_dict['cat']
    cat_preds = cat_model.predict(X_pred)
    
    # === Ensemble ===
    weights = models_dict['weights']
    ensemble_preds = (
        weights[0] * lgb_preds +
        weights[1] * xgb_preds +
        weights[2] * cat_preds
    )
    
    # Sestavit výsledky
    results = []
    for i, date in enumerate(df_pred['date']):
        model_std = np.std([lgb_preds[i], xgb_preds[i], cat_preds[i]])
        ensemble_pred = int(round(max(ensemble_preds[i], 0)))
        
        results.append({
            'date': date,
            'day_of_week': date.strftime('%A'),
            'prediction': ensemble_pred,
            'lower_bound': int(max(0, ensemble_pred - 1.96 * model_std)),
            'upper_bound': int(ensemble_pred + 1.96 * model_std),
            'lightgbm': int(round(lgb_preds[i])),
            'xgboost': int(round(xgb_preds[i])),
            'catboost': int(round(cat_preds[i]))
        })
    
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
