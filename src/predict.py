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

# Použít NOVÝ feature engineering v3!
from feature_engineering_v3 import create_features
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
        
        # Načíst V3 modely (s event features, Optuna tuning, atd.)
        models = {
            'lgb': joblib.load(os.path.join(models_dir, 'lightgbm_v3.pkl')),
            'xgb': joblib.load(os.path.join(models_dir, 'xgboost_v3.pkl')),
            'cat': joblib.load(os.path.join(models_dir, 'catboost_v3.pkl')),
            'weights': joblib.load(os.path.join(models_dir, 'ensemble_weights_v3.pkl')),
            'feature_cols': joblib.load(os.path.join(models_dir, 'feature_names_v3.pkl')),
            'google_trend_predictor': joblib.load(os.path.join(models_dir, 'google_trend_predictor_v3.pkl')),
            'historical_mae': joblib.load(os.path.join(models_dir, 'historical_mae_v3.pkl'))
        }
        
        # V3 používá vždy weighted ensemble
        models['ensemble_type'] = 'weighted'
        print(f"✅ Models V3 loaded successfully! (Ensemble: WEIGHTED - 3 models)")
        print(f"   Historical MAE - Weekday: {models['historical_mae']['weekday']:.2f}, Weekend: {models['historical_mae']['weekend']:.2f}")
        
        return models
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("   Please train the V3 models first by running: python src/ensemble_model_v3.py")
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
    # Načíst historická data S POČASÍM A SVÁTKY (potřebujeme pro správné features)
    if historical_df is None:
        script_dir = Path(__file__).parent
        
        # 1. Načíst historická data (do 2025)
        data_path = script_dir.parent / 'data' / 'processed' / 'techmania_with_weather_and_holidays.csv'
        
        if not data_path.exists():
            print("⚠️ techmania_with_weather_and_holidays.csv nenalezen, zkouším bez holidays")
            data_path = script_dir.parent / 'data' / 'processed' / 'techmania_with_weather.csv'
            
            if not data_path.exists():
                print("⚠️ techmania_with_weather.csv nenalezen, použiji data bez počasí")
                data_path = script_dir.parent / 'data' / 'raw' / 'techmania_cleaned_master.csv'
        
        try:
            df = pd.read_csv(data_path)
            df['date'] = pd.to_datetime(df['date'])
            print(f"   📂 Loaded historical data: {len(df)} rows (up to {df['date'].max().date()})")
        except FileNotFoundError as e:
            print(f"⚠️ Nepodařilo se načíst historická data: {e}")
            # Vytvoříme prázdný DataFrame s potřebnými sloupci
            df = pd.DataFrame(columns=['date', 'total_visitors'])
            df['date'] = pd.to_datetime(df['date'])
        
        # 2. Načíst template pro 2026 (s předvyplněnými holiday features)
        template_2026_path = script_dir.parent / 'data' / 'raw' / 'techmania_2026_template.csv'
        if template_2026_path.exists():
            df_2026 = pd.read_csv(template_2026_path)
            df_2026['date'] = pd.to_datetime(df_2026['date'])
            
            # Spojit s historickými daty (pokud už tam nejsou data z 2026)
            max_historical_date = df['date'].max()
            df_2026_filtered = df_2026[df_2026['date'] > max_historical_date]
            
            if len(df_2026_filtered) > 0:
                df = pd.concat([df, df_2026_filtered], ignore_index=True)
                print(f"   📂 Added 2026 template: {len(df_2026_filtered)} rows (holiday features pre-filled)")
        else:
            print(f"   ⚠️ 2026 template not found at {template_2026_path}")
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
            
            # Okamžitě přidat precipitation_probability (odhad na základě srážek)
            # Toto je feature z trénovacího CSV, který musí být vždy přítomen
            precip = weather_data.get('precipitation', 0)
            if precip > 5:
                weather_data['precipitation_probability'] = 90
            elif precip > 1:
                weather_data['precipitation_probability'] = 70
            elif precip > 0:
                weather_data['precipitation_probability'] = 50
            else:
                weather_data['precipitation_probability'] = 20
            
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
                
                # Precipitation probability - pokud jsou historická data, upřesníme z nich
                if 'precipitation_probability' in similar.columns and len(similar) > 0:
                    hist_precip_prob = similar['precipitation_probability'].median()
                    if pd.notna(hist_precip_prob):
                        # Průměr mezi odhadem a historií (kompromis)
                        weather_data['precipitation_probability'] = (
                            weather_data['precipitation_probability'] + hist_precip_prob
                        ) / 2
                
                # Další odvozené weather features z trénovacího CSV
                if 'cloud_cover_percent' in similar and len(similar) > 0:
                    weather_data['cloud_cover_percent'] = similar['cloud_cover_percent'].median()
                else:
                    weather_data['cloud_cover_percent'] = np.nan
                    
                if 'feels_like_delta' in similar and len(similar) > 0:
                    weather_data['feels_like_delta'] = similar['feels_like_delta'].median()
                else:
                    weather_data['feels_like_delta'] = np.nan
                    
                if 'weather_forecast_confidence' in similar and len(similar) > 0:
                    weather_data['weather_forecast_confidence'] = similar['weather_forecast_confidence'].median()
                else:
                    weather_data['weather_forecast_confidence'] = np.nan
                    
                if 'temperature_trend_3d' in similar and len(similar) > 0:
                    weather_data['temperature_trend_3d'] = similar['temperature_trend_3d'].median()
                else:
                    weather_data['temperature_trend_3d'] = np.nan
                    
                if 'is_weather_improving' in similar and len(similar) > 0:
                    weather_data['is_weather_improving'] = int(similar['is_weather_improving'].median())
                else:
                    weather_data['is_weather_improving'] = 0
            else:
                # Pokud nejsou historická data, žádné tiché výchozí hodnoty - použijeme NaN
                weather_data['apparent_temp_max'] = np.nan
                weather_data['apparent_temp_min'] = np.nan
                weather_data['apparent_temp_mean'] = np.nan
                weather_data['wind_direction'] = np.nan
                weather_data['sunshine_duration'] = np.nan
                weather_data['daylight_duration'] = np.nan
                weather_data['sunshine_ratio'] = np.nan
                
                # precipitation_probability už je nastavena výše (odhad z aktuálních srážek)
                # Nemusíme ji zde duplikovat
                    
                weather_data['cloud_cover_percent'] = np.nan
                weather_data['feels_like_delta'] = np.nan
                weather_data['weather_forecast_confidence'] = np.nan
                weather_data['temperature_trend_3d'] = np.nan
                weather_data['is_weather_improving'] = 0

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
    
    # Získat holiday/school break data pro predikované datum
    holiday_data = {}
    if SERVICES_AVAILABLE:
        try:
            pred_date = date.date() if isinstance(date, pd.Timestamp) else date
            holiday_info = holiday_service.get_holiday_info(pred_date)
            
            # Základní holiday features z API
            holiday_data = {
                'is_any_school_break': int(holiday_info.get('is_any_school_break', False)),
                'school_break_type': holiday_info.get('school_break_type', None),
                'is_spring_break': int(holiday_info.get('is_spring_break', False)),
                'is_autumn_break': int(holiday_info.get('is_autumn_break', False)),
                'is_winter_break': int(holiday_info.get('is_winter_break', False)),
                'is_easter_break': int(holiday_info.get('is_easter_break', False)),
                'is_halfyear_break': int(holiday_info.get('is_halfyear_break', False)),
                'is_summer_holiday': int(holiday_info.get('is_summer_holiday', False)),
                'days_to_next_break': holiday_info.get('days_to_next_break', 0),
                'days_from_last_break': holiday_info.get('days_from_last_break', 0),
                'is_week_before_break': int(holiday_info.get('is_week_before_break', False)),
                'is_week_after_break': int(holiday_info.get('is_week_after_break', False)),
                'season_exact': holiday_info.get('season_exact', None),
                'week_position': holiday_info.get('week_position', None),
                'is_month_end': int(holiday_info.get('is_month_end', False)),
                'school_week_number': holiday_info.get('school_week_number', 0),
                'is_bridge_day': int(holiday_info.get('is_bridge_day', False)),
                'long_weekend_length': holiday_info.get('long_weekend_length', 0),
            }
        except Exception as e:
            print(f"   ⚠️ Holiday API warning: {e}")
            # Pokud holiday API selže, ponecháme prázdné hodnoty (NaN se doplní později)
            holiday_data = {}
    
    # OPRAVENO: Zkontrolovat, zda datum již existuje v DataFrame (template)
    date_exists = (df['date'] == pd.to_datetime(date)).any()
    
    if date_exists:
        # Datum existuje v template - aktualizovat pouze weather data
        print(f"   ✓ Using template data for {date}")
        date_idx = df[df['date'] == pd.to_datetime(date)].index[0]
        # Aktualizovat weather data v existujícím řádku
        for k, v in weather_data.items():
            if k in df.columns:
                df.loc[date_idx, k] = v
        # Aktualizovat holiday data (přepsat template hodnoty novými z API)
        for k, v in holiday_data.items():
            if k in df.columns:
                df.loc[date_idx, k] = v
    else:
        # Datum neexistuje - vytvořit nový řádek
        print(f"   ⚠️ Date {date} not in template, creating new row")
        new_row = pd.DataFrame({
            'date': [date],
            'total_visitors': [np.nan],
            'school_visitors': [np.nan],
            'public_visitors': [np.nan],
            'extra': [None],
            'opening_hours': [None],
            **{k: [v] for k, v in weather_data.items()},
            **{k: [v] for k, v in holiday_data.items()}
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
    
    # === KONVERZE NA NUMERIC (fix pro V3 features) ===
    # V3 vytváří některé features jako object, musíme je převést na numeric
    for col in X_pred.columns:
        if X_pred[col].dtype == 'object':
            X_pred[col] = pd.to_numeric(X_pred[col], errors='coerce')
    
    # Fill any remaining NaN after conversion
    X_pred = X_pred.fillna(0)
    
    # === GOOGLE TREND PREDICTION (Improvement #5) ===
    # Predikovat Google Trend zvlášť a přidat jako feature
    google_trend_predictor = models_dict.get('google_trend_predictor')
    if google_trend_predictor is not None:
        # Google Trend predictor používá pouze časové/cyklické features (dle ensemble_model_v3.py řádek 70-75)
        trend_features = [
            'year', 'month', 'day_of_week', 'week_of_year', 'quarter',
            'is_weekend', 'is_summer_holiday', 'is_winter_holiday', 'is_school_year',
            'is_oct_28', 'is_autumn_break', 'is_summer_weekend_event', 'event_score',
            'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos',
            'week_sin', 'week_cos', 'normalized_time'
        ]
        
        # DŮLEŽITÉ: Tyto features nejsou v feature_cols (byly odstraněny při trénování),
        # takže je musíme vzít přímo z df (před filtrováním)
        pred_row_full = df[df['date'] == date]
        available_trend_features = [f for f in trend_features if f in pred_row_full.columns]
        
        X_trend = pred_row_full[available_trend_features]
        # Konvertovat na numeric (může obsahovat object types)
        for col in X_trend.columns:
            if X_trend[col].dtype == 'object':
                X_trend[col] = pd.to_numeric(X_trend[col], errors='coerce')
        X_trend = X_trend.fillna(0)
        
        predicted_trend = google_trend_predictor.predict(X_trend.values)[0]
        X_pred['predicted_google_trend'] = predicted_trend
    else:
        # Fallback pokud predictor není dostupný
        print(f"   Fallback: Google Trend predictor not available, using default value 50.0")
        X_pred['predicted_google_trend'] = X_pred.get('google_trend', 50.0)
    
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
    
    # DEBUG: Vypsat prvních 10 features pro debugging
    debug_features = X_pred.iloc[0].head(15).to_dict()
    print(f"   DEBUG: First 15 features for CatBoost:")
    for k, v in debug_features.items():
        print(f"      {k}: {v}")
    
    cat_pred = cat_model.predict(X_pred)[0]
    print(f"   DEBUG: CatBoost raw prediction: {cat_pred}")
    
    # === Ensemble - podle typu ===
    ensemble_type = models_dict.get('ensemble_type', 'weighted')
    
    # IMPROVEMENT: Zjistit, zda je víkend nebo svátek (CatBoost funguje lépe na tyto dny)
    is_weekend = X_pred['is_weekend'].iloc[0] == 1
    is_holiday = X_pred['is_holiday'].iloc[0] == 1
    use_catboost = is_weekend or is_holiday
    
    if ensemble_type == 'single_lgb':
        # SINGLE: Použít pouze LightGBM (ensemble nepomohl)
        ensemble_pred = lgb_pred
    elif ensemble_type == 'stacking':
        # STACKING: Použít meta-model
        meta_model = models_dict['meta_model']
        meta_features = np.array([[lgb_pred, xgb_pred, cat_pred]])
        ensemble_pred = meta_model.predict(meta_features)[0]
    else:
        # WEIGHTED: Použít váhy (V3 má dict s klíči 'LightGBM', 'XGBoost', 'CatBoost')
        weights = models_dict['weights']
        
        if isinstance(weights, dict):
            # V3 format: dict s klíči modelů
            
            # IMPROVEMENT: CatBoost jen pro víkendy/svátky (systematicky přestřeluje všední dny)
            if use_catboost:
                # Víkend/svátek: použít všechny 3 modely
                ensemble_pred = (
                    weights.get('LightGBM', 0.33) * lgb_pred +
                    weights.get('XGBoost', 0.33) * xgb_pred +
                    weights.get('CatBoost', 0.34) * cat_pred
                )
            else:
                # Všední den: jen LightGBM + XGBoost (přenormalizovat váhy)
                w_lgb = weights.get('LightGBM', 0.5)
                w_xgb = weights.get('XGBoost', 0.5)
                w_sum = w_lgb + w_xgb
                
                if w_sum > 0:
                    ensemble_pred = (
                        (w_lgb / w_sum) * lgb_pred +
                        (w_xgb / w_sum) * xgb_pred
                    )
                else:
                    # Fallback: rovnoměrné váhy
                    print("USING fallback for weight for CatBoost")
                    ensemble_pred = 0.5 * lgb_pred + 0.5 * xgb_pred
        else:
            # Starý format: list/array
            if use_catboost:
                ensemble_pred = (
                    weights[0] * lgb_pred +
                    weights[1] * xgb_pred +
                    weights[2] * cat_pred
                )
            else:
                # Jen LightGBM + XGBoost
                w_sum = weights[0] + weights[1]
                ensemble_pred = (
                    (weights[0] / w_sum) * lgb_pred +
                    (weights[1] / w_sum) * xgb_pred
                )
    
    # Zaokrouhlit na celé číslo
    ensemble_pred = int(round(max(ensemble_pred, 0)))
    
    # Confidence interval z historických reziduí (IMPROVEMENT: realističtější než variance modelů)
    historical_mae = models_dict.get('historical_mae', None)
    
    if historical_mae is not None:
        # Zjistit, zda je víkend nebo všední den
        is_weekend = X_pred['is_weekend'].iloc[0] == 1
        is_holiday = X_pred['is_holiday'].iloc[0] == 1
        
        # Použít odpovídající MAE (víkendy + svátky mají jiné chování)
        if is_weekend or is_holiday:
            mae = historical_mae['weekend']
        else:
            mae = historical_mae['weekday']
        
        # CI = predikce ± 1.96 * MAE (95% confidence interval)
        # Dolní mez minimálně 50 (Techmania nikdy nebude úplně prázdná)
        confidence_lower = int(max(50, ensemble_pred - 1.96 * mae))
        confidence_upper = int(ensemble_pred + 1.96 * mae)
    else:
        # Fallback na starou metodu (variance modelů) pokud historical_mae není dostupná
        model_std = np.std([lgb_pred, xgb_pred, cat_pred])
        confidence_lower = int(max(50, ensemble_pred - 1.96 * model_std))
        confidence_upper = int(ensemble_pred + 1.96 * model_std)
    
    # Vypočítat efektivní váhy (s přihlédnutím k selektivnímu použití CatBoost)
    if isinstance(weights, dict):
        if use_catboost:
            effective_weights = {
                'lightgbm': float(weights.get('LightGBM', 0.33)),
                'xgboost': float(weights.get('XGBoost', 0.33)),
                'catboost': float(weights.get('CatBoost', 0.34))
            }
        else:
            # CatBoost nepoužit, přenormalizovat
            w_lgb = weights.get('LightGBM', 0.5)
            w_xgb = weights.get('XGBoost', 0.5)
            w_sum = w_lgb + w_xgb
            if w_sum > 0:
                effective_weights = {
                    'lightgbm': float(w_lgb / w_sum),
                    'xgboost': float(w_xgb / w_sum),
                    'catboost': 0.0
                }
            else:
                effective_weights = {'lightgbm': 0.5, 'xgboost': 0.5, 'catboost': 0.0}
    else:
        effective_weights = {
            'lightgbm': float(weights[0] if isinstance(weights, (list, np.ndarray)) else 0.33),
            'xgboost': float(weights[1] if isinstance(weights, (list, np.ndarray)) else 0.33),
            'catboost': float(weights[2] if isinstance(weights, (list, np.ndarray)) else 0.34)
        }
    
    result = {
        'date': date,
        'day_of_week': date.strftime('%A'),
        'ensemble_prediction': ensemble_pred,
        'ensemble_type': ensemble_type,
        'confidence_interval': (confidence_lower, confidence_upper),
        'catboost_used': use_catboost,  # IMPROVEMENT: Info o použití CatBoost
        'individual_predictions': {
            'lightgbm': int(round(lgb_pred)),
            'xgboost': int(round(xgb_pred)),
            'catboost': int(round(cat_pred))
        },
        'model_weights': effective_weights,
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
    
    # Načíst historická data S POČASÍM A SVÁTKY
    script_dir = Path(__file__).parent
    
    # 1. Načíst historická data (do 2025)
    data_path = script_dir.parent / 'data' / 'processed' / 'techmania_with_weather_and_holidays.csv'
    
    if not data_path.exists():
        print("⚠️ techmania_with_weather_and_holidays.csv nenalezen, zkouším bez holidays")
        data_path = script_dir.parent / 'data' / 'processed' / 'techmania_with_weather.csv'
        
        if not data_path.exists():
            print("⚠️ techmania_with_weather.csv nenalezen, použiji data bez počasí")
            data_path = script_dir.parent / 'data' / 'raw' / 'techmania_cleaned_master.csv'
    
    df_historical = pd.read_csv(data_path)
    df_historical['date'] = pd.to_datetime(df_historical['date'])
    print(f"   📂 Loaded historical data: {len(df_historical)} rows")
    
    # 2. Načíst template pro 2026 (s předvyplněnými holiday features)
    template_2026_path = script_dir.parent / 'data' / 'raw' / 'techmania_2026_template.csv'
    if template_2026_path.exists():
        df_2026 = pd.read_csv(template_2026_path)
        df_2026['date'] = pd.to_datetime(df_2026['date'])
        
        # Spojit s historickými daty (pokud už tam nejsou data z 2026)
        max_historical_date = df_historical['date'].max()
        df_2026_filtered = df_2026[df_2026['date'] > max_historical_date]
        
        if len(df_2026_filtered) > 0:
            df_historical = pd.concat([df_historical, df_2026_filtered], ignore_index=True)
            print(f"   📂 Added 2026 template: {len(df_2026_filtered)} rows (holiday features pre-filled)")
    else:
        print(f"   ⚠️ 2026 template not found at {template_2026_path}")
    
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
            
            # Přidat precipitation_probability (odhad z aktuálních srážek)
            precip = weather_data.get('precipitation', 0)
            if precip > 5:
                weather_data['precipitation_probability'] = 90
            elif precip > 1:
                weather_data['precipitation_probability'] = 70
            elif precip > 0:
                weather_data['precipitation_probability'] = 50
            else:
                weather_data['precipitation_probability'] = 20
            
            # Získat holiday data
            try:
                holiday_info = holiday_service.get_holiday_info(pred_date)
                weather_data.update({
                    'is_any_school_break': int(holiday_info.get('is_any_school_break', False)),
                    'school_break_type': holiday_info.get('school_break_type', None),
                    'is_spring_break': int(holiday_info.get('is_spring_break', False)),
                    'is_autumn_break': int(holiday_info.get('is_autumn_break', False)),
                    'is_winter_break': int(holiday_info.get('is_winter_break', False)),
                    'is_easter_break': int(holiday_info.get('is_easter_break', False)),
                    'is_halfyear_break': int(holiday_info.get('is_halfyear_break', False)),
                    'is_summer_holiday': int(holiday_info.get('is_summer_holiday', False)),
                    'days_to_next_break': holiday_info.get('days_to_next_break', 0),
                    'days_from_last_break': holiday_info.get('days_from_last_break', 0),
                    'is_week_before_break': int(holiday_info.get('is_week_before_break', False)),
                    'is_week_after_break': int(holiday_info.get('is_week_after_break', False)),
                    'season_exact': holiday_info.get('season_exact', None),
                    'week_position': holiday_info.get('week_position', None),
                    'is_month_end': int(holiday_info.get('is_month_end', False)),
                    'school_week_number': holiday_info.get('school_week_number', 0),
                    'is_bridge_day': int(holiday_info.get('is_bridge_day', False)),
                    'long_weekend_length': holiday_info.get('long_weekend_length', 0),
                })
            except Exception as e:
                print(f"  ⚠️ Holiday API warning for {pred_date}: {e}")
            
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
                
                weather_data['apparent_temp_max'] = similar['apparent_temp_max'].median() if 'apparent_temp_max' in similar.columns and len(similar) > 0 else np.nan
                weather_data['apparent_temp_min'] = similar['apparent_temp_min'].median() if 'apparent_temp_min' in similar.columns and len(similar) > 0 else np.nan
                weather_data['apparent_temp_mean'] = similar['apparent_temp_mean'].median() if 'apparent_temp_mean' in similar.columns and len(similar) > 0 else np.nan
                weather_data['wind_direction'] = similar['wind_direction'].median() if 'wind_direction' in similar.columns and len(similar) > 0 else np.nan
                weather_data['sunshine_duration'] = similar['sunshine_duration'].median() if 'sunshine_duration' in similar.columns and len(similar) > 0 else np.nan
                weather_data['daylight_duration'] = similar['daylight_duration'].median() if 'daylight_duration' in similar.columns and len(similar) > 0 else np.nan
                weather_data['sunshine_ratio'] = similar['sunshine_ratio'].median() if 'sunshine_ratio' in similar.columns and len(similar) > 0 else np.nan
                
                # Další odvozené features
                weather_data['cloud_cover_percent'] = similar['cloud_cover_percent'].median() if 'cloud_cover_percent' in similar.columns and len(similar) > 0 else np.nan
                weather_data['feels_like_delta'] = similar['feels_like_delta'].median() if 'feels_like_delta' in similar.columns and len(similar) > 0 else np.nan
                weather_data['weather_forecast_confidence'] = similar['weather_forecast_confidence'].median() if 'weather_forecast_confidence' in similar.columns and len(similar) > 0 else np.nan
                weather_data['temperature_trend_3d'] = similar['temperature_trend_3d'].median() if 'temperature_trend_3d' in similar.columns and len(similar) > 0 else np.nan
                weather_data['is_weather_improving'] = int(similar['is_weather_improving'].median()) if 'is_weather_improving' in similar.columns and len(similar) > 0 else 0
            else:
                weather_data['apparent_temp_max'] = np.nan
                weather_data['apparent_temp_min'] = np.nan
                weather_data['apparent_temp_mean'] = np.nan
                weather_data['wind_direction'] = np.nan
                weather_data['sunshine_duration'] = np.nan
                weather_data['daylight_duration'] = np.nan
                weather_data['sunshine_ratio'] = np.nan
                weather_data['cloud_cover_percent'] = np.nan
                weather_data['feels_like_delta'] = np.nan
                weather_data['weather_forecast_confidence'] = np.nan
                weather_data['temperature_trend_3d'] = np.nan
                weather_data['is_weather_improving'] = 0
            
            new_rows.append(weather_data)
            
        except Exception as e:
            print(f"  ⚠️ Error getting weather for {date.strftime('%Y-%m-%d')}: {e}")
            raise RuntimeError(f"Failed to get weather data for {date.strftime('%Y-%m-%d')}: {e}")
    
    # Vytvořit DataFrame s novými daty
    df_new = pd.DataFrame(new_rows)
    
    # Spojit historická data s novými daty - odstranit duplicity
    df_combined = pd.concat([df_historical, df_new], ignore_index=True)
    df_combined = df_combined.drop_duplicates(subset=['date'], keep='last')  # Preferovat nová weather data
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
    
    # Konvertovat start_date na pandas Timestamp pro správné porovnání
    start_date_ts = pd.to_datetime(start_date)
    
    for col in available_features:
        if X_pred[col].isna().any():
            historical_median = df_combined[df_combined['date'] < start_date_ts][col].median()
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
    
    # === Ensemble - podle typu ===
    ensemble_type = models_dict.get('ensemble_type', 'weighted')
    
    if ensemble_type == 'single_lgb':
        # SINGLE: Použít pouze LightGBM
        ensemble_preds = lgb_preds
    elif ensemble_type == 'stacking':
        # STACKING: Použít meta-model
        meta_model = models_dict['meta_model']
        meta_features = np.column_stack([lgb_preds, xgb_preds, cat_preds])
        ensemble_preds = meta_model.predict(meta_features)
    else:
        # WEIGHTED: Použít váhy (V3 má dict, starší verze list)
        weights = models_dict['weights']
        if isinstance(weights, dict):
            # V3 format
            ensemble_preds = (
                weights.get('LightGBM', 0.33) * lgb_preds +
                weights.get('XGBoost', 0.33) * xgb_preds +
                weights.get('CatBoost', 0.34) * cat_preds
            )
        else:
            # Starý format
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
    
    # Info o použití CatBoost
    catboost_status = "ACTIVE" if result.get('catboost_used', True) else "DISABLED (weekday)"
    
    print(f"\n📊 Jednotlivé modely:")
    print(f"   LightGBM (váha {result['model_weights']['lightgbm']:.1%}): {result['individual_predictions']['lightgbm']} návštěvníků")
    print(f"   XGBoost (váha {result['model_weights']['xgboost']:.1%}): {result['individual_predictions']['xgboost']} návštěvníků")
    print(f"   CatBoost (váha {result['model_weights']['catboost']:.1%}, {catboost_status}): {result['individual_predictions']['catboost']} návštěvníků")
    
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
