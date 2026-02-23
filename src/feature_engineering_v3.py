"""
Feature engineering V3 
"""
import pandas as pd
import numpy as np
from typing import Tuple


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vytvoří všechny potřebné features pro ensemble včetně EVENT DETECTION
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    print("🔧 Creating OPTIMIZED features with EVENT DETECTION...")
    
    # === ČASOVÉ FEATURES ===
    print("  ✓ Časové features (optimalizované)")
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['quarter'] = df['date'].dt.quarter
    
    # Víkend
    if 'is_weekend' not in df.columns:
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # === EVENT FEATURES - NOVÉ! ===
    print("  ✓ 🎯 EVENT FEATURES (speciální akce)")
    
    # 28. října - Den vzniku Československa (NEJVYŠŠÍ návštěvnost 4000-6700!)
    df['is_oct_28'] = (
        (df['month'] == 10) & (df['date'].dt.day == 28)
    ).astype(int)
    
    # 27.-28. října extended (často celý víkend kolem 28.10)
    df['is_oct_28_weekend'] = (
        (df['month'] == 10) & 
        (df['date'].dt.day.isin([27, 28, 29])) &
        (df['day_of_week'].isin([4, 5, 6, 0]))  # Pátek-Pondělí
    ).astype(int)
    
    # Podzimní prázdniny (týden 43-44, konec října)
    df['is_autumn_break'] = (
        (df['week_of_year'].isin([43, 44])) &
        (df['month'] == 10)
    ).astype(int)
    
    # Letní víkendové eventy (září začátek školního roku, červen konec)
    df['is_summer_weekend_event'] = (
        (df['month'].isin([6, 9])) &
        (df['day_of_week'] >= 5) &  # Víkend
        (df['date'].dt.day <= 7)     # První týden v měsíci
    ).astype(int)
    
    # Jarní prázdniny víkend (duben)
    df['is_spring_break_weekend'] = (
        (df['month'].isin([4])) &
        (df['day_of_week'] >= 5)
    ).astype(int)
    
    # HIGH TRAFFIC SEASON (září-listopad, hlavní sezóna školních exkurzí)
    df['is_high_traffic_season'] = (
        df['month'].isin([9, 10, 11])
    ).astype(int)
    
    # Event score - kombinace faktorů
    df['event_score'] = (
        df['is_oct_28'] * 10.0 +
        df['is_oct_28_weekend'] * 5.0 +
        df['is_autumn_break'] * 3.0 +
        df['is_summer_weekend_event'] * 2.0 +
        df['is_spring_break_weekend'] * 1.5 +
        df['is_high_traffic_season'] * 1.0
    )
    
    # === SEZÓNNÍ FEATURES ===
    print("  ✓ Sezónní features")
    if 'is_summer_holiday' not in df.columns:
        df['is_summer_holiday'] = df['month'].isin([7, 8]).astype(int)
    
    if 'is_winter_holiday' not in df.columns:
        df['is_winter_holiday'] = (
            ((df['month'] == 12) & (df['date'].dt.day >= 23)) |
            ((df['month'] == 1) & (df['date'].dt.day <= 2))
        ).astype(int)
    
    df['is_school_year'] = (~df['month'].isin([7, 8])).astype(int)
    
    # === SVÁTKY ===
    print("  ✓ Svátky (upraveno)")
    if 'is_holiday' not in df.columns:
        if 'extra' in df.columns:
            df['is_holiday'] = df['extra'].notna().astype(int)
        else:
            df['is_holiday'] = 0
    
    # Hlavní svátky (bez škodlivého efektu)
    df['is_major_holiday'] = (
        ((df['month'] == 12) & (df['date'].dt.day.isin([24, 25, 26]))) |  # Vánoce
        ((df['month'] == 1) & (df['date'].dt.day == 1)) |  # Nový rok
        ((df['month'] == 5) & (df['date'].dt.day == 1))    # Svátek práce
    ).astype(int)
    
    # === CLOSURE DETECTION ===
    print("  ✓ Closure detection")
    df['is_monday_not_summer'] = (
        (df['day_of_week'] == 0) &
        (~df['month'].isin([7, 8]))
    ).astype(int)
    
    df['is_christmas_closure'] = (
        (df['month'] == 12) & 
        (df['date'].dt.day.isin([24, 25, 26]))
    ).astype(int)
    
    df['is_new_year_period'] = (
        ((df['month'] == 12) & (df['date'].dt.day == 31)) |
        ((df['month'] == 1) & (df['date'].dt.day == 1))
    ).astype(int)
    
    df['closure_risk_score'] = (
        df['is_monday_not_summer'] * 0.517 +
        df['is_christmas_closure'] * 1.0 +
        df['is_new_year_period'] * 0.8
    )
    
    df['monday_winter'] = (
        (df['day_of_week'] == 0) &
        (df['month'].isin([11, 12, 1, 2, 3]))
    ).astype(int)
    
    # === CYKLICKÉ ENKÓDOVÁNÍ ===
    print("  ✓ Cyklické enkódování")
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
    
    # === NORMALIZED TIME (místo days_since_start) ===
    print("  ✓ Normalized time")
    df['normalized_time'] = (df['date'] - df['date'].min()).dt.days / 365.25
    
    # === WEATHER LAG FEATURES ===
    print("  ✓ Weather lag features (3-day, 7-day)")
    if 'temperature_mean' in df.columns:
        df['temperature_3day_avg'] = df['temperature_mean'].rolling(window=3, min_periods=1).mean()
        df['temperature_7day_avg'] = df['temperature_mean'].rolling(window=7, min_periods=1).mean()
        df['temperature_3day_trend'] = df['temperature_mean'] - df['temperature_3day_avg']
        
    if 'precipitation' in df.columns:
        df['precipitation_3day_sum'] = df['precipitation'].rolling(window=3, min_periods=1).sum()
        df['precipitation_7day_sum'] = df['precipitation'].rolling(window=7, min_periods=1).sum()
    
    # === WEATHER INTERACTION FEATURES ===
    print("  ✓ Weather interaction features")
    
    # Alias pro wind_speed_max (kvůli zpětné kompatibilitě s modely)
    if 'wind_speed' in df.columns and 'wind_speed_max' not in df.columns:
        df['wind_speed_max'] = df['wind_speed']
    
    # REMOVED: sunshine_duration, daylight_duration, sunshine_ratio (není dostupné z API)
    # REMOVED: feels_like_delta (není dostupné z API)
    # REMOVED: weather_forecast_confidence (dopočítané, odstraněno)
    
    # Weather improving (BETTER VERSION)
    if 'temperature_mean' in df.columns and 'precipitation' in df.columns:
        df['weather_improving'] = (
            (df['temperature_mean'] > df['temperature_3day_avg']) &
            (df['precipitation'] < df['precipitation_3day_sum'] / 3)
        ).astype(int)
        
    # Heavy rain on weekend (může odrazovat návštěvníky)
    if 'precipitation' in df.columns:
        df['heavy_rain_weekend'] = (
            (df['precipitation'] > 5) &
            (df['is_weekend'] == 1)
        ).astype(int)
    
    # REMOVED: weather_forecast_confidence (dopočítané, odstraněno)
    
    # REMOVED: rain (korelace 0.988 s precipitation)
    # REMOVED: apparent_temp_* (korelace 0.99 s temperature_*)
    # REMOVED: wind_direction, sunshine_*, daylight_duration, cloud_cover_percent
    
    # === GOOGLE TRENDS ===
    print("  ✓ Google Trends")
    if 'google_trend' in df.columns:
        df['google_trend_lag1'] = df['google_trend'].shift(1).fillna(df['google_trend'].mean())
        df['google_trend_lag7'] = df['google_trend'].shift(7).fillna(df['google_trend'].mean())
        df['google_trend_rolling'] = df['google_trend'].rolling(window=7, min_periods=1).mean()
        df['google_trend_trend'] = df['google_trend'] - df['google_trend_rolling']
    
    # === STATISTICAL FEATURES ===
    
    print(f"✅ Total features created: {len(df.columns)}")
    
    # === FILL REMAINING NaN VALUES ===
    print("  ✓ Filling remaining NaN values")
    
    # precipitation_probability - fill with 0 (není dostupná)
    if 'precipitation_probability' in df.columns:
        df.loc[:, 'precipitation_probability'] = df['precipitation_probability'].fillna(0)
    
    # School features - fill with 0 (žádní návštěvníci ze škol)
    school_cols = ['Mateřská_škola', 'Základní_škola', 'Střední_škola']
    for col in school_cols:
        if col in df.columns:
            df[col].fillna(0, inplace=True)
    
    # Extra, school_break_type - kategorické, ponechat NaN nebo fill s ''
    # (tyto se stejně odstraní před tréninkem jako non-numeric)
    
    # Fill any other remaining NaN with 0
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    print(f"✅ All NaN values filled")
    
    return df


def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Připraví data pro trénování
    """
    df = create_features(df)
    
    # Features k odstranění
    remove_cols = ['date', 'total_visitors', 'school_visitors', 'public_visitors',
                   'extra', 'nazvy_svatek', 'school_break_type', 'season_exact', 'week_position']
    
    # Zachovat jen numeric features
    feature_cols = [col for col in df.columns if col not in remove_cols]
    numeric_features = [col for col in feature_cols if df[col].dtype in ['int64', 'float64', 'bool']]
    
    X = df[numeric_features]
    y = df['total_visitors'] if 'total_visitors' in df.columns else None
    
    print(f"\n📊 Feature matrix shape: {X.shape}")
    print(f"   Numeric features: {len(numeric_features)}")
    
    return X, y
