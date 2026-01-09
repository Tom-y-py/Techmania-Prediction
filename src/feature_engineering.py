"""
Feature Engineering pro Ensemble Model
Vytváří všechny potřebné features pro predikci návštěvnosti Techmanie
"""

import pandas as pd
import numpy as np
from typing import Tuple


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vytvoří všechny potřebné features pro ensemble
    
    Args:
        df: DataFrame s minimálně sloupci ['date', 'total_visitors']
        
    Returns:
        DataFrame s přidanými features
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    print("🔧 Creating features...")
    
    # === ČASOVÉ FEATURES ===
    print("  ✓ Časové features (rok, měsíc, den, týden...)")
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['quarter'] = df['date'].dt.quarter
    df['day_of_year'] = df['date'].dt.dayofyear
    
    # Víkend
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # === LAG FEATURES (historické hodnoty) ===
    print("  ✓ Lag features (1, 7, 14, 30 dní zpět)")
    for lag in [1, 7, 14, 30]:
        # Shift vytvoří NaN pro prvních 'lag' řádků
        # NaN hodnoty zůstanou - budou ošetřeny při splitu dat
        df[f'visitors_lag_{lag}'] = df['total_visitors'].shift(lag)
    
    # === ROLLING STATISTICS ===
    print("  ✓ Rolling statistics (mean, std, min, max)")
    for window in [7, 14, 30]:
        # min_periods nastavíme na window/2 aby byla statistika validní
        # NaN hodnoty zůstanou pro řádky bez dostatečné historie
        min_periods = max(1, window // 2)
        df[f'visitors_rolling_mean_{window}'] = (
            df['total_visitors'].rolling(window=window, min_periods=min_periods).mean()
        )
        df[f'visitors_rolling_std_{window}'] = (
            df['total_visitors'].rolling(window=window, min_periods=min_periods).std()
        )
        df[f'visitors_rolling_min_{window}'] = (
            df['total_visitors'].rolling(window=window, min_periods=min_periods).min()
        )
        df[f'visitors_rolling_max_{window}'] = (
            df['total_visitors'].rolling(window=window, min_periods=min_periods).max()
        )
    
    # === SEZÓNNÍ FEATURES ===
    print("  ✓ Sezónní features (prázdniny, školní rok)")
    # Letní prázdniny (červenec + srpen)
    df['is_summer_holiday'] = df['month'].isin([7, 8]).astype(int)
    
    # Vánoční prázdniny (23.12 - 2.1)
    df['is_winter_holiday'] = (
        ((df['month'] == 12) & (df['day'] >= 23)) |
        ((df['month'] == 1) & (df['day'] <= 2))
    ).astype(int)
    
    # Školní rok vs prázdniny
    df['is_school_year'] = (~df['month'].isin([7, 8])).astype(int)
    
    # === SVÁTKY (z extra sloupce) ===
    print("  ✓ Svátky")
    if 'extra' in df.columns:
        df['is_holiday'] = df['extra'].notna().astype(int)
    else:
        df['is_holiday'] = 0
    
    # === DERIVED FEATURES ===
    print("  ✓ Odvozené features")
    # Poměr školní/veřejní návštěvníci (pokud existují)
    if 'school_visitors' in df.columns and 'public_visitors' in df.columns:
        df['school_ratio'] = df['school_visitors'] / (df['total_visitors'] + 1)
        df['public_ratio'] = df['public_visitors'] / (df['total_visitors'] + 1)
    
    # Otevírací doba v hodinách
    if 'opening_hours' in df.columns:
        # Konverze textových hodnot na čísla
        df['is_closed'] = df['opening_hours'].fillna('').str.contains('zavřeno', case=False).astype(int)
    
    # Trend (lineární číslo dne)
    df['days_since_start'] = (df['date'] - df['date'].min()).dt.days
    
    # Cyklické features pro den v týdnu a měsíc (pro lepší zachycení periodicity)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # === WEATHER FEATURES ===
    # Pokud data obsahují weather sloupce, použijeme je přímo
    weather_cols = [
        'temperature_max', 'temperature_min', 'temperature_mean',
        'apparent_temp_max', 'apparent_temp_min', 'apparent_temp_mean',
        'precipitation', 'rain', 'snowfall', 'precipitation_hours',
        'weather_code', 'wind_speed_max', 'wind_gusts_max', 'wind_direction',
        'sunshine_duration', 'daylight_duration',
        'is_rainy', 'is_snowy', 'is_windy', 'is_nice_weather', 'sunshine_ratio'
    ]
    
    weather_present = [col for col in weather_cols if col in df.columns]
    if weather_present:
        print(f"  ✓ Weather features ({len(weather_present)} sloupců): {', '.join(weather_present[:5])}...")
    else:
        print("  ⚠️ Weather features nejsou v datech - byly přeskočeny")
    
    print(f"✅ Created {len(df.columns)} features total")
    
    return df


def split_data(
    df: pd.DataFrame, 
    train_end: str = '2024-12-31', 
    val_end: str = '2025-12-31'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronologický split dat
    
    Args:
        df: DataFrame s features
        train_end: Konec trénovací periody
        val_end: Konec validační periody
        
    Returns:
        Tuple[train, validation, test] DataFrames
    """
    print(f"\n📊 Splitting data...")
    
    numeric_cols = df.select_dtypes(include=['int64', 'int32', 'float64', 'float32', 'bool', 'uint8']).columns
    df_before = len(df)
    df = df.dropna(subset=numeric_cols)
    print(f"  Dropped {df_before - len(df)} rows with NaN in numeric features")
    print(f"  Remaining data: {len(df)} rows ({df['date'].min()} - {df['date'].max()})")
    
    print(f"\n  Train period: до {train_end}")
    print(f"  Validation period: {train_end} - {val_end}")
    print(f"  Test period: od {val_end}")
    
    train = df[df['date'] <= train_end].copy()
    val = df[(df['date'] > train_end) & (df['date'] <= val_end)].copy()
    test = df[df['date'] > val_end].copy()
    
    print(f"\n  Train: {len(train)} záznamů")
    print(f"  Validation: {len(val)} záznamů")
    print(f"  Test: {len(test)} záznamů")
    
    return train, val, test


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Vrátí seznam sloupců pro použití jako features (X)
    
    Args:
        df: DataFrame s všemi sloupci
        
    Returns:
        List feature column names (pouze číselné)
    """
    # Vyloučit target a metadata sloupce
    exclude_cols = [
        'date', 
        'total_visitors',  # target
        'school_visitors',  # součást targetu
        'public_visitors',  # součást targetu
        'extra',  # text metadata
        'opening_hours',  # text metadata
        'day_of_week_str',  # pokud existuje textová verze
        'nazvy_svatek',  # text názvy svátků
        'day_of_week',  # textový název dne (pokud existuje)
    ]
    
    # Vybrat pouze sloupce, které nejsou v exclude_cols
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Navíc vyfiltrovat pouze číselné sloupce (int, float, bool)
    numeric_features = []
    for col in feature_cols:
        if df[col].dtype in ['int64', 'int32', 'float64', 'float32', 'bool', 'uint8']:
            numeric_features.append(col)
    
    feature_cols = numeric_features
    
    print(f"\n📋 Feature columns ({len(feature_cols)}):")
    if len(feature_cols) <= 15:
        print(f"  {', '.join(feature_cols)}")
    else:
        print(f"  {', '.join(feature_cols[:15])}... (+{len(feature_cols)-15} more)")
    
    return feature_cols


if __name__ == '__main__':
    # Test feature engineering
    print("=" * 60)
    print("Testing Feature Engineering with Weather Data")
    print("=" * 60)
    
    # Načíst data S POČASÍM (již sloučená návštěvnost + počasí)
    import os
    from pathlib import Path
    
    # Získat správnou cestu (src složka -> parent -> data)
    script_dir = Path(__file__).parent
    data_file = script_dir.parent / 'data' / 'processed' / 'techmania_with_weather.csv'
    
    df = pd.read_csv(data_file)
    print(f"\n📂 Loaded {len(df)} records from: {data_file.name}")
    print(f"   Date range: {df['date'].min()} - {df['date'].max()}")
    
    # Ukázat, že máme weather data
    weather_cols = ['temperature_mean', 'precipitation', 'is_rainy', 'is_snowy']
    present_weather = [col for col in weather_cols if col in df.columns]
    print(f"   Weather columns present: {present_weather}")
    
    # Vytvořit features
    df = create_features(df)
    
    # Split data
    train, val, test = split_data(df)
    
    # Get feature columns
    feature_cols = get_feature_columns(df)
    
    print("\n" + "=" * 60)
    print("✅ Feature Engineering Test Complete!")
    print("=" * 60)
    
    print("\n📋 Všechny sloupce:")
    print(df.columns.tolist())
    print(f"   Total features: {len(feature_cols)}")
