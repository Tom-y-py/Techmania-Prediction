"""
Feature Engineering pro Ensemble Model
Vytváří všechny potřebné features pro predikci návštěvnosti Techmanie

DEPRICATED
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
    
    # Víkend - pouze pokud už není v datech
    if 'is_weekend' not in df.columns:
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # === LAG FEATURES (historické hodnoty) - VYPNUTO PRO LEPŠÍ POČASÍ ===
    # Tyto features způsobují, že model ignoruje počasí, protože se spoléhá na historii
    # Při predikci do budoucna nejsou dostupné, takže se nahrazují mediánem
    # print("  ✓ Lag features (1, 7, 14, 30 dní zpět)")
    # for lag in [1, 7, 14, 30]:
    #     df[f'visitors_lag_{lag}'] = df['total_visitors'].shift(lag)
    
    # === ROLLING STATISTICS - VYPNUTO PRO LEPŠÍ POČASÍ ===
    # print("  ✓ Rolling statistics (mean, std, min, max)")
    # for window in [7, 14, 30]:
    #     min_periods = max(1, window // 2)
    #     df[f'visitors_rolling_mean_{window}'] = (
    #         df['total_visitors'].rolling(window=window, min_periods=min_periods).mean()
    #     )
    #     df[f'visitors_rolling_std_{window}'] = (
    #         df['total_visitors'].rolling(window=window, min_periods=min_periods).std()
    #     )
    #     df[f'visitors_rolling_min_{window}'] = (
    #         df['total_visitors'].rolling(window=window, min_periods=min_periods).min()
    #     )
    #     df[f'visitors_rolling_max_{window}'] = (
    #         df['total_visitors'].rolling(window=window, min_periods=min_periods).max()
    #     )
    
    # === SEZÓNNÍ FEATURES ===
    print("  ✓ Sezónní features (prázdniny, školní rok)")
    # Letní prázdniny (červenec + srpen) - pouze pokud už není v datech
    if 'is_summer_holiday' not in df.columns:
        df['is_summer_holiday'] = df['month'].isin([7, 8]).astype(int)
    
    # Vánoční prázdniny (23.12 - 2.1)
    if 'is_winter_holiday' not in df.columns:
        df['is_winter_holiday'] = (
            ((df['month'] == 12) & (df['day'] >= 23)) |
            ((df['month'] == 1) & (df['day'] <= 2))
        ).astype(int)
    
    # Školní rok vs prázdniny
    df['is_school_year'] = (~df['month'].isin([7, 8])).astype(int)
    
    # === SVÁTKY (z extra sloupce) ===
    print("  ✓ Svátky")
    # is_holiday už je v datech z CSV
    if 'is_holiday' not in df.columns:
        if 'extra' in df.columns:
            df['is_holiday'] = df['extra'].notna().astype(int)
        else:
            df['is_holiday'] = 0
    
    # === FEATURES PRO DETEKCI ZAVŘENÝCH DNŮ ===
    print("  ✓ Detekce pravděpodobně zavřených dní")
    
    # 1. Pondělky mimo léto (51.7% je zavřeno - VELMI SILNÝ SIGNÁL!)
    df['is_monday_not_summer'] = (
        (df['day_of_week'] == 0) &  # Pondělí
        (~df['month'].isin([7, 8]))  # Není léto
    ).astype(int)
    
    # 2. Vánoční období (24-26.12) - VŽDY zavřeno
    df['is_christmas_closure'] = (
        (df['month'] == 12) & 
        (df['day'].isin([24, 25, 26]))
    ).astype(int)
    
    # 3. Silvestr a Nový rok
    df['is_new_year_period'] = (
        ((df['month'] == 12) & (df['day'] == 31)) |  # Silvester
        ((df['month'] == 1) & (df['day'] == 1))       # Nový rok
    ).astype(int)
    
    # 4. Kombinovaný "risk of closure" score
    # Čím vyšší, tím vyšší pravděpodobnost zavření
    df['closure_risk_score'] = (
        df['is_christmas_closure'] * 100 +        # Vánoce = 100% riziko
        df['is_new_year_period'] * 80 +           # Silvester/Nový rok = 80%
        df['is_monday_not_summer'] * 50 +         # Pondělí mimo léto = 50%
        (df['is_holiday'] * df['day_of_week'] == 0).astype(int) * 30  # Svátek v pondělí = +30
    )
    
    # 5. Interakce: Pondělí × zimní měsíce (ještě vyšší riziko zavření)
    df['monday_winter'] = (
        (df['day_of_week'] == 0) & 
        (df['month'].isin([11, 12, 1, 2]))  # Zima
    ).astype(int)
    
    # === DERIVED FEATURES ===
    print("  ✓ Odvozené features")
    # VYPNUTO: Poměr školní/veřejní návštěvníci - spoléhá na historická data
    # Tyto features nejsou dostupné při predikci do budoucna
    # if 'school_visitors' in df.columns and 'public_visitors' in df.columns:
    #     df['school_ratio'] = df['school_visitors'] / (df['total_visitors'] + 1)
    #     df['public_ratio'] = df['public_visitors'] / (df['total_visitors'] + 1)
    
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
        
        # === INTERAKCE POČASÍ × ČAS (klíčové pro predikci!) ===
        print("  ✓ Weather interactions (počasí × víkend, měsíc, atd.)")
        
        # Teplota × Víkend (v zimě víkend + špatné počasí = méně lidí)
        if 'temperature_mean' in df.columns:
            df['temp_x_weekend'] = df['temperature_mean'] * df['is_weekend']
            df['temp_x_summer'] = df['temperature_mean'] * df['is_summer_holiday']
            df['temp_x_month'] = df['temperature_mean'] * df['month']
            
            # Exponenciální penalizace pro nízké teploty (pod 0°C je mnohem horší)
            # Čím nižší teplota, tím silnější negativní efekt
            df['cold_penalty'] = np.where(
                df['temperature_mean'] < 0,
                -(df['temperature_mean'] ** 2) / 10,  # Kvadratická penalizace pro mráz
                0
            )
            
            # Mráz speciálně o víkendu (kdy by normálně bylo nejvíc lidí)
            df['weekend_cold_penalty'] = df['is_weekend'] * df['cold_penalty']
        
        # Srážky × Víkend - ROZLIŠUJEME déšť vs sníh!
        if 'precipitation' in df.columns:
            # Déšť o víkendu = lidé hledají vnitřní aktivity = MŮŽE BÝT BONUS
            df['rain_x_weekend'] = (
                (df['precipitation'] > 0).astype(int) * 
                (df['temperature_mean'] > 5).astype(int) *  # Bezpečné teploty
                df['is_weekend'] * 
                (df['is_snowy'] == 0).astype(int)
            )
            
            # Pro kompatibilitu (starý feature) - ale s menší váhou pro déšť
            df['precip_x_weekend'] = df['precipitation'] * df['is_weekend']
            df['precip_x_summer'] = df['precipitation'] * df['is_summer_holiday']
        
        # Sníh × Víkend
        if 'snowfall' in df.columns:
            df['snow_x_weekend'] = df['snowfall'] * df['is_weekend']
        
        # Špatné počasí indikátory (kombinace faktorů)
        if 'temperature_mean' in df.columns and 'precipitation' in df.columns:
            # EXTRÉMNĚ silné penalizace pro zimní podmínky
            df['is_freezing'] = (df['temperature_mean'] < 0).astype(int)
            df['is_very_cold'] = (df['temperature_mean'] < -5).astype(int)
            df['is_extreme_cold'] = (df['temperature_mean'] < -10).astype(int)
            
            # Kombinace mrazu a srážek/sněhu = katastrofa pro návštěvnost
            df['freezing_with_snow'] = (
                (df['temperature_mean'] < 0).astype(int) * 
                ((df['is_snowy'] > 0) | (df['snowfall'] > 0)).astype(int)
            )
            
            df['freezing_with_precip'] = (
                (df['temperature_mean'] < 0).astype(int) * 
                (df['precipitation'] > 0).astype(int)
            )
            
            # KLÍČOVÉ: Déšť vs Sníh - rozdílný efekt!
            # Déšť (teplo) = lidé jdou dovnitř = BONUS pro Techmanii ✅
            # Sníh (zima) = nebezpečné cesty = PENALIZACE ❌
            df['rain_indoor_bonus'] = (
                (df['temperature_mean'] > 5).astype(int) *  # Teplo = bezpečné cesty
                (df['precipitation'] > 2).astype(int) *      # Hodně prší
                (df['is_snowy'] == 0).astype(int)            # Není sníh
            )
            
            # Špatné počasí score - ROZDÍLNÉ pro sníh vs déšť
            df['bad_weather_score'] = (
                # MRÁZ a SNÍH = velmi špatné (nebezpečné cesty, školy zavřené)
                (df['temperature_mean'] < 0).astype(int) * 6 +       # Mráz = 6 bodů PENALIZACE
                (df['temperature_mean'] < -5).astype(int) * 4 +      # Pod -5°C = +4 body
                (df['temperature_mean'] < -10).astype(int) * 4 +     # Pod -10°C = +4 body
                df['is_snowy'] * 8 +                                 # Sníh = 8 bodů! (NEJVĚTŠÍ penalizace)
                (df['snowfall'] > 0).astype(int) * 5 +               # Sněžení = 5 bodů
                
                # DÉŠŤ (bez mrazu) = malá penalizace (jen nepohodlí, ale lidé stejně jedou)
                (
                    (df['temperature_mean'] >= 5).astype(int) *      # Bezpečné teploty
                    (df['precipitation'] > 5).astype(int) *          # Hodně prší
                    (df['is_snowy'] == 0).astype(int)                # Není sníh
                ) * 1 +                                              # Jen 1 bod (téměř žádná penalizace)
                
                df['is_windy'] * 2                                   # Vítr = 2 body
            )
            
            # Mráz + sníh + víkend = EXTRÉMNÍ penalizace (nikdo nejede)
            df['weekend_frozen_nightmare'] = (
                df['is_weekend'] * 
                (df['temperature_mean'] < 0).astype(int) * 
                df['is_snowy'] * 3  # Trojnásobek efektu!
            )
            
            # NOVÝ: Déšť + víkend = BONUS (lidé hledají vnitřní aktivity)
            df['rainy_weekend_bonus'] = (
                df['is_weekend'] * 
                (df['temperature_mean'] > 5).astype(int) *  # Bezpečné teploty
                (df['precipitation'] > 1).astype(int) *     # Prší
                (df['is_snowy'] == 0).astype(int)           # Není sníh
            )
            
            # Perfektní počasí pro návštěvu 
            # POZOR: Hezké počasí o víkendu může být HORŠÍ (konkurence venkovních aktivit!)
            df['perfect_weather_score'] = (
                (df['temperature_mean'] > 15).astype(int) * 2 +  # Teplo = 2 body
                (df['temperature_mean'] > 20).astype(int) +       # Ještě tepleji = +1 bod
                (df['precipitation'] == 0).astype(int) * 2 +      # Sucho = 2 body
                df['is_nice_weather'] * 2 +                       # Hezké počasí = 2 body
                df['is_weekend']                                   # Víkend = 1 bod
            )
            
            # NOVÝ: Konkurence venkovních aktivit (hezké počasí o víkendu = méně lidí)
            df['outdoor_competition'] = (
                (df['temperature_mean'] > 18).astype(int) *  # Krásné teplo
                (df['precipitation'] == 0).astype(int) *     # Sucho
                df['is_weekend'] *                           # Víkend
                (df['is_summer_holiday'] == 0).astype(int)   # Mimo hlavní prázdniny
            )
        
        # Tepelný komfort (ne moc horko, ne moc zima)
        if 'temperature_mean' in df.columns:
            df['temp_comfort'] = np.exp(-((df['temperature_mean'] - 18) ** 2) / 100)
        
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
    
    # Vyplnit NaN hodnoty místo mazání řádků
    numeric_cols = df.select_dtypes(include=['int64', 'int32', 'float64', 'float32', 'bool', 'uint8']).columns
    nan_counts = df[numeric_cols].isna().sum()
    cols_with_nan = nan_counts[nan_counts > 0]
    
    if len(cols_with_nan) > 0:
        print(f"  Found NaN values in {len(cols_with_nan)} columns, filling with 0...")
        for col in cols_with_nan.index:
            print(f"    - {col}: {cols_with_nan[col]} NaN values")
        df[numeric_cols] = df[numeric_cols].fillna(0)
    
    print(f"  Total data: {len(df)} rows ({df['date'].min()} - {df['date'].max()})")
    
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
        'day_of_week',  # textový název dne (pátek, sobota, ...)
        'nazvy_svatek',  # text názvy svátků
        'school_break_type',  # text typ prázdnin
        'season_exact',  # text název sezóny
        'week_position',  # text pozice v týdnu
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
