"""
Skript pro stažení historických dat o počasí pro Plzeň
a jejich spojení s existujícími daty o návštěvnosti.
"""

import pandas as pd
import requests
from datetime import date, datetime, timedelta
from pathlib import Path
import time


def download_weather_data(start_date: str, end_date: str, output_file: str):
    """
    Stáhne historická data o počasí z Open-Meteo API.
    
    Args:
        start_date: Začátek období (YYYY-MM-DD)
        end_date: Konec období (YYYY-MM-DD)
        output_file: Cesta k výstupnímu CSV souboru
    """
    # Plzeň souřadnice
    LAT = 49.7384
    LON = 13.3736
    
    print("=" * 70)
    print("📥 Stahování historických dat o počasí pro Plzeň")
    print("=" * 70)
    print(f"📅 Období: {start_date} až {end_date}")
    print(f"📍 Lokace: Plzeň ({LAT}, {LON})")
    print(f"💾 Výstupní soubor: {output_file}")
    print()
    
    # Open-Meteo Archive API (historická data od 1940) 
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        'latitude': LAT,
        'longitude': LON,
        'start_date': start_date,
        'end_date': end_date,
        'daily': 'temperature_2m_max,temperature_2m_min,temperature_2m_mean,'
                'apparent_temperature_max,apparent_temperature_min,apparent_temperature_mean,'
                'precipitation_sum,rain_sum,snowfall_sum,precipitation_hours,'
                'weathercode,windspeed_10m_max,windgusts_10m_max,'
                'winddirection_10m_dominant,sunshine_duration,daylight_duration,'
                'cloudcover_mean',  # uv_index_max není dostupný v archive API
        'timezone': 'Europe/Prague'
    }
    
    print("🌐 Dotazuji Open-Meteo API...")
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        print("✅ Data úspěšně stažena!")
        
        # Převést na DataFrame
        daily = data['daily']
        
        df = pd.DataFrame({
            'date': pd.to_datetime(daily['time']),
            'temperature_max': daily['temperature_2m_max'],
            'temperature_min': daily['temperature_2m_min'],
            'temperature_mean': daily['temperature_2m_mean'],
            'apparent_temp_max': daily['apparent_temperature_max'],
            'apparent_temp_min': daily['apparent_temperature_min'],
            'apparent_temp_mean': daily['apparent_temperature_mean'],
            'precipitation': daily['precipitation_sum'],
            'rain': daily['rain_sum'],
            'snowfall': daily['snowfall_sum'],
            'precipitation_hours': daily['precipitation_hours'],
            'weather_code': daily['weathercode'],
            'wind_speed_max': daily['windspeed_10m_max'],
            'wind_gusts_max': daily['windgusts_10m_max'],
            'wind_direction': daily['winddirection_10m_dominant'],
            'sunshine_duration': daily['sunshine_duration'],
            'daylight_duration': daily['daylight_duration'],
            'cloud_cover_percent': daily.get('cloudcover_mean', [None]*len(daily['time'])),  
        })
        
        # Přidat odvozené features
        df['is_rainy'] = (df['precipitation'] > 1.0).astype(int)
        df['is_snowy'] = (df['snowfall'] > 1.0).astype(int)
        df['is_windy'] = (df['wind_speed_max'] > 30).astype(int)
        df['is_nice_weather'] = (
            (df['temperature_mean'] > 15) & 
            (df['precipitation'] < 1.0) &
            (df['weather_code'].isin([0, 1, 2]))
        ).astype(int)
        
        # Sunshine ratio (procento možného slunečního svitu)
        df['sunshine_ratio'] = df['sunshine_duration'] / df['daylight_duration']
        
        # Feels like delta (rozdíl mezi pocitovou a skutečnou teplotou)
        df['feels_like_delta'] = df['apparent_temp_mean'] - df['temperature_mean']
        
        # Weather forecast confidence (pro historická data = 1.0)
        df['weather_forecast_confidence'] = 1.0
        
        # Temperature trend 3d (bude vypočten později při slučování)
        df['temperature_trend_3d'] = 0.0
        
        # Is weather improving (bude vypočten později)
        df['is_weather_improving'] = 0
        
        print(f"📊 Zpracováno {len(df)} záznamů")
        print(f"   Rozsah dat: {df['date'].min()} až {df['date'].max()}")
        
        # Uložit
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        print(f"💾 Data uložena do: {output_path}")
        print()
        print("📋 Náhled dat:")
        print(df.head(10).to_string())
        print()
        print("📈 Statistiky:")
        print(df.describe())
        
        return df
        
    except requests.RequestException as e:
        print(f"❌ Chyba při stahování dat: {e}")
        return None
    except Exception as e:
        print(f"❌ Neočekávaná chyba: {e}")
        return None


def merge_weather_with_visitors(
    visitors_file: str,
    weather_file: str,
    output_file: str
):
    """
    Sloučí data o návštěvnosti s daty o počasí.
    
    Args:
        visitors_file: CSV s daty o návštěvnosti
        weather_file: CSV s daty o počasí
        output_file: Výstupní CSV s sloučenými daty
    """
    print("=" * 70)
    print("🔗 Slučování dat o návštěvnosti s počasím")
    print("=" * 70)
    
    # Načíst návštěvnost
    print(f"📂 Načítám návštěvnost: {visitors_file}")
    df_visitors = pd.read_csv(visitors_file)
    df_visitors['date'] = pd.to_datetime(df_visitors['date'])
    print(f"   ✓ {len(df_visitors)} záznamů")
    
    # Načíst počasí
    print(f"📂 Načítám počasí: {weather_file}")
    df_weather = pd.read_csv(weather_file)
    df_weather['date'] = pd.to_datetime(df_weather['date'])
    print(f"   ✓ {len(df_weather)} záznamů")
    
    # Sloučit (left join - zachovat všechny záznamy návštěvnosti)
    print("🔗 Slučuji data...")
    df_merged = pd.merge(
        df_visitors,
        df_weather,
        on='date',
        how='left'
    )
    
    print(f"✅ Sloučeno: {len(df_merged)} záznamů")
    print(f"   Záznamy s počasím: {df_merged['temperature_mean'].notna().sum()}")
    print(f"   Záznamy bez počasí: {df_merged['temperature_mean'].isna().sum()}")
    
    # Uložit
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(output_path, index=False)
    
    print(f"💾 Sloučená data uložena do: {output_path}")
    print()
    print("📋 Náhled sloučených dat:")
    print(df_merged.head(10).to_string())
    
    return df_merged


if __name__ == '__main__':
    # Nastavení cest
    project_root = Path(__file__).parent.parent
    
    visitors_file = project_root / 'data' / 'raw' / 'techmania_cleaned_master.csv'
    weather_file = project_root / 'data' / 'external' / 'weather_data.csv'
    merged_file = project_root / 'data' / 'processed' / 'techmania_with_weather.csv'
    
    # Načíst data návštěvnosti pro zjištění rozsahu
    df_visitors = pd.read_csv(visitors_file)
    df_visitors['date'] = pd.to_datetime(df_visitors['date'])
    
    start_date = df_visitors['date'].min().strftime('%Y-%m-%d')
    end_date = df_visitors['date'].max().strftime('%Y-%m-%d')
    
    print(f"\n📅 Rozsah dat návštěvnosti: {start_date} až {end_date}")
    print()
    
    # 1. Stáhnout historická data o počasí
    print("KROK 1/2: Stahování dat o počasí")
    print("-" * 70)
    df_weather = download_weather_data(start_date, end_date, str(weather_file))
    
    if df_weather is not None:
        print()
        print("KROK 2/2: Slučování dat")
        print("-" * 70)
        df_merged = merge_weather_with_visitors(
            str(visitors_file),
            str(weather_file),
            str(merged_file)
        )
        
        print()
        print("=" * 70)
        print("✅ HOTOVO! Data byla úspěšně stažena a sloučena.")
        print("=" * 70)
        print()
        print("📝 Další kroky:")
        print("1. Zkontrolujte soubor:", merged_file)
        print("2. Přetrénujte modely s novými features (počasí)")
        print("3. Aktualizujte feature_engineering.py pro podporu weather features")
    else:
        print()
        print("=" * 70)
        print("❌ CHYBA: Nepodařilo se stáhnout data o počasí")
        print("=" * 70)
