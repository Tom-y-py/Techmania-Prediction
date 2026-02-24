"""
Cleanup CSV files - odstraní dopočítané weather features které nejsou dostupné z API.

Odstraňované sloupce:
- apparent_temp_max, apparent_temp_min, apparent_temp_mean
- wind_direction
- sunshine_duration, daylight_duration, sunshine_ratio
- cloud_cover_percent
- feels_like_delta
- weather_forecast_confidence
- temperature_trend_3d
- is_weather_improving
"""

import pandas as pd
from pathlib import Path

# Sloupce k odstranění
COLUMNS_TO_REMOVE = [
    'apparent_temp_max',
    'apparent_temp_min', 
    'apparent_temp_mean',
    'wind_direction',
    'sunshine_duration',
    'daylight_duration',
    'sunshine_ratio',
    'cloud_cover_percent',
    'feels_like_delta',
    'weather_forecast_confidence',
    'temperature_trend_3d',
    'is_weather_improving',
]

# CSV soubory k vyčištění
CSV_FILES = [
    'techmania_with_weather.csv',
    'data/processed/techmania_with_weather_and_holidays.csv',
    'data/raw/techmania_2026_template.csv',
]


def cleanup_csv(file_path: str):
    """Vyčistí CSV soubor od nepotřebných sloupců."""
    path = Path(file_path)
    
    if not path.exists():
        print(f"❌ Soubor neexistuje: {file_path}")
        return
    
    print(f"\n📄 Zpracovávám: {file_path}")
    
    # Načíst CSV
    df = pd.read_csv(path)
    original_cols = len(df.columns)
    print(f"   Původní počet sloupců: {original_cols}")
    
    # Najít sloupce k odstranění (které existují v CSV)
    cols_to_drop = [col for col in COLUMNS_TO_REMOVE if col in df.columns]
    
    if not cols_to_drop:
        print(f"   ✅ Žádné sloupce k odstranění")
        return
    
    print(f"   🗑️  Odstraňuji sloupce ({len(cols_to_drop)}): {', '.join(cols_to_drop)}")
    
    # Odstranit sloupce
    df_cleaned = df.drop(columns=cols_to_drop)
    
    # Uložit zpět (BACKUP originálu)
    backup_path = path.with_suffix('.csv.backup')
    print(f"   💾 Zálohuji originál: {backup_path.name}")
    df.to_csv(backup_path, index=False)
    
    # Uložit vyčištěnou verzi
    print(f"   💾 Ukládám vyčištěný soubor...")
    df_cleaned.to_csv(path, index=False)
    
    new_cols = len(df_cleaned.columns)
    print(f"   ✅ Hotovo! Nový počet sloupců: {new_cols} (odstraněno {original_cols - new_cols})")
    print(f"   📊 Počet řádků: {len(df_cleaned)}")


def main():
    print("=" * 60)
    print("🧹 CSV CLEANUP - Odstranění dopočítaných weather features")
    print("=" * 60)
    
    for csv_file in CSV_FILES:
        cleanup_csv(csv_file)
    
    print("\n" + "=" * 60)
    print("✅ HOTOVO - Všechny CSV soubory vyčištěny!")
    print("=" * 60)
    print("\n💡 TIP: Originální soubory jsou zálohovány s příponou .csv.backup")
    print("   Pokud něco není OK, můžeš je obnovit:")
    print("   mv techmania_with_weather.csv.backup techmania_with_weather.csv")


if __name__ == '__main__':
    main()
