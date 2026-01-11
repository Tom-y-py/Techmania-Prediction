"""
Skript pro kontrolu CSV souboru - ověření všech požadovaných sloupců a smysluplnosti dat
"""

import pandas as pd
import sys
from pathlib import Path

# Definice očekávaných sloupců (AKTUALIZOVÁNO podle skutečných názvů v CSV)
REQUIRED_COLUMNS = {
    # Dostupné historicky i do budoucna - ŠKOLNÍ PRÁZDNINY
    'is_any_school_break': {'type': 'bool', 'range': [0, 1], 'description': 'Jakékoliv školní prázdniny'},
    'days_to_next_break': {'type': 'int', 'range': [0, 365], 'description': 'Dny do začátku dalších prázdnin'},
    'days_from_last_break': {'type': 'int', 'range': [0, 365], 'description': 'Dny od konce posledních prázdnin'},
    'is_week_before_break': {'type': 'bool', 'range': [0, 1], 'description': 'Týden před prázdninami'},
    'is_week_after_break': {'type': 'bool', 'range': [0, 1], 'description': 'Týden po prázdninách'},
    'school_week_number': {'type': 'int', 'range': [0, 52], 'description': 'Týden školního roku'},
    
    # Detailní typy prázdnin
    'is_spring_break': {'type': 'bool', 'range': [0, 1], 'description': 'Jarní prázdniny'},
    'is_autumn_break': {'type': 'bool', 'range': [0, 1], 'description': 'Podzimní prázdniny'},
    'is_winter_break': {'type': 'bool', 'range': [0, 1], 'description': 'Zimní prázdniny'},
    'is_easter_break': {'type': 'bool', 'range': [0, 1], 'description': 'Velikonoční prázdniny'},
    'is_halfyear_break': {'type': 'bool', 'range': [0, 1], 'description': 'Pololetní prázdniny'},
    'is_summer_holiday': {'type': 'bool', 'range': [0, 1], 'description': 'Letní prázdniny'},
    
    # Svátky
    'is_holiday': {'type': 'bool', 'range': [0, 1], 'description': 'Státní svátek'},
    
    # Lze vypočítat z data
    'is_bridge_day': {'type': 'bool', 'range': [0, 1], 'description': 'Most mezi svátkem a víkendem'},
    'long_weekend_length': {'type': 'int', 'range': [0, 6], 'description': 'Délka prodlouženého víkendu'},
    'week_position': {'type': 'int', 'range': [1, 5], 'description': 'Pozice v měsíci (týden)'},
    'is_month_end': {'type': 'bool', 'range': [0, 1], 'description': 'Konec měsíce'},
    'season_exact': {'type': 'int', 'range': [1, 4], 'description': 'Přesné roční období (1-4)'},
    'is_weekend': {'type': 'bool', 'range': [0, 1], 'description': 'Víkend'},
    
    # Z weather API - základní počasí
    'temperature_mean': {'type': 'float', 'range': [-30, 40], 'description': 'Průměrná teplota'},
    'precipitation': {'type': 'float', 'range': [0, 200], 'description': 'Srážky (mm)'},
    'cloud_cover_percent': {'type': 'float', 'range': [0, 100], 'description': 'Oblačnost v %'},
    'wind_speed_max': {'type': 'float', 'range': [0, 150], 'description': 'Maximální rychlost větru'},
    'sunshine_duration': {'type': 'float', 'range': [0, 86400], 'description': 'Délka slunečního svitu (s)'},
    
    # Z weather API - odvozené features
    'weather_forecast_confidence': {'type': 'float', 'range': [0, 1], 'description': 'Spolehlivost předpovědi'},
    'temperature_trend_3d': {'type': 'float', 'range': [-30, 30], 'description': 'Trend teploty za 3 dny'},
    'is_weather_improving': {'type': 'bool', 'range': [0, 1], 'description': 'Počasí se zlepšuje'},
    'feels_like_delta': {'type': 'float', 'range': [-30, 30], 'description': 'Rozdíl pocitové vs skutečné teploty'},
    'is_rainy': {'type': 'bool', 'range': [0, 1], 'description': 'Prší'},
    'is_snowy': {'type': 'bool', 'range': [0, 1], 'description': 'Sněží'},
    'is_nice_weather': {'type': 'bool', 'range': [0, 1], 'description': 'Hezké počasí'},
}


def check_column_exists(df: pd.DataFrame, column_name: str) -> dict:
    """Zkontroluje, zda sloupec existuje"""
    exists = column_name in df.columns
    return {
        'exists': exists,
        'message': '✅ Existuje' if exists else '❌ CHYBÍ'
    }


def check_data_validity(df: pd.DataFrame, column_name: str, config: dict) -> dict:
    """Zkontroluje smysluplnost dat ve sloupci"""
    if column_name not in df.columns:
        return {'valid': False, 'issues': ['Sloupec neexistuje']}
    
    col_data = df[column_name]
    issues = []
    warnings = []
    
    # Kontrola prázdných hodnot
    null_count = col_data.isnull().sum()
    null_percent = (null_count / len(col_data)) * 100
    
    if null_percent > 0:
        warnings.append(f'🟡 {null_count} prázdných hodnot ({null_percent:.1f}%)')
    
    # Kontrola rozsahu hodnot (pouze pro ne-prázdné hodnoty)
    non_null_data = col_data.dropna()
    
    if len(non_null_data) > 0:
        min_val = non_null_data.min()
        max_val = non_null_data.max()
        expected_min, expected_max = config['range']
        
        # Kontrola, zda jsou hodnoty v očekávaném rozsahu
        out_of_range = ((non_null_data < expected_min) | (non_null_data > expected_max)).sum()
        
        if out_of_range > 0:
            issues.append(f'❌ {out_of_range} hodnot mimo rozsah [{expected_min}, {expected_max}]')
        
        # Info o rozsahu dat
        range_info = f'Rozsah: [{min_val:.2f}, {max_val:.2f}]'
        
        # Kontrola variance (zda nejsou všechny hodnoty stejné)
        if non_null_data.nunique() == 1:
            warnings.append(f'🟡 Všechny hodnoty jsou stejné: {non_null_data.iloc[0]}')
        
        # Kontrola pro boolean sloupce
        if config['type'] == 'bool':
            unique_vals = set(non_null_data.unique())
            if not unique_vals.issubset({0, 1}):
                issues.append(f'❌ Boolean sloupec obsahuje jiné hodnoty než 0/1: {unique_vals}')
    else:
        issues.append('❌ Žádná validní data')
        range_info = 'N/A'
    
    is_valid = len(issues) == 0
    
    return {
        'valid': is_valid,
        'issues': issues,
        'warnings': warnings,
        'range_info': range_info if len(non_null_data) > 0 else 'N/A',
        'null_percent': null_percent,
        'unique_values': int(non_null_data.nunique()) if len(non_null_data) > 0 else 0
    }


def check_csv_file(csv_path: str) -> None:
    """Hlavní funkce pro kontrolu CSV souboru"""
    
    print("="*80)
    print("🔍 KONTROLA CSV SOUBORU - POŽADOVANÉ SLOUPCE")
    print("="*80)
    print(f"\n📄 Soubor: {csv_path}\n")
    
    # Načtení CSV
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ CSV úspěšně načteno: {len(df)} řádků, {len(df.columns)} sloupců\n")
    except Exception as e:
        print(f"❌ CHYBA při načítání CSV: {e}")
        sys.exit(1)
    
    # Kontrola všech požadovaných sloupců
    missing_columns = []
    invalid_columns = []
    warning_columns = []
    valid_columns = []
    
    print("-"*80)
    print("📋 KONTROLA SLOUPCŮ:")
    print("-"*80)
    
    for col_name, col_config in REQUIRED_COLUMNS.items():
        print(f"\n🔹 {col_name}")
        print(f"   Popis: {col_config['description']}")
        print(f"   Typ: {col_config['type']}, Očekávaný rozsah: {col_config['range']}")
        
        # Kontrola existence
        exists_check = check_column_exists(df, col_name)
        print(f"   Existence: {exists_check['message']}")
        
        if not exists_check['exists']:
            missing_columns.append(col_name)
            continue
        
        # Kontrola validity dat
        validity_check = check_data_validity(df, col_name, col_config)
        
        print(f"   {validity_check['range_info']}")
        print(f"   Unikátních hodnot: {validity_check['unique_values']}")
        print(f"   Prázdné hodnoty: {validity_check['null_percent']:.1f}%")
        
        # Výpis issues
        if validity_check['issues']:
            for issue in validity_check['issues']:
                print(f"   {issue}")
            invalid_columns.append(col_name)
        else:
            valid_columns.append(col_name)
        
        # Výpis warnings
        if validity_check['warnings']:
            for warning in validity_check['warnings']:
                print(f"   {warning}")
            if col_name not in invalid_columns:
                warning_columns.append(col_name)
    
    # Souhrnné výsledky
    print("\n" + "="*80)
    print("📊 SOUHRNNÉ VÝSLEDKY:")
    print("="*80)
    
    total_columns = len(REQUIRED_COLUMNS)
    print(f"\n✅ Validní sloupce: {len(valid_columns)}/{total_columns}")
    print(f"🟡 Sloupce s varováními: {len(warning_columns)}/{total_columns}")
    print(f"❌ Nevalidní sloupce: {len(invalid_columns)}/{total_columns}")
    print(f"❌ Chybějící sloupce: {len(missing_columns)}/{total_columns}")
    
    if valid_columns:
        print(f"\n✅ Validní: {', '.join(valid_columns)}")
    
    if warning_columns:
        print(f"\n🟡 S varováními: {', '.join(warning_columns)}")
    
    if invalid_columns:
        print(f"\n❌ Nevalidní: {', '.join(invalid_columns)}")
    
    if missing_columns:
        print(f"\n❌ Chybějící: {', '.join(missing_columns)}")
    
    # Závěrečné hodnocení
    print("\n" + "="*80)
    if len(missing_columns) == 0 and len(invalid_columns) == 0:
        print("✅ VÝSLEDEK: CSV soubor je VALIDNÍ!")
        if len(warning_columns) > 0:
            print(f"⚠️  Upozornění: {len(warning_columns)} sloupců má menší problémy (viz výše)")
    else:
        print("❌ VÝSLEDEK: CSV soubor NENÍ validní!")
        print(f"   Opravte {len(missing_columns)} chybějících a {len(invalid_columns)} nevalidních sloupců.")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Defaultní cesta k CSV
    default_csv = Path(__file__).parent.parent / "data" / "processed" / "techmania_with_weather_and_holidays.csv"
    
    # Možnost zadat cestu jako argument
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = str(default_csv)
    
    # Kontrola, že soubor existuje
    if not Path(csv_path).exists():
        print(f"❌ CHYBA: Soubor neexistuje: {csv_path}")
        print(f"\nPoužití: python check_csv_columns.py [cesta_k_csv]")
        print(f"Výchozí cesta: {default_csv}")
        sys.exit(1)
    
    check_csv_file(csv_path)
