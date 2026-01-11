"""
Data Loader - Načítání historických dat a templatu.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


def load_historical_data(
    include_weather: bool = True,
    include_holidays: bool = True
) -> pd.DataFrame:
    """
    Načte historická data z CSV souboru.
    
    Args:
        include_weather: Zda načíst data s počasím
        include_holidays: Zda načíst data se svátky
        
    Returns:
        DataFrame s historickými daty
    """
    script_dir = Path(__file__).parent.parent
    
    # Priorita: weather + holidays > weather > raw
    if include_weather and include_holidays:
        data_path = script_dir.parent / 'data' / 'processed' / 'techmania_with_weather_and_holidays.csv'
        if not data_path.exists():
            print("⚠️ techmania_with_weather_and_holidays.csv nenalezen, zkouším bez holidays")
            include_holidays = False
    
    if include_weather and not include_holidays:
        data_path = script_dir.parent / 'data' / 'processed' / 'techmania_with_weather.csv'
        if not data_path.exists():
            print("⚠️ techmania_with_weather.csv nenalezen, použiji raw data")
            include_weather = False
    
    if not include_weather:
        data_path = script_dir.parent / 'data' / 'raw' / 'techmania_cleaned_master.csv'
    
    try:
        df = pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['date'])
        print(f"   📂 Loaded historical data: {len(df)} rows (up to {df['date'].max().date()})")
        return df
    except FileNotFoundError as e:
        print(f"⚠️ Nepodařilo se načíst historická data: {e}")
        # Vrátit prázdný DataFrame s potřebnými sloupci
        df = pd.DataFrame(columns=['date', 'total_visitors'])
        df['date'] = pd.to_datetime(df['date'])
        return df


def load_template_2026() -> Optional[pd.DataFrame]:
    """
    Načte template pro rok 2026 s předvyplněnými holiday features.
    
    Returns:
        DataFrame s template daty nebo None pokud neexistuje
    """
    script_dir = Path(__file__).parent.parent
    template_path = script_dir.parent / 'data' / 'raw' / 'techmania_2026_template.csv'
    
    if not template_path.exists():
        print(f"   ⚠️ 2026 template not found at {template_path}")
        return None
    
    df_template = pd.read_csv(template_path)
    df_template['date'] = pd.to_datetime(df_template['date'])
    print(f"   📂 Loaded 2026 template: {len(df_template)} rows (holiday features)")
    return df_template


def combine_historical_and_new(
    df_historical: pd.DataFrame,
    df_new: pd.DataFrame
) -> pd.DataFrame:
    """
    Spojí historická data s novými daty pro predikci.
    
    Potřebné pro výpočet lag features a podobných dnů.
    
    Args:
        df_historical: Historická data
        df_new: Nová data pro predikci
        
    Returns:
        Kombinovaný DataFrame seřazený podle data
    """
    df_combined = pd.concat([df_historical, df_new], ignore_index=True)
    df_combined = df_combined.sort_values('date').reset_index(drop=True)
    return df_combined
