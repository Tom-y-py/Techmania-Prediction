"""
Weather Processor - Zpracování weather dat z API a dopočítávání chybějících features.
"""

import pandas as pd
import numpy as np
from datetime import date as date_type
from typing import Dict, Optional
import sys
from pathlib import Path

# Přidat app do path
sys.path.append(str(Path(__file__).parent.parent.parent / 'app'))

try:
    from services import weather_service
    WEATHER_SERVICE_AVAILABLE = True
except ImportError:
    WEATHER_SERVICE_AVAILABLE = False


def estimate_precipitation_probability(precipitation: float) -> int:
    """
    Odhadne pravděpodobnost srážek na základě množství srážek.
    
    Args:
        precipitation: Množství srážek v mm
        
    Returns:
        Pravděpodobnost srážek v %
    """
    if precipitation > 5:
        return 90
    elif precipitation > 1:
        return 70
    elif precipitation > 0:
        return 50
    else:
        return 20


def get_weather_for_date(
    pred_date: date_type,
    historical_df: Optional[pd.DataFrame] = None
) -> Dict:
    """
    Získá kompletní weather data pro dané datum.
    
    Kombinuje:
    - Data z Weather API (základní aktuální data)
    - Dopočítané features z historických dat (apparent temp, sunshine, etc.)
    
    Args:
        pred_date: Datum pro predikci
        historical_df: Historická data pro dopočítání chybějících features
        
    Returns:
        Dict s kompletními weather features
        
    Raises:
        RuntimeError: Pokud weather API není dostupné nebo vrátí chybu
    """
    if not WEATHER_SERVICE_AVAILABLE:
        raise RuntimeError("Weather service not available")
    
    try:
        # Získat základní weather data z API
        weather_info = weather_service.get_weather(pred_date)
        
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
            'wind_speed': weather_info.get('wind_speed_max'),
            'wind_gusts_max': weather_info.get('wind_gusts_max'),
            'is_rainy': int(weather_info.get('is_rainy', False)),
            'is_snowy': int(weather_info.get('is_snowy', False)),
            'is_windy': int(weather_info.get('is_windy', False)),
            'is_nice_weather': int(weather_info.get('is_nice_weather', False)),
            'weather_description': weather_info.get('weather_description', 'N/A'),
        }
        
        # Odhadnout precipitation probability
        weather_data['precipitation_probability'] = estimate_precipitation_probability(
            weather_data.get('precipitation', 0)
        )
        
        # Doplnit chybějící features z historických dat
        weather_data = fill_missing_weather_features(weather_data, historical_df, pred_date)
        
        return weather_data
        
    except Exception as e:
        raise RuntimeError(f"Failed to get weather data for {pred_date}: {e}")


def fill_missing_weather_features(
    weather_data: Dict,
    historical_df: Optional[pd.DataFrame],
    pred_date: date_type
) -> Dict:
    """
    Doplní chybějící weather features z historických dat.
    
    Features jako apparent_temp, wind_direction, sunshine, etc. nejsou dostupné
    v weather API, takže je dopočítáme z podobných dnů v historických datech.
    
    Args:
        weather_data: Základní weather data z API
        historical_df: Historická data
        pred_date: Datum predikce
        
    Returns:
        Dict s doplněnými weather features
    """
    if historical_df is None or len(historical_df) == 0:
        # Žádná historická data - nastavit NaN pro chybějící features
        weather_data.update({
            'apparent_temp_max': np.nan,
            'apparent_temp_min': np.nan,
            'apparent_temp_mean': np.nan,
            'wind_direction': np.nan,
            'sunshine_duration': np.nan,
            'daylight_duration': np.nan,
            'sunshine_ratio': np.nan,
            'cloud_cover_percent': np.nan,
            'feels_like_delta': np.nan,
            'weather_forecast_confidence': np.nan,
            'temperature_trend_3d': np.nan,
            'is_weather_improving': 0,
        })
        return weather_data
    
    # Najít podobné dny v historických datech (±15 dní)
    pred_month = pred_date.month
    pred_day = pred_date.day
    
    df_hist = historical_df[historical_df['date'] < pd.to_datetime(pred_date)].copy()
    
    if len(df_hist) == 0:
        # Žádná historická data před tímto datem
        weather_data.update({
            'apparent_temp_max': np.nan,
            'apparent_temp_min': np.nan,
            'apparent_temp_mean': np.nan,
            'wind_direction': np.nan,
            'sunshine_duration': np.nan,
            'daylight_duration': np.nan,
            'sunshine_ratio': np.nan,
            'cloud_cover_percent': np.nan,
            'feels_like_delta': np.nan,
            'weather_forecast_confidence': np.nan,
            'temperature_trend_3d': np.nan,
            'is_weather_improving': 0,
        })
        return weather_data
    
    df_hist['month'] = df_hist['date'].dt.month
    df_hist['day'] = df_hist['date'].dt.day
    
    # Najít podobné dny (±15 dní v měsíci)
    similar = df_hist[
        ((df_hist['month'] == pred_month) & 
         (abs(df_hist['day'] - pred_day) <= 15)) |
        ((pred_month == 1) & (df_hist['month'] == 12) & (df_hist['day'] >= 17)) |
        ((pred_month == 12) & (df_hist['month'] == 1) & (df_hist['day'] <= 15))
    ]
    
    # Fallbacky pokud není dost podobných dnů
    if len(similar) < 10:
        similar = df_hist[df_hist['month'] == pred_month]
    if len(similar) < 5:
        similar = df_hist
    
    # Doplnit features mediánem z podobných dnů
    def get_median(col_name):
        """Helper pro získání mediánu, pokud sloupec existuje"""
        if col_name in similar.columns and len(similar) > 0:
            return similar[col_name].median()
        return np.nan
    
    # Apparent temperature
    weather_data['apparent_temp_max'] = get_median('apparent_temp_max')
    weather_data['apparent_temp_min'] = get_median('apparent_temp_min')
    weather_data['apparent_temp_mean'] = get_median('apparent_temp_mean')
    
    # Wind
    weather_data['wind_direction'] = get_median('wind_direction')
    
    # Sunshine
    weather_data['sunshine_duration'] = get_median('sunshine_duration')
    weather_data['daylight_duration'] = get_median('daylight_duration')
    weather_data['sunshine_ratio'] = get_median('sunshine_ratio')
    
    # Precipitation probability - upřesnit z historických dat
    if 'precipitation_probability' in similar.columns and len(similar) > 0:
        hist_precip_prob = similar['precipitation_probability'].median()
        if pd.notna(hist_precip_prob):
            # Průměr mezi odhadem a historií
            weather_data['precipitation_probability'] = (
                weather_data['precipitation_probability'] + hist_precip_prob
            ) / 2
    
    # Další odvozené features
    weather_data['cloud_cover_percent'] = get_median('cloud_cover_percent')
    weather_data['feels_like_delta'] = get_median('feels_like_delta')
    weather_data['weather_forecast_confidence'] = get_median('weather_forecast_confidence')
    weather_data['temperature_trend_3d'] = get_median('temperature_trend_3d')
    
    # Binary features
    is_improving = get_median('is_weather_improving')
    weather_data['is_weather_improving'] = int(is_improving) if pd.notna(is_improving) else 0
    
    return weather_data
