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
    raise RuntimeError(
        "Heuristický odhad precipitation_probability je zakázaný. "
        "Použijte pouze hodnotu dodanou weather API."
    )


def get_weather_for_date(
    pred_date: date_type,
    historical_df: Optional[pd.DataFrame] = None
) -> Dict:
    """
    Získá kompletní weather data pro dané datum.
    
    Kombinuje data z Weather API do feature slovníku.
    
    Args:
        pred_date: Datum pro predikci
        
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
        
        # Použít pouze hodnotu z API; pokud není dostupná, držet missing flag.
        api_precip_prob = weather_info.get('precipitation_probability')
        weather_data['precipitation_probability'] = (
            float(api_precip_prob) if api_precip_prob is not None else np.nan
        )
        weather_data['precipitation_probability_missing'] = int(api_precip_prob is None)
        
        return weather_data
        
    except Exception as e:
        raise RuntimeError(f"Failed to get weather data for {pred_date}: {e}")
