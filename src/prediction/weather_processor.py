"""
Weather Processor - Zpracování weather dat z API a dopočítávání chybějících features.
"""

import pandas as pd
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
    
    Vrací pouze pole přímo z API + odvozené příznaky z WeatherService.
    
    Pole z API:
        temperature_max, temperature_min, temperature_mean
        precipitation, rain, snowfall, precipitation_hours
        daylight_duration
    
    Odvozené (dopočítává WeatherService):
        is_rainy, is_snowy, temperature_trend_3d, is_weather_improving
    
    Args:
        pred_date: Datum pro predikci
        
    Returns:
        Dict s weather features
        
    Raises:
        RuntimeError: Pokud weather API není dostupné nebo vrátí chybu
    """
    if not WEATHER_SERVICE_AVAILABLE:
        raise RuntimeError("Weather service not available")
    
    try:
        # Získat weather data z API
        weather_info = weather_service.get_weather(pred_date)
        
        weather_data = {
            # ── Přímo z API ──────────────────────────────────────────
            'temperature_max':      weather_info['temperature_max'],
            'temperature_min':      weather_info['temperature_min'],
            'temperature_mean':     weather_info['temperature_mean'],
            'precipitation':        weather_info['precipitation'],
            'rain':                 weather_info['rain'],
            'snowfall':             weather_info['snowfall'],
            'precipitation_hours':  weather_info['precipitation_hours'],
            'daylight_duration':    weather_info['daylight_duration'],
            # ── Odvozené (dopočítává WeatherService) ─────────────────
            'is_rainy':             int(weather_info.get('is_rainy', False)),
            'is_snowy':             int(weather_info.get('is_snowy', False)),
            'temperature_trend_3d': weather_info.get('temperature_trend_3d', 0.0),
            'is_weather_improving': weather_info.get('is_weather_improving', 0),
        }
        
        return weather_data
        
    except Exception as e:
        raise RuntimeError(f"Failed to get weather data for {pred_date}: {e}")
