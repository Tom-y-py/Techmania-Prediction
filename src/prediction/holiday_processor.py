"""
Holiday Processor - Zpracování holiday dat z templatu nebo API.
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
    from services import holiday_service
    HOLIDAY_SERVICE_AVAILABLE = True
except ImportError:
    HOLIDAY_SERVICE_AVAILABLE = False


def get_holiday_from_template(
    pred_date: date_type,
    template_df: pd.DataFrame
) -> Optional[Dict]:
    """
    Získá holiday features z template DataFrame.
    
    Template obsahuje předvyplněné správné holiday features pro celý rok.
    
    Args:
        pred_date: Datum predikce
        template_df: DataFrame s template daty
        
    Returns:
        Dict s holiday features nebo None pokud datum není v templatu
    """
    pred_date_ts = pd.to_datetime(pred_date)
    template_row = template_df[template_df['date'] == pred_date_ts]
    
    if template_row.empty:
        return None
    
    tr = template_row.iloc[0]
    
    # Extrahovat všechny holiday features
    holiday_data = {
        'day_of_week': tr['day_of_week'] if pd.notna(tr.get('day_of_week')) else pred_date.strftime('%A'),
        'is_weekend': int(tr['is_weekend']) if pd.notna(tr.get('is_weekend')) else int(pred_date.weekday() >= 5),
        'is_holiday': int(tr['is_holiday']) if pd.notna(tr.get('is_holiday')) else 0,
        'nazvy_svatek': tr['nazvy_svatek'] if pd.notna(tr.get('nazvy_svatek')) else None,
        'is_any_school_break': int(tr['is_any_school_break']) if pd.notna(tr.get('is_any_school_break')) else 0,
        'school_break_type': tr['school_break_type'] if pd.notna(tr.get('school_break_type')) else None,
        'is_spring_break': int(tr['is_spring_break']) if pd.notna(tr.get('is_spring_break')) else 0,
        'is_autumn_break': int(tr['is_autumn_break']) if pd.notna(tr.get('is_autumn_break')) else 0,
        'is_winter_break': int(tr['is_winter_break']) if pd.notna(tr.get('is_winter_break')) else 0,
        'is_easter_break': int(tr['is_easter_break']) if pd.notna(tr.get('is_easter_break')) else 0,
        'is_halfyear_break': int(tr['is_halfyear_break']) if pd.notna(tr.get('is_halfyear_break')) else 0,
        'is_summer_holiday': int(tr['is_summer_holiday']) if pd.notna(tr.get('is_summer_holiday')) else 0,
        'days_to_next_break': int(tr['days_to_next_break']) if pd.notna(tr.get('days_to_next_break')) else 0,
        'days_from_last_break': int(tr['days_from_last_break']) if pd.notna(tr.get('days_from_last_break')) else 0,
        'is_week_before_break': int(tr['is_week_before_break']) if pd.notna(tr.get('is_week_before_break')) else 0,
        'is_week_after_break': int(tr['is_week_after_break']) if pd.notna(tr.get('is_week_after_break')) else 0,
        'season_exact': tr['season_exact'] if pd.notna(tr.get('season_exact')) else None,
        'week_position': tr['week_position'] if pd.notna(tr.get('week_position')) else None,
        'is_month_end': int(tr['is_month_end']) if pd.notna(tr.get('is_month_end')) else 0,
        'school_week_number': int(tr['school_week_number']) if pd.notna(tr.get('school_week_number')) else 0,
        'is_bridge_day': int(tr['is_bridge_day']) if pd.notna(tr.get('is_bridge_day')) else 0,
        'long_weekend_length': int(tr['long_weekend_length']) if pd.notna(tr.get('long_weekend_length')) else 0,
        'is_event': int(tr['is_event']) if pd.notna(tr.get('is_event')) else 0,
    }
    
    return holiday_data


def get_holiday_from_api(pred_date: date_type) -> Dict:
    """
    Získá holiday features z Holiday API.
    
    Fallback pokud template není dostupný.
    
    Args:
        pred_date: Datum predikce
        
    Returns:
        Dict s holiday features
    """
    if not HOLIDAY_SERVICE_AVAILABLE:
        # Minimální fallback bez API
        return {
            'day_of_week': pred_date.strftime('%A'),
            'is_weekend': int(pred_date.weekday() >= 5),
            'is_holiday': 0,
            'nazvy_svatek': None,
            'is_event': 0,
        }
    
    try:
        holiday_info = holiday_service.get_holiday_info(pred_date)
        
        holiday_data = {
            'day_of_week': pred_date.strftime('%A'),
            'is_weekend': int(pred_date.weekday() >= 5),
            'is_holiday': int(holiday_info.get('is_holiday', False)),
            'nazvy_svatek': holiday_info.get('holiday_name', None),
            'is_any_school_break': int(holiday_info.get('is_any_school_break', False)),
            'school_break_type': holiday_info.get('school_break_type', None),
            'is_spring_break': int(holiday_info.get('is_spring_break', False)),
            'is_autumn_break': int(holiday_info.get('is_autumn_break', False)),
            'is_winter_break': int(holiday_info.get('is_winter_break', False)),
            'is_easter_break': int(holiday_info.get('is_easter_break', False)),
            'is_halfyear_break': int(holiday_info.get('is_halfyear_break', False)),
            'is_summer_holiday': int(holiday_info.get('is_summer_holiday', False)),
            'days_to_next_break': holiday_info.get('days_to_next_break', 0),
            'days_from_last_break': holiday_info.get('days_from_last_break', 0),
            'is_week_before_break': int(holiday_info.get('is_week_before_break', False)),
            'is_week_after_break': int(holiday_info.get('is_week_after_break', False)),
            'season_exact': holiday_info.get('season_exact', None),
            'week_position': holiday_info.get('week_position', None),
            'is_month_end': int(holiday_info.get('is_month_end', False)),
            'school_week_number': holiday_info.get('school_week_number', 0),
            'is_bridge_day': int(holiday_info.get('is_bridge_day', False)),
            'long_weekend_length': holiday_info.get('long_weekend_length', 0),
            'is_event': 0,  # Default
        }
        
        return holiday_data
        
    except Exception as e:
        print(f"   ⚠️ Holiday API warning: {e}")
        # Minimální fallback
        return {
            'day_of_week': pred_date.strftime('%A'),
            'is_weekend': int(pred_date.weekday() >= 5),
            'is_holiday': 0,
            'nazvy_svatek': None,
            'is_event': 0,
        }


def get_holiday_features(
    pred_date: date_type,
    template_df: Optional[pd.DataFrame] = None
) -> Dict:
    """
    Získá kompletní holiday features pro dané datum.
    
    Priorita:
    1. Template (pokud je dostupný) - nejpřesnější
    2. Holiday API - fallback
    
    Args:
        pred_date: Datum predikce
        template_df: Template DataFrame (pokud je dostupný)
        
    Returns:
        Dict s kompletními holiday features
    """
    # Priorita 1: Template
    if template_df is not None:
        holiday_data = get_holiday_from_template(pred_date, template_df)
        if holiday_data is not None:
            print(f"   ✓ Using template holiday features for {pred_date}")
            return holiday_data
    
    # Priorita 2: Holiday API
    holiday_data = get_holiday_from_api(pred_date)
    
    # Doplnit Google Trends features (budou predikované později)
    holiday_data['google_trend'] = np.nan
    holiday_data['Mateřská_škola'] = np.nan
    holiday_data['Střední_škola'] = np.nan
    holiday_data['Základní_škola'] = np.nan
    
    return holiday_data
