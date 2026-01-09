"""
Modul pro vytváření příznaků (feature engineering).
"""

import pandas as pd
import numpy as np
from typing import List


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vytvoří časové příznaky z datumu.
    
    Args:
        df: DataFrame s datem
        
    Returns:
        DataFrame s novými časovými features
    """
    df = df.copy()
    
    # Základní časové features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['quarter'] = df['date'].dt.quarter
    
    # Den v týdnu (0 = pondělí, 6 = neděle)
    df['day_of_week_num'] = df['date'].dt.dayofweek
    
    # Boolean features
    df['is_weekend'] = df['day_of_week_num'].isin([5, 6]).astype(int)
    df['is_holiday'] = df['extra'].notna().astype(int) if 'extra' in df.columns else 0
    
    # Sezónní features
    df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
    df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
    
    return df


def create_lag_features(df: pd.DataFrame, target_col: str = 'total_visitors', 
                       lags: List[int] = [1, 7, 14, 30]) -> pd.DataFrame:
    """
    Vytvoří lag features (hodnoty z minulosti).
    
    Args:
        df: DataFrame seřazený podle data
        target_col: Název cílového sloupce
        lags: Seznam lagů (dnů zpět)
        
    Returns:
        DataFrame s lag features
    """
    df = df.copy()
    
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    
    return df


def create_rolling_features(df: pd.DataFrame, target_col: str = 'total_visitors',
                           windows: List[int] = [7, 14, 30]) -> pd.DataFrame:
    """
    Vytvoří klouzavé průměry a další rolling features.
    
    Args:
        df: DataFrame seřazený podle data
        target_col: Název cílového sloupce
        windows: Seznam velikostí oken
        
    Returns:
        DataFrame s rolling features
    """
    df = df.copy()
    
    for window in windows:
        # Klouzavý průměr
        df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(
            window=window, min_periods=1
        ).mean()
        
        # Klouzavá směrodatná odchylka
        df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(
            window=window, min_periods=1
        ).std()
    
    return df


def create_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vytvoří všechny features najednou.
    
    Args:
        df: Původní DataFrame
        
    Returns:
        DataFrame se všemi features
    """
    df = create_time_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    
    return df


if __name__ == "__main__":
    # Test funkcí
    from data_processing import load_data
    
    df = load_data('../data/raw/techmania_cleaned_master.csv')
    df = create_all_features(df)
    print(f"Vytvořeno {len(df.columns)} features")
    print(df.columns.tolist())
