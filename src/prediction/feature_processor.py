"""
Feature Processor - Příprava features pro predikci.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def convert_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Konvertuje object sloupce na numeric.
    
    Args:
        df: DataFrame s features
        
    Returns:
        DataFrame s numeric sloupci
    """
    df_numeric = df.copy()
    
    for col in df_numeric.columns:
        if df_numeric[col].dtype == 'object':
            df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
    
    return df_numeric


def add_google_trend_feature(
    X: pd.DataFrame,
    df_full: pd.DataFrame,
    pred_dates: pd.Series,
    google_trend_predictor,
    trend_features: List[str]
) -> pd.DataFrame:
    """
    Predikuje Google Trend a přidá jako feature.
    
    Args:
        X: DataFrame s features
        df_full: Plný DataFrame s časovými features
        pred_dates: Série s daty pro predikci
        google_trend_predictor: Model pro predikci Google Trend
        trend_features: Seznam features pro trend predictor
        
    Returns:
        DataFrame s doplněným 'predicted_google_trend' sloupcem
    """
    X_with_trend = X.copy()
    
    if google_trend_predictor is None:
        raise ValueError("Google trend predictor není dostupný.")
    
    # Získat trend features z plného DataFrame
    df_for_trend = df_full[df_full['date'].isin(pred_dates)]
    
    # Odstranit duplikáty - vzít pouze poslední výskyt každého data
    df_for_trend = df_for_trend.drop_duplicates(subset=['date'], keep='last')
    
    available_trend_features = [f for f in trend_features if f in df_for_trend.columns]
    
    if not available_trend_features:
        raise ValueError("Chybí dostupné trend featury pro predikci Google Trend.")
    
    X_trend = df_for_trend[available_trend_features].copy()
    
    # Konvertovat na numeric
    for col in X_trend.columns:
        if X_trend[col].dtype == 'object':
            X_trend[col] = pd.to_numeric(X_trend[col], errors='coerce')
    nan_cols = X_trend.columns[X_trend.isna().any()].tolist()
    if nan_cols:
        raise ValueError(
            f"Trend feature matrix obsahuje NaN hodnoty ve sloupcích: {nan_cols}"
        )
    
    # Predikovat
    predicted_trends = google_trend_predictor.predict(X_trend.values)
    X_with_trend['predicted_google_trend'] = predicted_trends
    
    return X_with_trend
