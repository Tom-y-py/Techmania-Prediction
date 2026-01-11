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


def handle_missing_features(
    X: pd.DataFrame,
    feature_cols: List[str],
    historical_df: pd.DataFrame,
    start_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """
    Doplní chybějící features mediánem z historických dat.
    
    Args:
        X: DataFrame s features (může obsahovat NaN)
        feature_cols: Seznam požadovaných features
        historical_df: Historická data pro výpočet mediánu
        start_date: Datum od kterého jsou nová data (pro filtrování historie)
        
    Returns:
        DataFrame s doplněnými features
    """
    X_filled = X.copy()
    
    # Projít všechny požadované features
    for col in feature_cols:
        if col not in X_filled.columns:
            # Sloupec chybí úplně - přidat s nulami
            X_filled[col] = 0
            continue
        
        # Doplnit NaN hodnoty
        if X_filled[col].isna().any():
            if pd.api.types.is_numeric_dtype(X_filled[col]):
                # Vypočítat medián z historických dat
                if start_date is not None and col in historical_df.columns:
                    historical_median = historical_df[
                        historical_df['date'] < start_date
                    ][col].tail(90).median()
                    
                    if pd.isna(historical_median):
                        historical_median = historical_df[col].median()
                    
                    if not pd.isna(historical_median):
                        X_filled[col] = X_filled[col].fillna(historical_median)
                    else:
                        X_filled[col] = X_filled[col].fillna(0)
                else:
                    X_filled[col] = X_filled[col].fillna(0)
            else:
                X_filled[col] = X_filled[col].fillna(0)
    
    # Finální fillna pro jistotu
    X_filled = X_filled.fillna(0)
    
    return X_filled


def prepare_features_for_prediction(
    df: pd.DataFrame,
    feature_cols: List[str],
    historical_df: Optional[pd.DataFrame] = None,
    start_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """
    Připraví features pro predikci.
    
    Kroky:
    1. Vybere požadované sloupce
    2. Konvertuje na numeric
    3. Doplní chybějící hodnoty
    4. Vrátí DataFrame ve správném pořadí sloupců
    
    Args:
        df: DataFrame s daty (po feature engineering)
        feature_cols: Seznam požadovaných features
        historical_df: Historická data pro doplnění chybějících hodnot
        start_date: Datum od kterého jsou nová data
        
    Returns:
        DataFrame připravený pro model.predict()
    """
    # Vybrat dostupné features
    available_features = [col for col in feature_cols if col in df.columns]
    X = df[available_features].copy()
    
    # Konvertovat na numeric
    X = convert_to_numeric(X)
    
    # Doplnit chybějící features
    if historical_df is not None:
        X = handle_missing_features(X, feature_cols, historical_df, start_date)
    else:
        # Bez historických dat - jen fillna
        for col in feature_cols:
            if col not in X.columns:
                X[col] = 0
        X = X.fillna(0)
    
    # Ujistit se, že máme všechny features ve správném pořadí
    X = X[feature_cols]
    
    return X


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
        # Fallback - použít default hodnotu
        X_with_trend['predicted_google_trend'] = X.get('google_trend', 50.0)
        return X_with_trend
    
    # Získat trend features z plného DataFrame
    df_for_trend = df_full[df_full['date'].isin(pred_dates)]
    
    # Odstranit duplikáty - vzít pouze poslední výskyt každého data
    df_for_trend = df_for_trend.drop_duplicates(subset=['date'], keep='last')
    
    available_trend_features = [f for f in trend_features if f in df_for_trend.columns]
    
    if not available_trend_features:
        # Žádné trend features - použít fallback
        X_with_trend['predicted_google_trend'] = 50.0
        return X_with_trend
    
    X_trend = df_for_trend[available_trend_features].copy()
    
    # Konvertovat na numeric
    for col in X_trend.columns:
        if X_trend[col].dtype == 'object':
            X_trend[col] = pd.to_numeric(X_trend[col], errors='coerce')
    X_trend = X_trend.fillna(0)
    
    # Predikovat
    predicted_trends = google_trend_predictor.predict(X_trend.values)
    X_with_trend['predicted_google_trend'] = predicted_trends
    
    return X_with_trend
