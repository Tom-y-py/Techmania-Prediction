"""
Confidence Calculator - Výpočet confidence intervalů.
"""

import numpy as np
from typing import Tuple, Dict, Optional


def calculate_confidence_interval(
    prediction: float,
    is_weekend: bool,
    is_holiday: bool,
    historical_mae: Optional[Dict[str, float]] = None,
    model_predictions: Optional[Dict[str, float]] = None
) -> Tuple[int, int]:
    """
    Vypočítá 95% confidence interval pro predikci.
    
    Dvě metody:
    1. Z historical MAE (preferováno) - realističtější
    2. Z variance modelů (fallback)
    
    Args:
        prediction: Predikovaná hodnota
        is_weekend: Je víkend?
        is_holiday: Je svátek?
        historical_mae: Dict s MAE pro weekday/weekend
        model_predictions: Dict s predikcemi jednotlivých modelů
        
    Returns:
        Tuple (lower_bound, upper_bound)
    """
    if historical_mae is not None:
        # Metoda 1: Z historical MAE (lepší)
        if is_weekend or is_holiday:
            mae = historical_mae.get('weekend', historical_mae.get('weekday', 100))
        else:
            mae = historical_mae.get('weekday', 100)
        
        # CI = predikce ± 1.96 * MAE (95% confidence)
        confidence_lower = int(max(50, prediction - 1.96 * mae))
        confidence_upper = int(prediction + 1.96 * mae)
        
    elif model_predictions is not None:
        # Metoda 2: Z variance modelů (fallback)
        predictions_list = list(model_predictions.values())
        model_std = np.std(predictions_list)
        
        confidence_lower = int(max(50, prediction - 1.96 * model_std))
        confidence_upper = int(prediction + 1.96 * model_std)
        
    else:
        # Žádná data - použít ±20%
        confidence_lower = int(max(50, prediction * 0.8))
        confidence_upper = int(prediction * 1.2)
    
    return confidence_lower, confidence_upper


def calculate_confidence_intervals_batch(
    predictions: np.ndarray,
    is_weekend: np.ndarray,
    is_holiday: np.ndarray,
    lgb_preds: np.ndarray,
    xgb_preds: np.ndarray,
    cat_preds: np.ndarray,
    historical_mae: Optional[Dict[str, float]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vypočítá confidence intervaly pro batch predikcí.
    
    Args:
        predictions: Array s predikcemi
        is_weekend: Boolean array - je víkend?
        is_holiday: Boolean array - je svátek?
        lgb_preds: LightGBM predikce
        xgb_preds: XGBoost predikce
        cat_preds: CatBoost predikce
        historical_mae: Dict s MAE pro weekday/weekend
        
    Returns:
        Tuple (lower_bounds, upper_bounds) jako numpy arrays
    """
    n = len(predictions)
    lower_bounds = np.zeros(n, dtype=int)
    upper_bounds = np.zeros(n, dtype=int)
    
    for i in range(n):
        model_preds = {
            'lightgbm': lgb_preds[i],
            'xgboost': xgb_preds[i],
            'catboost': cat_preds[i]
        }
        
        lower, upper = calculate_confidence_interval(
            prediction=predictions[i],
            is_weekend=bool(is_weekend[i]),
            is_holiday=bool(is_holiday[i]),
            historical_mae=historical_mae,
            model_predictions=model_preds
        )
        
        lower_bounds[i] = lower
        upper_bounds[i] = upper
    
    return lower_bounds, upper_bounds
