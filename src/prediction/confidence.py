"""
Confidence Calculator - Výpočet confidence intervalů.
"""

import numpy as np
from typing import Tuple, Dict


def calculate_confidence_interval(
    prediction: float,
    is_weekend: bool,
    is_holiday: bool,
    historical_mae: Dict[str, float]
) -> Tuple[int, int]:
    """
    Vypočítá 95% confidence interval pro predikci z historical MAE.
    
    Args:
        prediction: Predikovaná hodnota
        is_weekend: Je víkend?
        is_holiday: Je svátek?
        historical_mae: Dict s MAE pro weekday/weekend (POVINNÉ)
        
    Returns:
        Tuple (lower_bound, upper_bound)
        
    Raises:
        ValueError: Pokud chybí MAE pro daný typ dne
    """
    # Vybrat správné MAE podle typu dne
    if is_weekend or is_holiday:
        if 'weekend' not in historical_mae:
            raise ValueError(
                f"Missing 'weekend' MAE in historical_mae. "
                f"Available keys: {list(historical_mae.keys())}"
            )
        mae = historical_mae['weekend']
    else:
        if 'weekday' not in historical_mae:
            raise ValueError(
                f"Missing 'weekday' MAE in historical_mae. "
                f"Available keys: {list(historical_mae.keys())}"
            )
        mae = historical_mae['weekday']
    
    # CI = predikce ± 1.96 * MAE (95% confidence)
    confidence_lower = int(max(0, prediction - 1.96 * mae))
    confidence_upper = int(prediction + 1.96 * mae)
    
    return confidence_lower, confidence_upper


def calculate_confidence_intervals_batch(
    predictions: np.ndarray,
    is_weekend: np.ndarray,
    is_holiday: np.ndarray,
    historical_mae: Dict[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vypočítá confidence intervaly pro batch predikcí.
    
    Args:
        predictions: Array s predikcemi
        is_weekend: Boolean array - je víkend?
        is_holiday: Boolean array - je svátek?
        historical_mae: Dict s MAE pro weekday/weekend (POVINNÉ)
        
    Returns:
        Tuple (lower_bounds, upper_bounds) jako numpy arrays
        
    Raises:
        ValueError: Pokud chybí MAE pro daný typ dne
    """
    n = len(predictions)
    lower_bounds = np.zeros(n, dtype=int)
    upper_bounds = np.zeros(n, dtype=int)
    
    for i in range(n):
        lower, upper = calculate_confidence_interval(
            prediction=predictions[i],
            is_weekend=bool(is_weekend[i]),
            is_holiday=bool(is_holiday[i]),
            historical_mae=historical_mae
        )
        
        lower_bounds[i] = lower
        upper_bounds[i] = upper
    
    return lower_bounds, upper_bounds
