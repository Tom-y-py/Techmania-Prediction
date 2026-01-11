"""
Model Predictor - Predikce z jednotlivých modelů a ensemble.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, Tuple


def should_use_catboost(is_weekend: bool, is_holiday: bool) -> bool:
    """
    Rozhodne, zda použít CatBoost v ensemble.
    
    CatBoost systematicky přestřeluje všední dny, ale funguje dobře
    na víkendy a svátky.
    
    Args:
        is_weekend: Je víkend?
        is_holiday: Je svátek?
        
    Returns:
        True pokud použít CatBoost, False jinak
    """
    return is_weekend or is_holiday


def predict_with_models(
    X: pd.DataFrame,
    models_dict: Dict
) -> Dict[str, np.ndarray]:
    """
    Provede predikci ze všech tří modelů.
    
    Args:
        X: DataFrame s features
        models_dict: Dict s modely ('lgb', 'xgb', 'cat')
        
    Returns:
        Dict s predikcemi: {'lightgbm': array, 'xgboost': array, 'catboost': array}
    """
    # LightGBM
    lgb_model = models_dict['lgb']
    try:
        lgb_preds = lgb_model.predict(X, num_iteration=lgb_model.best_iteration)
    except:
        lgb_preds = lgb_model.predict(X)
    
    # XGBoost
    xgb_model = models_dict['xgb']
    dmatrix = xgb.DMatrix(X)
    xgb_preds = xgb_model.predict(dmatrix)
    
    # CatBoost
    cat_model = models_dict['cat']
    cat_preds = cat_model.predict(X)
    
    return {
        'lightgbm': lgb_preds,
        'xgboost': xgb_preds,
        'catboost': cat_preds
    }


def ensemble_prediction(
    predictions: Dict[str, np.ndarray],
    weights: Dict,
    is_weekend: np.ndarray,
    is_holiday: np.ndarray,
    ensemble_type: str = 'weighted',
    meta_model = None
) -> np.ndarray:
    """
    Provede ensemble predikci.
    
    Podporuje:
    - 'weighted': Vážený průměr (s/bez CatBoost podle typu dne)
    - 'stacking': Meta-model
    - 'single_lgb': Pouze LightGBM
    
    Args:
        predictions: Dict s predikcemi z jednotlivých modelů
        weights: Dict s váhami modelů
        is_weekend: Boolean array - je víkend?
        is_holiday: Boolean array - je svátek?
        ensemble_type: Typ ensemble ('weighted', 'stacking', 'single_lgb')
        meta_model: Meta-model pro stacking (pokud ensemble_type=='stacking')
        
    Returns:
        Array s ensemble predikcemi
    """
    lgb_preds = predictions['lightgbm']
    xgb_preds = predictions['xgboost']
    cat_preds = predictions['catboost']
    
    if ensemble_type == 'single_lgb':
        # SINGLE: Pouze LightGBM
        return lgb_preds
    
    elif ensemble_type == 'stacking' and meta_model is not None:
        # STACKING: Meta-model
        meta_features = np.column_stack([lgb_preds, xgb_preds, cat_preds])
        return meta_model.predict(meta_features)
    
    else:
        # WEIGHTED: Vážený průměr
        ensemble_preds = np.zeros_like(lgb_preds)
        
        # Rozdělit na všední dny a víkendy/svátky
        use_cat = is_weekend | is_holiday
        
        # Extrahovat váhy
        if isinstance(weights, dict):
            w_lgb = weights.get('LightGBM', weights.get('lgb', 0.33))
            w_xgb = weights.get('XGBoost', weights.get('xgb', 0.33))
            w_cat = weights.get('CatBoost', weights.get('cat', 0.34))
        else:
            # Legacy format: array
            w_lgb = weights[0]
            w_xgb = weights[1]
            w_cat = weights[2]
        
        # Víkendy/svátky: všechny 3 modely
        if np.any(use_cat):
            ensemble_preds[use_cat] = (
                w_lgb * lgb_preds[use_cat] +
                w_xgb * xgb_preds[use_cat] +
                w_cat * cat_preds[use_cat]
            )
        
        # Všední dny: jen LightGBM + XGBoost
        if np.any(~use_cat):
            w_sum = w_lgb + w_xgb
            if w_sum > 0:
                ensemble_preds[~use_cat] = (
                    (w_lgb / w_sum) * lgb_preds[~use_cat] +
                    (w_xgb / w_sum) * xgb_preds[~use_cat]
                )
            else:
                # Fallback
                ensemble_preds[~use_cat] = (
                    0.5 * lgb_preds[~use_cat] +
                    0.5 * xgb_preds[~use_cat]
                )
        
        return ensemble_preds


def get_effective_weights(
    weights: Dict,
    use_catboost: bool
) -> Dict[str, float]:
    """
    Vypočítá efektivní váhy použité v ensemble.
    
    Args:
        weights: Původní váhy
        use_catboost: Zda byl použit CatBoost
        
    Returns:
        Dict s efektivními váhami
    """
    if isinstance(weights, dict):
        w_lgb = weights.get('LightGBM', weights.get('lgb', 0.33))
        w_xgb = weights.get('XGBoost', weights.get('xgb', 0.33))
        w_cat = weights.get('CatBoost', weights.get('cat', 0.34))
    else:
        w_lgb = weights[0]
        w_xgb = weights[1]
        w_cat = weights[2]
    
    if use_catboost:
        return {
            'lightgbm': float(w_lgb),
            'xgboost': float(w_xgb),
            'catboost': float(w_cat)
        }
    else:
        # CatBoost nepoužit - přenormalizovat
        w_sum = w_lgb + w_xgb
        if w_sum > 0:
            return {
                'lightgbm': float(w_lgb / w_sum),
                'xgboost': float(w_xgb / w_sum),
                'catboost': 0.0
            }
        else:
            return {
                'lightgbm': 0.5,
                'xgboost': 0.5,
                'catboost': 0.0
            }
