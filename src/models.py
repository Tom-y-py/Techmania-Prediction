"""
Modul pro trénování a práci s modely.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import joblib
from typing import Tuple, Dict, Any


def prepare_features(df: pd.DataFrame, target_col: str = 'total_visitors',
                     exclude_cols: list = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Připraví features a target pro trénování.
    
    Args:
        df: DataFrame s daty
        target_col: Název cílové proměnné
        exclude_cols: Sloupce k vyloučení z features
        
    Returns:
        Tuple (X, y)
    """
    if exclude_cols is None:
        exclude_cols = ['date', 'day_of_week', target_col]
    
    # Odstranit řádky s NaN v targetu
    df = df.dropna(subset=[target_col])
    
    # Vytvořit X a y
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].select_dtypes(include=[np.number])
    y = df[target_col]
    
    # Odstranit řádky s NaN ve features
    mask = ~X.isna().any(axis=1)
    X = X[mask]
    y = y[mask]
    
    return X, y


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series,
                       n_estimators: int = 100, max_depth: int = None,
                       random_state: int = 42) -> RandomForestRegressor:
    """
    Natrénuje Random Forest model.
    
    Args:
        X_train: Trénovací features
        y_train: Trénovací target
        n_estimators: Počet stromů
        max_depth: Maximální hloubka stromů
        random_state: Random seed
        
    Returns:
        Natrénovaný model
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    return model


def tune_random_forest(X_train: pd.DataFrame, y_train: pd.Series,
                      param_grid: Dict[str, Any] = None) -> RandomForestRegressor:
    """
    Provede hyperparameter tuning pro Random Forest.
    
    Args:
        X_train: Trénovací features
        y_train: Trénovací target
        param_grid: Grid parametrů pro hledání
        
    Returns:
        Nejlepší model
    """
    if param_grid is None:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10]
        }
    
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    grid_search = GridSearchCV(
        rf, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Nejlepší parametry: {grid_search.best_params_}")
    print(f"Nejlepší skóre: {-grid_search.best_score_:.2f}")
    
    return grid_search.best_estimator_


def save_model(model, filepath: str = '../models/best_model.pkl'):
    """
    Uloží model do souboru.
    
    Args:
        model: Model k uložení
        filepath: Cesta k souboru
    """
    joblib.dump(model, filepath)
    print(f"Model uložen do {filepath}")


def load_model(filepath: str = '../models/best_model.pkl'):
    """
    Načte model ze souboru.
    
    Args:
        filepath: Cesta k souboru
        
    Returns:
        Načtený model
    """
    model = joblib.load(filepath)
    print(f"Model načten z {filepath}")
    return model


if __name__ == "__main__":
    print("Modul models.py je připraven k použití")
