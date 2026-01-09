"""
Modul pro evaluaci modelů a vizualizaci výsledků.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Any


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Vypočítá metriky pro vyhodnocení modelu.
    
    Args:
        y_true: Skutečné hodnoty
        y_pred: Predikované hodnoty
        
    Returns:
        Dictionary s metrikami
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape
    }
    
    return metrics


def print_metrics(metrics: Dict[str, float]):
    """
    Vytiskne metriky přehledně.
    
    Args:
        metrics: Dictionary s metrikami
    """
    print("=" * 50)
    print("METRIKY MODELU")
    print("=" * 50)
    for name, value in metrics.items():
        if name == 'MAPE':
            print(f"{name:10s}: {value:.2f}%")
        else:
            print(f"{name:10s}: {value:.2f}")
    print("=" * 50)


def plot_predictions(y_true: pd.Series, y_pred: np.ndarray, 
                    dates: pd.Series = None, title: str = "Predikce vs. Skutečnost"):
    """
    Vizualizuje predikce vs. skutečné hodnoty.
    
    Args:
        y_true: Skutečné hodnoty
        y_pred: Predikované hodnoty
        dates: Data (optional)
        title: Název grafu
    """
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Graf 1: Časová řada
    ax1 = axes[0]
    if dates is not None:
        ax1.plot(dates, y_true, label='Skutečnost', alpha=0.7)
        ax1.plot(dates, y_pred, label='Predikce', alpha=0.7)
    else:
        ax1.plot(y_true.values, label='Skutečnost', alpha=0.7)
        ax1.plot(y_pred, label='Predikce', alpha=0.7)
    
    ax1.set_title(title)
    ax1.set_xlabel('Datum' if dates is not None else 'Index')
    ax1.set_ylabel('Počet návštěvníků')
    ax1.legend()
    ax1.grid(True)
    
    # Graf 2: Scatter plot
    ax2 = axes[1]
    ax2.scatter(y_true, y_pred, alpha=0.5)
    
    # Ideální přímka (y = x)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideální predikce')
    
    ax2.set_xlabel('Skutečné hodnoty')
    ax2.set_ylabel('Predikované hodnoty')
    ax2.set_title('Scatter Plot: Predikce vs. Skutečnost')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feature_names: list, top_n: int = 20):
    """
    Vizualizuje důležitost features.
    
    Args:
        model: Natrénovaný model (musí mít atribut feature_importances_)
        feature_names: Názvy features
        top_n: Počet top features k zobrazení
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(12, 8))
    plt.title(f'Top {top_n} nejdůležitějších features')
    plt.barh(range(top_n), importances[indices])
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Důležitost')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def plot_residuals(y_true: pd.Series, y_pred: np.ndarray):
    """
    Vizualizuje residuals (chyby predikce).
    
    Args:
        y_true: Skutečné hodnoty
        y_pred: Predikované hodnoty
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histogram residuals
    axes[0].hist(residuals, bins=50, edgecolor='black')
    axes[0].set_title('Distribuce chyb (residuals)')
    axes[0].set_xlabel('Chyba')
    axes[0].set_ylabel('Četnost')
    axes[0].axvline(x=0, color='r', linestyle='--')
    
    # Residuals vs. predicted
    axes[1].scatter(y_pred, residuals, alpha=0.5)
    axes[1].set_title('Residuals vs. Predikované hodnoty')
    axes[1].set_xlabel('Predikované hodnoty')
    axes[1].set_ylabel('Chyba (residual)')
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Modul evaluation.py je připraven k použití")
