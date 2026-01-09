"""
Prediction Module - Použití natrénovaných ensemble modelů
"""

import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from feature_engineering import create_features


def load_models():
    """
    Načte všechny natrénované modely
    
    Returns:
        Dict s modely a pomocnými objekty
    """
    print("📦 Loading models...")
    
    try:
        import os
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        
        models = {
            'lgb': joblib.load(os.path.join(models_dir, 'lightgbm_model.pkl')),
            'xgb': joblib.load(os.path.join(models_dir, 'xgboost_model.pkl')),
            'cat': joblib.load(os.path.join(models_dir, 'catboost_model.pkl')),
            'weights': joblib.load(os.path.join(models_dir, 'ensemble_weights.pkl')),
            'feature_cols': joblib.load(os.path.join(models_dir, 'feature_columns.pkl'))
        }
        print("✅ Models loaded successfully!")
        return models
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("   Please train the models first by running: python src/ensemble_model.py")
        return None


def predict_single_date(date, models_dict, historical_df=None):
    """
    Predikuje návštěvnost pro konkrétní datum
    
    Args:
        date: datetime nebo string ve formátu 'YYYY-MM-DD'
        models_dict: Dict s natrénovanými modely
        historical_df: DataFrame s historickými daty (pokud není, načte se)
        
    Returns:
        Dict s predikcemi a detaily
    """
    # Načíst historická data (potřebujeme pro lag features)
    if historical_df is None:
        import os
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'techmania_cleaned_master.csv')
        df = pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['date'])
    else:
        df = historical_df.copy()
    
    # Přidat nový řádek pro predikci
    if isinstance(date, str):
        date = pd.to_datetime(date)
    
    new_row = pd.DataFrame({
        'date': [date],
        'total_visitors': [np.nan],
        'school_visitors': [np.nan],
        'public_visitors': [np.nan],
        'extra': [None],
        'opening_hours': [None]
    })
    
    df = pd.concat([df, new_row], ignore_index=True)
    
    # Feature engineering
    df = create_features(df)
    
    # Vybrat poslední řádek (náš prediction date)
    feature_cols = models_dict['feature_cols']
    
    # Najít společné sloupce
    available_features = [col for col in feature_cols if col in df.columns]
    
    pred_row = df[df['date'] == date]
    
    # Pro chybějící features použijeme mediánové hodnoty z historických dat
    X_pred = pred_row[available_features].copy()
    for col in available_features:
        if X_pred[col].isna().any():
            # Použij mediánovou hodnotu z posledních 90 dní historických dat
            historical_median = df[df['date'] < date][col].tail(90).median()
            if pd.isna(historical_median):
                historical_median = 0
            X_pred[col] = X_pred[col].fillna(historical_median)
    
    # === Predikce z každého modelu ===
    
    # 1. LightGBM
    lgb_model = models_dict['lgb']
    try:
        lgb_pred = lgb_model.predict(X_pred, num_iteration=lgb_model.best_iteration)[0]
    except:
        lgb_pred = lgb_model.predict(X_pred)[0]
    
    # 2. XGBoost
    xgb_model = models_dict['xgb']
    dmatrix = xgb.DMatrix(X_pred)
    xgb_pred = xgb_model.predict(dmatrix)[0]
    
    # 3. CatBoost
    cat_model = models_dict['cat']
    cat_pred = cat_model.predict(X_pred)[0]
    
    # === Ensemble ===
    weights = models_dict['weights']
    ensemble_pred = (
        weights[0] * lgb_pred +
        weights[1] * xgb_pred +
        weights[2] * cat_pred
    )
    
    # Zaokrouhlit na celé číslo
    ensemble_pred = int(round(max(ensemble_pred, 0)))
    
    # Confidence interval (aproximace z variance modelů)
    model_std = np.std([lgb_pred, xgb_pred, cat_pred])
    confidence_lower = int(max(0, ensemble_pred - 1.96 * model_std))
    confidence_upper = int(ensemble_pred + 1.96 * model_std)
    
    result = {
        'date': date,
        'day_of_week': date.strftime('%A'),
        'ensemble_prediction': ensemble_pred,
        'confidence_interval': (confidence_lower, confidence_upper),
        'individual_predictions': {
            'lightgbm': int(round(lgb_pred)),
            'xgboost': int(round(xgb_pred)),
            'catboost': int(round(cat_pred))
        },
        'model_weights': {
            'lightgbm': float(weights[0]),
            'xgboost': float(weights[1]),
            'catboost': float(weights[2])
        }
    }
    
    return result


def predict_date_range(start_date, end_date, models_dict):
    """
    Predikuje návštěvnost pro rozsah dat
    
    Args:
        start_date: Začátek období
        end_date: Konec období
        models_dict: Dict s natrénovanými modely
        
    Returns:
        DataFrame s predikcemi
    """
    # Načíst historická data jednou
    import os
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'techmania_cleaned_master.csv')
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Vytvořit rozsah dat
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    print(f"\n🔮 Predicting {len(date_range)} days...")
    
    results = []
    for date in date_range:
        try:
            pred = predict_single_date(date, models_dict, df)
            results.append({
                'date': pred['date'],
                'day_of_week': pred['day_of_week'],
                'prediction': pred['ensemble_prediction'],
                'lower_bound': pred['confidence_interval'][0],
                'upper_bound': pred['confidence_interval'][1],
                'lightgbm': pred['individual_predictions']['lightgbm'],
                'xgboost': pred['individual_predictions']['xgboost'],
                'catboost': pred['individual_predictions']['catboost']
            })
        except Exception as e:
            print(f"  ⚠️ Error predicting {date}: {e}")
    
    results_df = pd.DataFrame(results)
    print(f"✅ Predicted {len(results_df)} days successfully!")
    
    return results_df


def print_prediction(result):
    """
    Pěkně vypíše výsledek predikce
    
    Args:
        result: Dict s predikcí
    """
    print("\n" + "=" * 60)
    print(f"🔮 PREDIKCE PRO {result['date'].strftime('%d.%m.%Y')} ({result['day_of_week']})")
    print("=" * 60)
    
    print(f"\n🎯 ENSEMBLE PREDIKCE: {result['ensemble_prediction']} návštěvníků")
    print(f"   95% Confidence Interval: [{result['confidence_interval'][0]} - {result['confidence_interval'][1]}]")
    
    print(f"\n📊 Jednotlivé modely:")
    print(f"   LightGBM (váha {result['model_weights']['lightgbm']:.1%}): {result['individual_predictions']['lightgbm']} návštěvníků")
    print(f"   XGBoost (váha {result['model_weights']['xgboost']:.1%}): {result['individual_predictions']['xgboost']} návštěvníků")
    print(f"   CatBoost (váha {result['model_weights']['catboost']:.1%}): {result['individual_predictions']['catboost']} návštěvníků")
    
    print("=" * 60)


def main():
    """
    Demo použití predikčního modulu
    """
    print("\n" + "=" * 60)
    print("🎯 ENSEMBLE PREDICTION SYSTEM")
    print("=" * 60)
    
    # Načíst modely
    models = load_models()
    
    if models is None:
        return
    
    # Příklad 1: Predikce pro následující den
    print("\n📅 Příklad 1: Predikce pro následující den")
    
    from datetime import date as dt_date, timedelta
    next_day = dt_date.today() + timedelta(days=1)
    next_day_str = next_day.strftime('%Y-%m-%d')
    
    print(f"   Predikuji pro datum: {next_day_str}")
    result = predict_single_date(next_day_str, models)
    print_prediction(result)
    
    # Příklad 2: Predikce pro následujících 7 dní
    print("\n📅 Příklad 2: Predikce pro následujících 7 dní")
    
    start_date = dt_date.today() + timedelta(days=1)
    end_date = start_date + timedelta(days=6)
    
    print(f"   Období: {start_date.strftime('%Y-%m-%d')} až {end_date.strftime('%Y-%m-%d')}")
    
    predictions = predict_date_range(start_date, end_date, models)
    print("\n" + str(predictions))
    
    # Uložit výsledky
    import os
    output_file = os.path.join(os.path.dirname(__file__), '..', 'predictions_next_week.csv')
    predictions.to_csv(output_file, index=False)
    print(f"\n💾 Predictions saved to: {output_file}")
    
    print("\n" + "=" * 60)
    print("✅ PREDICTION COMPLETE!")
    print("=" * 60)


if __name__ == '__main__':
    main()
