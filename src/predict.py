"""
Prediction Module - Použití natrénovaných ensemble modelů
"""

import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
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
        models = {
            'lgb': joblib.load('models/lightgbm_model.pkl'),
            'prophet': joblib.load('models/prophet_model.pkl'),
            'nn': tf.keras.models.load_model('models/neural_network_model.h5'),
            'scaler_X': joblib.load('models/scaler_X.pkl'),
            'scaler_y': joblib.load('models/scaler_y.pkl'),
            'weights': joblib.load('models/ensemble_weights.pkl'),
            'feature_cols': joblib.load('models/feature_columns.pkl')
        }
        print("✅ Models loaded successfully!")
        return models
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("   Please train the models first by running: python src/ensemble_model.py")
        return None


def predict_single_date(date, models_dict, historical_df=None, seq_length=7):
    """
    Predikuje návštěvnost pro konkrétní datum
    
    Args:
        date: datetime nebo string ve formátu 'YYYY-MM-DD'
        models_dict: Dict s natrénovanými modely
        historical_df: DataFrame s historickými daty (pokud není, načte se)
        seq_length: Délka sekvence pro LSTM
        
    Returns:
        Dict s predikcemi a detaily
    """
    # Načíst historická data (potřebujeme pro lag features)
    if historical_df is None:
        df = pd.read_csv('data/raw/techmania_cleaned_master.csv')
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
    X_pred = pred_row[available_features].fillna(0)
    
    # === Predikce z každého modelu ===
    
    # 1. LightGBM
    lgb_model = models_dict['lgb']
    try:
        lgb_pred = lgb_model.predict(X_pred, num_iteration=lgb_model.best_iteration)[0]
    except:
        lgb_pred = lgb_model.predict(X_pred)[0]
    
    # 2. Prophet
    prophet_model = models_dict['prophet']
    future = pd.DataFrame({'ds': [date]})
    prophet_forecast = prophet_model.predict(future)
    prophet_pred = prophet_forecast['yhat'].values[0]
    prophet_lower = prophet_forecast['yhat_lower'].values[0]
    prophet_upper = prophet_forecast['yhat_upper'].values[0]
    
    # Ošetření záporných hodnot
    prophet_pred = max(prophet_pred, 0)
    prophet_lower = max(prophet_lower, 0)
    prophet_upper = max(prophet_upper, 0)
    
    # 3. Neural Network (složitější - potřebujeme sekvenci)
    nn_model = models_dict['nn']
    scaler_X = models_dict['scaler_X']
    scaler_y = models_dict['scaler_y']
    
    # Vytvořit sekvenci posledních N dnů
    historical_data = df[df['date'] < date].tail(seq_length)
    
    if len(historical_data) >= seq_length:
        X_seq = historical_data[available_features].fillna(0)
        X_seq_scaled = scaler_X.transform(X_seq)
        X_pred_seq = X_seq_scaled.reshape(1, seq_length, -1)
        
        nn_pred_scaled = nn_model.predict(X_pred_seq, verbose=0)
        nn_pred = scaler_y.inverse_transform(nn_pred_scaled)[0][0]
    else:
        # Nemáme dostatek historických dat
        nn_pred = (lgb_pred + prophet_pred) / 2  # Fallback na průměr
    
    # === Ensemble ===
    weights = models_dict['weights']
    ensemble_pred = (
        weights[0] * lgb_pred +
        weights[1] * prophet_pred +
        weights[2] * nn_pred
    )
    
    # Zaokrouhlit na celé číslo
    ensemble_pred = int(round(max(ensemble_pred, 0)))
    
    # Confidence interval (aproximace z Prophet + variance modelů)
    model_std = np.std([lgb_pred, prophet_pred, nn_pred])
    confidence_lower = int(max(0, ensemble_pred - 1.96 * model_std))
    confidence_upper = int(ensemble_pred + 1.96 * model_std)
    
    result = {
        'date': date,
        'day_of_week': date.strftime('%A'),
        'ensemble_prediction': ensemble_pred,
        'confidence_interval': (confidence_lower, confidence_upper),
        'individual_predictions': {
            'lightgbm': int(round(lgb_pred)),
            'prophet': int(round(prophet_pred)),
            'neural_network': int(round(nn_pred))
        },
        'prophet_interval': (int(prophet_lower), int(prophet_upper)),
        'model_weights': {
            'lightgbm': float(weights[0]),
            'prophet': float(weights[1]),
            'neural_network': float(weights[2])
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
    df = pd.read_csv('data/raw/techmania_cleaned_master.csv')
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
                'prophet': pred['individual_predictions']['prophet'],
                'neural_network': pred['individual_predictions']['neural_network']
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
    print(f"   Prophet (váha {result['model_weights']['prophet']:.1%}): {result['individual_predictions']['prophet']} návštěvníků")
    print(f"   Neural Net (váha {result['model_weights']['neural_network']:.1%}): {result['individual_predictions']['neural_network']} návštěvníků")
    
    print(f"\n📈 Prophet interval: [{result['prophet_interval'][0]} - {result['prophet_interval'][1]}]")
    
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
    
    # Příklad 1: Predikce pro konkrétní datum
    print("\n📅 Příklad 1: Predikce pro konkrétní datum")
    date = '2026-02-14'  # Valentýn
    result = predict_single_date(date, models)
    print_prediction(result)
    
    # Příklad 2: Predikce pro další týden
    print("\n📅 Příklad 2: Predikce pro příští týden")
    from datetime import date as dt_date, timedelta
    
    today = dt_date.today()
    next_week = today + timedelta(days=7)
    
    predictions = predict_date_range(today, next_week, models)
    print("\n" + str(predictions))
    
    # Uložit výsledky
    predictions.to_csv('predictions_next_week.csv', index=False)
    print("\n💾 Predictions saved to: predictions_next_week.csv")
    
    print("\n" + "=" * 60)
    print("✅ PREDICTION COMPLETE!")
    print("=" * 60)


if __name__ == '__main__':
    main()
