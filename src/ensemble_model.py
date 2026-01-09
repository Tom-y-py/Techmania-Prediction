"""
Ensemble Model - LightGBM + Prophet + Neural Network
Hlavní implementace pro trénování a kombinaci modelů
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from prophet import Prophet
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.optimize import minimize
import joblib
import warnings
warnings.filterwarnings('ignore')

from feature_engineering import create_features, split_data, get_feature_columns


def train_lightgbm(X_train, y_train, X_val, y_val):
    """
    Trénuje LightGBM model
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        
    Returns:
        Tuple[model, predictions]
    """
    print("\n" + "=" * 60)
    print("🌳 Training LightGBM...")
    print("=" * 60)
    
    # Parametry
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'max_depth': 8,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'random_state': 42
    }
    
    # Dataset pro LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Trénování
    model = lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=100)
        ]
    )
    
    # Predikce
    train_pred = model.predict(X_train, num_iteration=model.best_iteration)
    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    
    # Metriky
    print("\n=== LightGBM Results ===")
    print(f"Train MAE: {mean_absolute_error(y_train, train_pred):.2f}")
    print(f"Val MAE: {mean_absolute_error(y_val, val_pred):.2f}")
    print(f"Val RMSE: {np.sqrt(mean_squared_error(y_val, val_pred)):.2f}")
    print(f"Val R²: {r2_score(y_val, val_pred):.4f}")
    
    # Feature importance
    print("\n=== Top 10 Important Features ===")
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False).head(10)
    
    for idx, row in feature_importance.iterrows():
        print(f"  {row['feature']}: {row['importance']:.0f}")
    
    return model, val_pred


def train_prophet(train_df, val_df):
    """
    Trénuje Prophet model
    
    Args:
        train_df: Training DataFrame s columns ['date', 'total_visitors', 'is_holiday', 'extra']
        val_df: Validation DataFrame
        
    Returns:
        Tuple[model, predictions]
    """
    print("\n" + "=" * 60)
    print("📈 Training Prophet...")
    print("=" * 60)
    
    # Prophet vyžaduje speciální formát: 'ds' (datum) a 'y' (target)
    prophet_train = pd.DataFrame({
        'ds': train_df['date'],
        'y': train_df['total_visitors']
    })
    
    # Přidání svátků
    if 'is_holiday' in train_df.columns and 'extra' in train_df.columns:
        holidays = train_df[train_df['is_holiday'] == 1][['date', 'extra']].copy()
        holidays.columns = ['ds', 'holiday']
        holidays = holidays.dropna()
        holidays = holidays.drop_duplicates()
    else:
        holidays = None
    
    # Model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative',
        holidays=holidays,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10,
        interval_width=0.95
    )
    
    # Přidání custom seasonality
    model.add_seasonality(
        name='monthly',
        period=30.5,
        fourier_order=5
    )
    
    # Trénování
    print("  Fitting Prophet model...")
    model.fit(prophet_train)
    
    # Predikce na validaci
    future = pd.DataFrame({'ds': val_df['date']})
    forecast = model.predict(future)
    val_pred = forecast['yhat'].values
    
    # Ošetření záporných hodnot
    val_pred = np.maximum(val_pred, 0)
    
    # Metriky
    print("\n=== Prophet Results ===")
    print(f"Val MAE: {mean_absolute_error(val_df['total_visitors'], val_pred):.2f}")
    print(f"Val RMSE: {np.sqrt(mean_squared_error(val_df['total_visitors'], val_pred)):.2f}")
    print(f"Val R²: {r2_score(val_df['total_visitors'], val_pred):.4f}")
    
    return model, val_pred


def create_sequences(X, y, seq_length=7):
    """
    Vytvoří sekvence pro LSTM
    
    Args:
        X: Feature array
        y: Target array
        seq_length: Kolik dnů zpět model vidí
        
    Returns:
        Tuple[X_sequences, y_sequences]
    """
    Xs, ys = [], []
    for i in range(len(X) - seq_length):
        Xs.append(X[i:(i + seq_length)])
        ys.append(y[i + seq_length])
    return np.array(Xs), np.array(ys)


def train_neural_network(X_train, y_train, X_val, y_val, seq_length=7):
    """
    Trénuje Neural Network (LSTM) model
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        seq_length: Délka sekvence pro LSTM
        
    Returns:
        Tuple[model, predictions, scaler_X, scaler_y]
    """
    print("\n" + "=" * 60)
    print("🧠 Training Neural Network (LSTM)...")
    print("=" * 60)
    
    # Normalizace dat
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1)).flatten()
    
    # Vytvoření sekvencí
    print(f"  Creating sequences (length={seq_length})...")
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, seq_length)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_scaled, seq_length)
    
    print(f"  Train sequences: {X_train_seq.shape}")
    print(f"  Val sequences: {X_val_seq.shape}")
    
    # Model
    model = keras.Sequential([
        layers.LSTM(128, return_sequences=True, input_shape=(seq_length, X_train.shape[1])),
        layers.Dropout(0.3),
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])
    
    # Kompilace
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    print("\n  Model architecture:")
    model.summary()
    
    # Callbacks
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7,
        verbose=1
    )
    
    # Trénování
    print("\n  Training Neural Network...")
    history = model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=200,
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # Predikce
    val_pred_scaled = model.predict(X_val_seq, verbose=0)
    val_pred = scaler_y.inverse_transform(val_pred_scaled).flatten()
    
    # Upravit délku (kvůli sequences)
    y_val_adjusted = y_val.values[seq_length:]
    
    # Metriky
    print("\n=== Neural Network Results ===")
    print(f"Val MAE: {mean_absolute_error(y_val_adjusted, val_pred):.2f}")
    print(f"Val RMSE: {np.sqrt(mean_squared_error(y_val_adjusted, val_pred)):.2f}")
    print(f"Val R²: {r2_score(y_val_adjusted, val_pred):.4f}")
    
    return model, val_pred, scaler_X, scaler_y


def optimize_weights(predictions_dict, y_true):
    """
    Najde optimální váhy pro ensemble pomocí optimalizace
    
    Args:
        predictions_dict: Dict s predikcemi z každého modelu
        y_true: Skutečné hodnoty
        
    Returns:
        Optimální váhy (numpy array)
    """
    def ensemble_mae(weights):
        ensemble_pred = sum(w * pred for w, pred in zip(weights, predictions_dict.values()))
        return mean_absolute_error(y_true, ensemble_pred)
    
    # Počáteční váhy (rovnoměrné)
    n_models = len(predictions_dict)
    initial_weights = [1.0 / n_models] * n_models
    
    # Omezení: váhy musí být mezi 0 a 1, a součet = 1
    bounds = [(0, 1)] * n_models
    constraints = {'type': 'eq', 'fun': lambda w: sum(w) - 1}
    
    # Optimalizace
    result = minimize(
        ensemble_mae,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    return result.x


def create_ensemble(lgb_pred, prophet_pred, nn_pred, y_true, optimize=True):
    """
    Vytvoří ensemble z predikcí všech modelů
    
    Args:
        lgb_pred: LightGBM predikce
        prophet_pred: Prophet predikce
        nn_pred: Neural Network predikce
        y_true: Skutečné hodnoty
        optimize: Zda optimalizovat váhy nebo použít defaultní
        
    Returns:
        Tuple[ensemble_predictions, weights]
    """
    print("\n" + "=" * 60)
    print("🎯 Creating Ensemble...")
    print("=" * 60)
    
    # Ujistit se, že všechny predikce mají stejnou délku
    min_len = min(len(lgb_pred), len(prophet_pred), len(nn_pred))
    lgb_pred = lgb_pred[:min_len]
    prophet_pred = prophet_pred[:min_len]
    nn_pred = nn_pred[:min_len]
    y_true = y_true[:min_len]
    
    predictions = {
        'lightgbm': lgb_pred,
        'prophet': prophet_pred,
        'neural_net': nn_pred
    }
    
    if optimize:
        # Optimalizuj váhy na validačních datech
        weights = optimize_weights(predictions, y_true)
        print(f"\n=== Optimized Weights ===")
        print(f"LightGBM: {weights[0]:.3f}")
        print(f"Prophet: {weights[1]:.3f}")
        print(f"Neural Network: {weights[2]:.3f}")
    else:
        # Použij defaultní váhy
        weights = np.array([0.45, 0.30, 0.25])
        print(f"\n=== Default Weights ===")
        print(f"LightGBM: {weights[0]:.3f}")
        print(f"Prophet: {weights[1]:.3f}")
        print(f"Neural Network: {weights[2]:.3f}")
    
    # Finální predikce
    ensemble_pred = (
        weights[0] * lgb_pred +
        weights[1] * prophet_pred +
        weights[2] * nn_pred
    )
    
    # Metriky
    print("\n=== ENSEMBLE Results ===")
    print(f"Val MAE: {mean_absolute_error(y_true, ensemble_pred):.2f}")
    print(f"Val RMSE: {np.sqrt(mean_squared_error(y_true, ensemble_pred)):.2f}")
    print(f"Val R²: {r2_score(y_true, ensemble_pred):.4f}")
    
    # Porovnání s jednotlivými modely
    print("\n=== Comparison ===")
    print(f"LightGBM MAE: {mean_absolute_error(y_true, lgb_pred):.2f}")
    print(f"Prophet MAE: {mean_absolute_error(y_true, prophet_pred):.2f}")
    print(f"Neural Network MAE: {mean_absolute_error(y_true, nn_pred):.2f}")
    print(f"Ensemble MAE: {mean_absolute_error(y_true, ensemble_pred):.2f}")
    
    best_single = min(
        mean_absolute_error(y_true, lgb_pred),
        mean_absolute_error(y_true, prophet_pred),
        mean_absolute_error(y_true, nn_pred)
    )
    improvement = best_single - mean_absolute_error(y_true, ensemble_pred)
    improvement_pct = (improvement / best_single) * 100
    
    print(f"\n🎉 Improvement over best single model:")
    print(f"   {improvement:.2f} visitors ({improvement_pct:.1f}% better!)")
    
    return ensemble_pred, weights


def main():
    """
    Hlavní pipeline pro ensemble model
    """
    print("\n" + "=" * 70)
    print("🚀 ENSEMBLE MODEL TRAINING PIPELINE")
    print("=" * 70)
    
    # 1. Načíst data
    print("\n📂 Loading data...")
    df = pd.read_csv('data/raw/techmania_cleaned_master.csv')
    print(f"   Loaded {len(df)} records")
    print(f"   Date range: {df['date'].min()} - {df['date'].max()}")
    
    # 2. Feature engineering
    df = create_features(df)
    
    # 3. Split data
    train, val, test = split_data(df)
    
    # Připravit X, y
    feature_cols = get_feature_columns(df)
    
    X_train = train[feature_cols]
    y_train = train['total_visitors']
    X_val = val[feature_cols]
    y_val = val['total_visitors']
    
    # 4. Trénovat modely
    lgb_model, lgb_pred = train_lightgbm(X_train, y_train, X_val, y_val)
    
    prophet_model, prophet_pred = train_prophet(train, val)
    
    nn_model, nn_pred, scaler_X, scaler_y = train_neural_network(
        X_train, y_train, X_val, y_val, seq_length=7
    )
    
    # 5. Ensemble
    # NN má kratší predikce kvůli sequences (seq_length=7)
    seq_length = 7
    y_val_adjusted = y_val.values[seq_length:]
    lgb_pred_adjusted = lgb_pred[seq_length:]
    prophet_pred_adjusted = prophet_pred[seq_length:]
    
    ensemble_pred, weights = create_ensemble(
        lgb_pred_adjusted,
        prophet_pred_adjusted,
        nn_pred,
        y_val_adjusted,
        optimize=True
    )
    
    # 6. Uložit modely
    print("\n💾 Saving models...")
    import os
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(lgb_model, 'models/lightgbm_model.pkl')
    joblib.dump(prophet_model, 'models/prophet_model.pkl')
    nn_model.save('models/neural_network_model.h5')
    joblib.dump(scaler_X, 'models/scaler_X.pkl')
    joblib.dump(scaler_y, 'models/scaler_y.pkl')
    joblib.dump(weights, 'models/ensemble_weights.pkl')
    joblib.dump(feature_cols, 'models/feature_columns.pkl')
    
    print("\n✅ Models saved successfully!")
    print("   📁 models/lightgbm_model.pkl")
    print("   📁 models/prophet_model.pkl")
    print("   📁 models/neural_network_model.h5")
    print("   📁 models/scaler_X.pkl")
    print("   📁 models/scaler_y.pkl")
    print("   📁 models/ensemble_weights.pkl")
    print("   📁 models/feature_columns.pkl")
    
    print("\n" + "=" * 70)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 70)
    
    return {
        'lgb': lgb_model,
        'prophet': prophet_model,
        'nn': nn_model,
        'weights': weights,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'feature_cols': feature_cols
    }


if __name__ == '__main__':
    models = main()
