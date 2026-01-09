"""
Ensemble Model - LightGBM + XGBoost + CatBoost
Hlavní implementace pro trénování a kombinaci modelů
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
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


def train_xgboost(X_train, y_train, X_val, y_val):
    """
    Trénuje XGBoost model
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        
    Returns:
        Tuple[model, predictions]
    """
    print("\n" + "=" * 60)
    print("� Training XGBoost...")
    print("=" * 60)
    
    # Parametry
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    }
    
    # Dataset pro XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Trénování
    evals = [(dtrain, 'train'), (dval, 'valid')]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        evals=evals,
        early_stopping_rounds=100,
        verbose_eval=100
    )
    
    # Predikce
    train_pred = model.predict(dtrain)
    val_pred = model.predict(dval)
    
    # Metriky
    print("\n=== XGBoost Results ===")
    print(f"Train MAE: {mean_absolute_error(y_train, train_pred):.2f}")
    print(f"Val MAE: {mean_absolute_error(y_val, val_pred):.2f}")
    print(f"Val RMSE: {np.sqrt(mean_squared_error(y_val, val_pred)):.2f}")
    print(f"Val R²: {r2_score(y_val, val_pred):.4f}")
    
    # Feature importance
    print("\n=== Top 10 Important Features ===")
    importance = model.get_score(importance_type='gain')
    feature_importance = pd.DataFrame({
        'feature': list(importance.keys()),
        'importance': list(importance.values())
    }).sort_values('importance', ascending=False).head(10)
    
    for idx, row in feature_importance.iterrows():
        print(f"  {row['feature']}: {row['importance']:.0f}")
    
    return model, val_pred


def train_catboost(X_train, y_train, X_val, y_val):
    """
    Trénuje CatBoost model
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        
    Returns:
        Tuple[model, predictions]
    """
    print("\n" + "=" * 60)
    print("🐱 Training CatBoost...")
    print("=" * 60)
    
    # Model
    model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3,
        random_strength=0.1,
        bagging_temperature=0.2,
        od_type='Iter',
        od_wait=100,
        random_seed=42,
        verbose=100,
        task_type='CPU'
    )
    
    # Trénování
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
        verbose=100
    )
    
    # Predikce
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    # Metriky
    print("\n=== CatBoost Results ===")
    print(f"Train MAE: {mean_absolute_error(y_train, train_pred):.2f}")
    print(f"Val MAE: {mean_absolute_error(y_val, val_pred):.2f}")
    print(f"Val RMSE: {np.sqrt(mean_squared_error(y_val, val_pred)):.2f}")
    print(f"Val R²: {r2_score(y_val, val_pred):.4f}")
    
    # Feature importance
    print("\n=== Top 10 Important Features ===")
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    
    for idx, row in feature_importance.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    return model, val_pred


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


def create_ensemble(lgb_pred, xgb_pred, cat_pred, y_true, optimize=True):
    """
    Vytvoří ensemble z predikcí všech modelů
    
    Args:
        lgb_pred: LightGBM predikce
        xgb_pred: XGBoost predikce
        cat_pred: CatBoost predikce
        y_true: Skutečné hodnoty
        optimize: Zda optimalizovat váhy nebo použít defaultní
        
    Returns:
        Tuple[ensemble_predictions, weights]
    """
    print("\n" + "=" * 60)
    print("🎯 Creating Ensemble...")
    print("=" * 60)
    
    # Ujistit se, že všechny predikce mají stejnou délku
    min_len = min(len(lgb_pred), len(xgb_pred), len(cat_pred))
    lgb_pred = lgb_pred[:min_len]
    xgb_pred = xgb_pred[:min_len]
    cat_pred = cat_pred[:min_len]
    y_true = y_true[:min_len]
    
    predictions = {
        'lightgbm': lgb_pred,
        'xgboost': xgb_pred,
        'catboost': cat_pred
    }
    
    if optimize:
        # Optimalizuj váhy na validačních datech
        weights = optimize_weights(predictions, y_true)
        print(f"\n=== Optimized Weights ===")
        print(f"LightGBM: {weights[0]:.3f}")
        print(f"XGBoost: {weights[1]:.3f}")
        print(f"CatBoost: {weights[2]:.3f}")
    else:
        # Pokud není optimalizace, použijeme rovnoměrné váhy
        n_models = len(predictions)
        weights = np.array([1.0 / n_models] * n_models)
        print(f"\n=== Equal Weights (No Optimization) ===")
        print(f"LightGBM: {weights[0]:.3f}")
        print(f"XGBoost: {weights[1]:.3f}")
        print(f"CatBoost: {weights[2]:.3f}")
    
    # Finální predikce
    ensemble_pred = (
        weights[0] * lgb_pred +
        weights[1] * xgb_pred +
        weights[2] * cat_pred
    )
    
    # Metriky
    print("\n=== ENSEMBLE Results ===")
    print(f"Val MAE: {mean_absolute_error(y_true, ensemble_pred):.2f}")
    print(f"Val RMSE: {np.sqrt(mean_squared_error(y_true, ensemble_pred)):.2f}")
    print(f"Val R²: {r2_score(y_true, ensemble_pred):.4f}")
    
    # Porovnání s jednotlivými modely
    print("\n=== Comparison ===")
    print(f"LightGBM MAE: {mean_absolute_error(y_true, lgb_pred):.2f}")
    print(f"XGBoost MAE: {mean_absolute_error(y_true, xgb_pred):.2f}")
    print(f"CatBoost MAE: {mean_absolute_error(y_true, cat_pred):.2f}")
    print(f"Ensemble MAE: {mean_absolute_error(y_true, ensemble_pred):.2f}")
    
    best_single = min(
        mean_absolute_error(y_true, lgb_pred),
        mean_absolute_error(y_true, xgb_pred),
        mean_absolute_error(y_true, cat_pred)
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
    import os
    # Cesta relativně k src/ adresáři (kde běží skript)
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'techmania_cleaned_master.csv')
    df = pd.read_csv(data_path)
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
    
    xgb_model, xgb_pred = train_xgboost(X_train, y_train, X_val, y_val)
    
    cat_model, cat_pred = train_catboost(X_train, y_train, X_val, y_val)
    
    # 5. Ensemble
    ensemble_pred, weights = create_ensemble(
        lgb_pred,
        xgb_pred,
        cat_pred,
        y_val.values,
        optimize=True
    )
    
    # 6. Uložit modely
    print("\n💾 Saving models...")
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    joblib.dump(lgb_model, os.path.join(models_dir, 'lightgbm_model.pkl'))
    joblib.dump(xgb_model, os.path.join(models_dir, 'xgboost_model.pkl'))
    joblib.dump(cat_model, os.path.join(models_dir, 'catboost_model.pkl'))
    joblib.dump(weights, os.path.join(models_dir, 'ensemble_weights.pkl'))
    joblib.dump(feature_cols, os.path.join(models_dir, 'feature_columns.pkl'))
    
    print("\n✅ Models saved successfully!")
    print("   📁 models/lightgbm_model.pkl")
    print("   📁 models/xgboost_model.pkl")
    print("   📁 models/catboost_model.pkl")
    print("   📁 models/ensemble_weights.pkl")
    print("   📁 models/feature_columns.pkl")
    
    print("\n" + "=" * 70)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 70)
    
    return {
        'lgb': lgb_model,
        'xgb': xgb_model,
        'cat': cat_model,
        'weights': weights,
        'feature_cols': feature_cols
    }


if __name__ == '__main__':
    models = main()
