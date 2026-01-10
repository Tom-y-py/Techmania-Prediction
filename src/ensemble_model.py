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
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import minimize
import joblib
import os
from pathlib import Path
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
    
    # Parametry - optimalizováno pro weather features
    # LightGBM je dobrý na zachycení komplexních interakcí mezi features
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,  # Zvýšeno - více kapacity pro weather patterns
        'learning_rate': 0.02,  # Nižší learning rate pro lepší generalizaci
        'feature_fraction': 0.75,  # Střední hodnota - dostatek features ale i randomizace
        'bagging_fraction': 0.75,
        'bagging_freq': 5,
        'max_depth': 7,  # Zvýšeno - weather interakce mohou být komplexní
        'min_child_samples': 25,  # Balancovaná hodnota
        'reg_alpha': 0.3,  # Střední L1 regularizace
        'reg_lambda': 0.3,  # Střední L2 regularizace
        'min_split_gain': 0.01,  # Minimální zisk pro split - redukce noise
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
    
    # Parametry - XGBoost s jinou strategií než LightGBM pro diverzitu
    # XGBoost používá jiný algoritmus pro split finding = jiné chyby než LightGBM
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 8,  # Hlubší stromy než LightGBM - jiná struktura
        'learning_rate': 0.015,  # Ještě nižší než LightGBM
        'subsample': 0.8,  # Více dat než LightGBM
        'colsample_bytree': 0.6,  # Méně features - větší randomizace
        'colsample_bylevel': 0.8,  # Další rozměr randomizace
        'min_child_weight': 3,  # Nižší než LightGBM - jiné chování
        'gamma': 0.1,  # Nižší než před tím - umožní více splits
        'reg_alpha': 0.2,  # Nižší L1 než LightGBM
        'reg_lambda': 0.8,  # Vyšší L2 než LightGBM - jiný typ regularizace
        'random_state': 43,  # JINÝ seed = jiná randomizace = diverzita!
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
    
    # Model - CatBoost s native categorical handling
    # CatBoost je speciálně dobrý pro kategorické features (weather_code, atd.)
    model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.025,  # Mezi LightGBM a XGBoost
        depth=8,  # Hlubší - CatBoost dobře zvládá hloubku
        l2_leaf_reg=3,  # Nižší než před tím
        random_strength=0.5,  # Více randomizace pro diverzitu
        bagging_temperature=0.8,  # Vyšší - agresivnější bagging
        rsm=0.7,  # Random subspace method - náhodný výběr features
        od_type='Iter',
        od_wait=100,
        random_seed=44,  # JINÝ seed než ostatní modely!
        verbose=100,
        task_type='CPU',
        bootstrap_type='Bayesian'  # Jiný typ bagging než gradient boosting
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


def optimize_weights(predictions_dict, y_true, min_weight=0.15):
    """
    Najde optimální váhy pro ensemble pomocí optimalizace
    S OMEZENÍM: každý model musí mít minimální váhu pro zajištění diverzity
    
    Args:
        predictions_dict: Dict s predikcemi z každého modelu
        y_true: Skutečné hodnoty
        min_weight: Minimální váha pro každý model (default 0.15 = 15%)
        
    Returns:
        Optimální váhy (numpy array)
    """
    def ensemble_mae(weights):
        ensemble_pred = sum(w * pred for w, pred in zip(weights, predictions_dict.values()))
        return mean_absolute_error(y_true, ensemble_pred)
    
    # Počáteční váhy (rovnoměrné)
    n_models = len(predictions_dict)
    initial_weights = [1.0 / n_models] * n_models
    
    # DŮLEŽITÉ OMEZENÍ: každý model musí mít min_weight až 1, součet = 1
    # Tím zajistíme, že všechny modely přispějí do ensemble
    bounds = [(min_weight, 1.0)] * n_models
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


def create_stacking_ensemble(lgb_pred, xgb_pred, cat_pred, y_true, use_meta_model='ridge'):
    """
    Vytvoří STACKING ensemble - používá meta-model k naučení optimální kombinace
    Meta-model se učí, jak nejlépe kombinovat predikce základních modelů
    
    Args:
        lgb_pred: LightGBM predikce
        xgb_pred: XGBoost predikce  
        cat_pred: CatBoost predikce
        y_true: Skutečné hodnoty
        use_meta_model: Typ meta-modelu ('ridge', 'lasso', 'rf')
        
    Returns:
        Tuple[meta_model, ensemble_predictions]
    """
    print("\n" + "=" * 60)
    print("🧠 Creating STACKING Ensemble with Meta-Model...")
    print("=" * 60)
    
    # Ujistit se, že všechny predikce mají stejnou délku
    min_len = min(len(lgb_pred), len(xgb_pred), len(cat_pred))
    lgb_pred = lgb_pred[:min_len]
    xgb_pred = xgb_pred[:min_len]
    cat_pred = cat_pred[:min_len]
    y_true = y_true[:min_len]
    
    # Vytvořit features pro meta-model = predikce základních modelů
    meta_features = np.column_stack([lgb_pred, xgb_pred, cat_pred])
    
    # Vybrat meta-model
    if use_meta_model == 'ridge':
        # Ridge regression - penalizuje velké váhy, dobře generalizuje
        meta_model = Ridge(alpha=1.0, random_state=42)
        print(f"   Meta-model: Ridge Regression (L2 regularization)")
    elif use_meta_model == 'lasso':
        # Lasso - může nastavit některé váhy na 0 (feature selection)
        meta_model = Lasso(alpha=0.1, random_state=42)
        print(f"   Meta-model: Lasso Regression (L1 regularization)")
    elif use_meta_model == 'rf':
        # Random Forest - nelineární kombinace, zachytí komplexní interakce
        meta_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=3,  # Shallow trees - meta-model by neměl být příliš komplexní
            random_state=42,
            n_jobs=-1
        )
        print(f"   Meta-model: Random Forest (non-linear combination)")
    else:
        raise ValueError(f"Neznámý meta-model: {use_meta_model}")
    
    # Trénovat meta-model
    meta_model.fit(meta_features, y_true)
    
    # Predikce
    ensemble_pred = meta_model.predict(meta_features)
    
    # Zjistit váhy (pro lineární modely)
    if hasattr(meta_model, 'coef_'):
        raw_weights = meta_model.coef_
        # Normalizovat na [0, 1] které sečtou do 1
        weights_sum = np.abs(raw_weights).sum()
        if weights_sum > 0:
            weights = np.abs(raw_weights) / weights_sum
        else:
            weights = np.array([1/3, 1/3, 1/3])
        
        print(f"\n=== Meta-Model Learned Weights ===")
        print(f"LightGBM: {weights[0]:.3f}")
        print(f"XGBoost: {weights[1]:.3f}")
        print(f"CatBoost: {weights[2]:.3f}")
        print(f"Intercept: {meta_model.intercept_:.2f}")
    else:
        print(f"\n=== Meta-Model Info ===")
        print(f"   Non-linear model - no explicit weights")
        weights = None
    
    # Metriky
    print("\n=== STACKING ENSEMBLE Results ===")
    print(f"Val MAE: {mean_absolute_error(y_true, ensemble_pred):.2f}")
    print(f"Val RMSE: {np.sqrt(mean_squared_error(y_true, ensemble_pred)):.2f}")
    print(f"Val R²: {r2_score(y_true, ensemble_pred):.4f}")
    
    # Porovnání s jednotlivými modely
    print("\n=== Comparison ===")
    print(f"LightGBM MAE: {mean_absolute_error(y_true, lgb_pred):.2f}")
    print(f"XGBoost MAE: {mean_absolute_error(y_true, xgb_pred):.2f}")
    print(f"CatBoost MAE: {mean_absolute_error(y_true, cat_pred):.2f}")
    print(f"Stacking Ensemble MAE: {mean_absolute_error(y_true, ensemble_pred):.2f}")
    
    best_single = min(
        mean_absolute_error(y_true, lgb_pred),
        mean_absolute_error(y_true, xgb_pred),
        mean_absolute_error(y_true, cat_pred)
    )
    ensemble_mae = mean_absolute_error(y_true, ensemble_pred)
    improvement = best_single - ensemble_mae
    improvement_pct = (improvement / best_single) * 100
    
    if improvement > 0:
        print(f"\n✅ Improvement over best single model:")
        print(f"   {improvement:.2f} visitors better ({improvement_pct:.1f}% improvement)")
    elif improvement < 0:
        print(f"\n⚠️ Worse than best single model:")
        print(f"   {abs(improvement):.2f} visitors worse ({abs(improvement_pct):.1f}% degradation)")
    else:
        print(f"\n➡️ Same as best single model")
    
    return meta_model, ensemble_pred, weights


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
    ensemble_mae = mean_absolute_error(y_true, ensemble_pred)
    improvement = best_single - ensemble_mae
    improvement_pct = (improvement / best_single) * 100
    
    if improvement > 0:
        print(f"\n✅ Improvement over best single model:")
        print(f"   {improvement:.2f} visitors better ({improvement_pct:.1f}% improvement)")
    elif improvement < 0:
        print(f"\n⚠️ Worse than best single model:")
        print(f"   {abs(improvement):.2f} visitors worse ({abs(improvement_pct):.1f}% degradation)")
    else:
        print(f"\n➡️ Same as best single model")
    
    return ensemble_pred, weights


def main():
    """
    Hlavní pipeline pro ensemble model
    """
    print("\n" + "=" * 70)
    print("🚀 ENSEMBLE MODEL TRAINING PIPELINE - WITH WEATHER & HOLIDAYS DATA")
    print("=" * 70)
    
    # 1. Načíst data S POČASÍM A SVÁTKY
    print("\n📂 Loading data...")
    
    # Cesta k sloučeným datům (návštěvnost + počasí + svátky + všechny features)
    script_dir = Path(__file__).parent
    data_path = script_dir.parent / 'data' / 'processed' / 'techmania_with_weather_and_holidays.csv'
    
    df = pd.read_csv(data_path)
    print(f"   Loaded {len(df)} records from: {data_path.name}")
    print(f"   Date range: {df['date'].min()} - {df['date'].max()}")
    print(f"   Total columns: {len(df.columns)}")
    
    # Ověřit, že máme weather data
    weather_cols = ['temperature_mean', 'precipitation', 'weather_code', 'cloud_cover_percent']
    has_weather = all(col in df.columns for col in weather_cols)
    print(f"   Weather data present: {'✅ YES' if has_weather else '❌ NO'}")
    
    # Ověřit, že máme školní prázdniny
    holiday_cols = ['is_any_school_break', 'school_break_type', 'is_summer_holiday']
    has_holidays = all(col in df.columns for col in holiday_cols)
    print(f"   School break data present: {'✅ YES' if has_holidays else '❌ NO'}")
    
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
    
    # === DŮLEŽITÉ: Normalizace dominantního feature 'is_closed' ===
    # is_closed má obrovskou importance (1+ miliarda), což potlačuje vliv počasí
    # Snížíme jeho váhu, aby ostatní features (počasí, svátky, atd.) měly větší vliv
    print("\n🎯 Normalizing 'is_closed' feature to balance importance...")
    if 'is_closed' in X_train.columns:
        # Původní hodnoty: 0 nebo 1
        # Nové hodnoty: 0 nebo 0.1 (10x menší vliv)
        # Model se tak musí více spoléhat na ostatní features
        original_closed_count_train = X_train['is_closed'].sum()
        original_closed_count_val = X_val['is_closed'].sum()
        
        X_train['is_closed'] = X_train['is_closed'] * 0.1
        X_val['is_closed'] = X_val['is_closed'] * 0.1
        
        print(f"   Train: {original_closed_count_train} zavřených dní → váha snížena na 0.1")
        print(f"   Val: {original_closed_count_val} zavřených dní → váha snížena na 0.1")
        print(f"   ✅ Weather a ostatní features nyní dostanou větší prostor!")
    else:
        print("   ⚠️ Feature 'is_closed' nebyl nalezen")
    
    # 4. Trénovat modely
    lgb_model, lgb_pred = train_lightgbm(X_train, y_train, X_val, y_val)
    
    xgb_model, xgb_pred = train_xgboost(X_train, y_train, X_val, y_val)
    
    cat_model, cat_pred = train_catboost(X_train, y_train, X_val, y_val)
    
    # 5a. Weighted Ensemble (původní metoda s minimální váhou)
    print("\n" + "=" * 70)
    print("📊 METHOD 1: WEIGHTED ENSEMBLE (optimized weights with minimum)")
    print("=" * 70)
    ensemble_pred_weighted, weights = create_ensemble(
        lgb_pred,
        xgb_pred,
        cat_pred,
        y_val.values,
        optimize=True
    )
    mae_weighted = mean_absolute_error(y_val.values, ensemble_pred_weighted)
    
    # 5b. Stacking Ensemble (meta-model)
    print("\n" + "=" * 70)
    print("📊 METHOD 2: STACKING ENSEMBLE (meta-model learns combination)")
    print("=" * 70)
    meta_model, ensemble_pred_stacking, meta_weights = create_stacking_ensemble(
        lgb_pred,
        xgb_pred,
        cat_pred,
        y_val.values,
        use_meta_model='ridge'  # Ridge je dobrý default
    )
    mae_stacking = mean_absolute_error(y_val.values, ensemble_pred_stacking)
    
    # 5c. Porovnat s nejlepším jednotlivým modelem
    mae_lgb = mean_absolute_error(y_val.values, lgb_pred)
    mae_xgb = mean_absolute_error(y_val.values, xgb_pred)
    mae_cat = mean_absolute_error(y_val.values, cat_pred)
    best_single_mae = min(mae_lgb, mae_xgb, mae_cat)
    
    print("\n" + "=" * 70)
    print("🏆 FINAL COMPARISON")
    print("=" * 70)
    print(f"Best Single Model MAE: {best_single_mae:.2f}")
    print(f"Weighted Ensemble MAE: {mae_weighted:.2f}")
    print(f"Stacking Ensemble MAE: {mae_stacking:.2f}")
    
    # Vybrat nejlepší z VŠECH možností (včetně jednotlivých modelů!)
    candidates = [
        ('best_single', best_single_mae, None),
        ('weighted', mae_weighted, ensemble_pred_weighted),
        ('stacking', mae_stacking, ensemble_pred_stacking)
    ]
    
    winner = min(candidates, key=lambda x: x[1])
    final_ensemble_type = winner[0]
    final_mae = winner[1]
    final_ensemble_pred = winner[2]
    
    if final_ensemble_type == 'stacking':
        print(f"\n✅ Winner: STACKING ENSEMBLE")
        print(f"   MAE: {final_mae:.2f}")
        print(f"   Better than best single by: {best_single_mae - final_mae:.2f} visitors ({((best_single_mae - final_mae) / best_single_mae * 100):.1f}%)")
    elif final_ensemble_type == 'weighted':
        print(f"\n✅ Winner: WEIGHTED ENSEMBLE")
        print(f"   MAE: {final_mae:.2f}")
        print(f"   Better than best single by: {best_single_mae - final_mae:.2f} visitors ({((best_single_mae - final_mae) / best_single_mae * 100):.1f}%)")
    else:
        print(f"\n✅ Winner: BEST SINGLE MODEL (LightGBM)")
        print(f"   MAE: {final_mae:.2f}")
        print(f"   ⚠️ Ensemble methods didn't improve - using single model")
        final_ensemble_type = 'single_lgb'
    
    # 6. Uložit modely
    print("\n💾 Saving models...")
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    joblib.dump(lgb_model, os.path.join(models_dir, 'lightgbm_model.pkl'))
    joblib.dump(xgb_model, os.path.join(models_dir, 'xgboost_model.pkl'))
    joblib.dump(cat_model, os.path.join(models_dir, 'catboost_model.pkl'))
    joblib.dump(weights, os.path.join(models_dir, 'ensemble_weights.pkl'))
    joblib.dump(feature_cols, os.path.join(models_dir, 'feature_columns.pkl'))
    
    # Uložit informaci o typu ensemble a meta-model (pokud je stacking)
    ensemble_info = {
        'type': final_ensemble_type,
        'mae': final_mae
    }
    if final_ensemble_type == 'stacking':
        joblib.dump(meta_model, os.path.join(models_dir, 'meta_model.pkl'))
        if meta_weights is not None:
            joblib.dump(meta_weights, os.path.join(models_dir, 'meta_weights.pkl'))
        ensemble_info['meta_weights'] = meta_weights
    
    joblib.dump(ensemble_info, os.path.join(models_dir, 'ensemble_info.pkl'))
    
    print("\n✅ Models saved successfully!")
    print("   📁 models/lightgbm_model.pkl")
    print("   📁 models/xgboost_model.pkl")
    print("   📁 models/catboost_model.pkl")
    print("   📁 models/ensemble_weights.pkl")
    print("   📁 models/feature_columns.pkl")
    print("   📁 models/ensemble_info.pkl")
    if final_ensemble_type == 'stacking':
        print("   📁 models/meta_model.pkl")
        if meta_weights is not None:
            print("   📁 models/meta_weights.pkl")
    
    print("\n" + "=" * 70)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Final Ensemble Type: {final_ensemble_type.upper()}")
    print(f"Final Validation MAE: {final_mae:.2f}")
    
    return {
        'lgb': lgb_model,
        'xgb': xgb_model,
        'cat': cat_model,
        'weights': weights,
        'feature_cols': feature_cols,
        'ensemble_type': final_ensemble_type,
        'meta_model': meta_model if final_ensemble_type == 'stacking' else None,
        'meta_weights': meta_weights if final_ensemble_type == 'stacking' else None
    }


if __name__ == '__main__':
    models = main()
