"""
Ensemble Model V3 - 4 MAJOR IMPROVEMENTS:
1. Event features 
2. Weighted loss (penalizace velkých chyb 800+)
3. Hyperparameter tuning přes Optuna
4. Google Trend jako separate target
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from scipy.optimize import minimize
import joblib
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import optimalizovaného feature engineering s event features
from feature_engineering_v3 import create_features

print("=" * 80)
print("🚀 ENSEMBLE MODEL V3 - ALL 5 IMPROVEMENTS")
print("=" * 80)


def calculate_sample_weights(y, threshold=800, penalty=5.0):
    """
    IMPROVEMENT #2: Weighted loss
    Vypočítá váhy pro každý sample - vyšší váha pro dny s vysokou návštěvností
    
    Args:
        y: Target values (počet návštěvníků)
        threshold: Práh pro "high traffic" dny (default 800)
        penalty: Multiplikátor váhy pro high traffic dny (default 5.0)
    
    Returns:
        numpy array s váhami
    """
    weights = np.ones(len(y))
    high_traffic_mask = y >= threshold
    weights[high_traffic_mask] = penalty
    
    print(f"  ⚖️ Sample weights: {high_traffic_mask.sum()} high-traffic days (>={threshold}) with {penalty}x weight")
    
    return weights


def train_google_trend_model(df):
    """
    IMPROVEMENT #5: Google Trend jako separate target
    Trénuje model pro predikci google_trend z ostatních features
    
    Args:
        df: DataFrame s features včetně google_trend
    
    Returns:
        Trained model pro predikci google_trend
    """
    print("\n" + "=" * 80)
    print("📈 TRAINING GOOGLE TREND PREDICTOR (Improvement #5)")
    print("=" * 80)
    
    # Features pro predikci trendu (bez google_trend samotného!)
    trend_features = [
        'year', 'month', 'day_of_week', 'week_of_year', 'quarter',
        'is_weekend', 'is_summer_holiday', 'is_winter_holiday', 'is_school_year',
        'is_oct_28', 'is_autumn_break', 'is_summer_weekend_event', 'event_score',
        'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos',
        'week_sin', 'week_cos', 'normalized_time'
    ]
    
    # Filtrovat jen existující features
    available_features = [f for f in trend_features if f in df.columns]
    
    if 'google_trend' not in df.columns:
        print("⚠️ Google trend not available, skipping trend predictor")
        return None
    
    # Odstranit řádky s missing google_trend
    df_trend = df[df['google_trend'].notna()].copy()
    
    if len(df_trend) == 0:
        print("⚠️ No google trend data available")
        return None
    
    # Split data (80/20)
    train_size = int(len(df_trend) * 0.8)
    X_trend_train = df_trend[available_features].iloc[:train_size]
    y_trend_train = df_trend['google_trend'].iloc[:train_size]
    X_trend_val = df_trend[available_features].iloc[train_size:]
    y_trend_val = df_trend['google_trend'].iloc[train_size:]
    
    # Trénovat lightweight LightGBM pro trend prediction
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'num_leaves': 15,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'verbose': -1,
        'random_state': 42
    }
    
    train_data = lgb.Dataset(X_trend_train, label=y_trend_train)
    val_data = lgb.Dataset(X_trend_val, label=y_trend_val, reference=train_data)
    
    trend_model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    # Evaluate
    val_pred = trend_model.predict(X_trend_val, num_iteration=trend_model.best_iteration)
    mae = mean_absolute_error(y_trend_val, val_pred)
    r2 = r2_score(y_trend_val, val_pred)
    
    print(f"  ✓ Google Trend Predictor MAE: {mae:.2f}")
    print(f"  ✓ Google Trend Predictor R²: {r2:.4f}")
    
    return trend_model, available_features


def train_lightgbm_optuna(X_train, y_train, X_val, y_val, sample_weights=None, n_trials=50):
    """
    IMPROVEMENT #4: Hyperparameter tuning přes Optuna
    Trénuje LightGBM s Optuna optimalizací
    """
    import optuna
    from optuna.samplers import TPESampler
    
    print("\n" + "=" * 80)
    print("🔍 LIGHTGBM OPTUNA TUNING (Improvement #4)")
    print("=" * 80)
    
    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 15, 40),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.9),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.9),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 3.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 3.0),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.2),
            'verbose': -1,
            'random_state': 42
        }
        
        train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        mae = mean_absolute_error(y_val, val_pred)
        
        return mae
    
    # Optuna study
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\n  ✓ Best MAE: {study.best_value:.2f}")
    print(f"  ✓ Best params:")
    for key, value in study.best_params.items():
        print(f"      {key}: {value}")
    
    # Train final model s best params
    best_params = study.best_params.copy()
    best_params.update({
        'objective': 'regression',
        'metric': 'mae',
        'verbose': -1,
        'random_state': 42
    })
    
    train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    final_model = lgb.train(
        best_params,
        train_data,
        num_boost_round=2000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
    )
    
    return final_model


def train_xgboost_optuna(X_train, y_train, X_val, y_val, sample_weights=None, n_trials=50):
    """
    IMPROVEMENT #4: XGBoost s Optuna tuningem
    """
    import optuna
    from optuna.samplers import TPESampler
    
    print("\n" + "=" * 80)
    print("🔍 XGBOOST OPTUNA TUNING (Improvement #4)")
    print("=" * 80)
    
    def objective(trial):
        params = {
            'objective': 'reg:absoluteerror',
            'eval_metric': 'mae',
            'tree_method': 'hist',
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 3.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 3.0),
            'random_state': 42
        }
        
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dval, 'eval')],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        val_pred = model.predict(dval)
        mae = mean_absolute_error(y_val, val_pred)
        
        return mae
    
    # Optuna study
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\n  ✓ Best MAE: {study.best_value:.2f}")
    print(f"  ✓ Best params:")
    for key, value in study.best_params.items():
        print(f"      {key}: {value}")
    
    # Train final model
    best_params = study.best_params.copy()
    best_params.update({
        'objective': 'reg:absoluteerror',
        'eval_metric': 'mae',
        'tree_method': 'hist',
        'random_state': 42
    })
    
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    final_model = xgb.train(
        best_params,
        dtrain,
        num_boost_round=2000,
        evals=[(dtrain, 'train'), (dval, 'eval')],
        early_stopping_rounds=100,
        verbose_eval=100
    )
    
    return final_model


def train_catboost_optuna(X_train, y_train, X_val, y_val, sample_weights=None, n_trials=30):
    """
    IMPROVEMENT #4: CatBoost s Optuna tuningem
    """
    import optuna
    from optuna.samplers import TPESampler
    
    print("\n" + "=" * 80)
    print("🔍 CATBOOST OPTUNA TUNING (Improvement #4)")
    print("=" * 80)
    
    def objective(trial):
        params = {
            'loss_function': 'MAE',
            'eval_metric': 'MAE',
            'iterations': 1000,
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'random_strength': trial.suggest_float('random_strength', 0.0, 2.0),
            'verbose': False,
            'random_state': 42,
            'early_stopping_rounds': 50
        }
        
        model = CatBoostRegressor(**params)
        model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=(X_val, y_val),
            verbose=False
        )
        
        val_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, val_pred)
        
        return mae
    
    # Optuna study (méně trials pro CatBoost - je pomalejší)
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\n  ✓ Best MAE: {study.best_value:.2f}")
    print(f"  ✓ Best params:")
    for key, value in study.best_params.items():
        print(f"      {key}: {value}")
    
    # Train final model
    best_params = study.best_params.copy()
    best_params.update({
        'loss_function': 'MAE',
        'eval_metric': 'MAE',
        'iterations': 2000,
        'verbose': False,
        'random_state': 42,
        'early_stopping_rounds': 100
    })
    
    final_model = CatBoostRegressor(**best_params)
    final_model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=(X_val, y_val),
        verbose=100
    )
    
    return final_model


def train_tabnet(X_train, y_train, X_val, y_val, sample_weights=None):
    """
    IMPROVEMENT #3: Neural Network (TabNet) model
    """
    try:
        from pytorch_tabnet.tab_model import TabNetRegressor
        
        print("\n" + "=" * 80)
        print("🧠 TRAINING TABNET (Improvement #3)")
        print("=" * 80)
        
        # Convert to numpy if needed
        X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
        X_val_np = X_val.values if hasattr(X_val, 'values') else X_val
        y_train_np = y_train.values.reshape(-1, 1) if hasattr(y_train, 'values') else y_train.reshape(-1, 1)
        y_val_np = y_val.values.reshape(-1, 1) if hasattr(y_val, 'values') else y_val.reshape(-1, 1)
        
        # Fill NaN values (TabNet nemůže pracovat s NaN)
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        X_train_np = imputer.fit_transform(X_train_np)
        X_val_np = imputer.transform(X_val_np)
        
        print(f"  ✓ Filled NaN values with mean")
        
        # TabNet params
        import torch
        
        tabnet_params = {
            'n_d': 64,
            'n_a': 64,
            'n_steps': 5,
            'gamma': 1.5,
            'lambda_sparse': 1e-4,
            'optimizer_fn': torch.optim.Adam,
            'optimizer_params': {'lr': 2e-2},
            'scheduler_fn': torch.optim.lr_scheduler.StepLR,
            'scheduler_params': {'step_size': 50, 'gamma': 0.9},
            'mask_type': 'entmax',
            'verbose': 1,
            'seed': 42
        }
        
        model = TabNetRegressor(**tabnet_params)
        
        # Train
        model.fit(
            X_train_np, y_train_np,
            eval_set=[(X_val_np, y_val_np)],
            eval_metric=['mae'],
            max_epochs=200,
            patience=20,
            batch_size=256,
            virtual_batch_size=128,
            weights=sample_weights if sample_weights is not None else 1
        )
        
        # Predictions
        train_pred = model.predict(X_train_np).flatten()
        val_pred = model.predict(X_val_np).flatten()
        
        # Metrics
        print("\n=== TabNet Results ===")
        print(f"Train MAE: {mean_absolute_error(y_train, train_pred):.2f}")
        print(f"Val MAE: {mean_absolute_error(y_val, val_pred):.2f}")
        print(f"Val R²: {r2_score(y_val, val_pred):.4f}")
        
        return model
        
    except ImportError:
        print("\n⚠️ pytorch-tabnet not installed, skipping TabNet")
        print("   Install with: pip install pytorch-tabnet")
        return None


def optimize_ensemble_weights(predictions_dict, y_true):
    """
    Optimalizuje váhy pro ensemble
    """
    print("\n" + "=" * 80)
    print("⚖️ OPTIMIZING ENSEMBLE WEIGHTS")
    print("=" * 80)
    
    # Convert to matrix
    pred_matrix = np.column_stack([predictions_dict[name] for name in predictions_dict.keys()])
    
    def objective(weights):
        weights = weights / weights.sum()  # Normalize
        ensemble_pred = pred_matrix @ weights
        return mean_absolute_error(y_true, ensemble_pred)
    
    # Initial weights
    n_models = len(predictions_dict)
    initial_weights = np.ones(n_models) / n_models
    
    # Optimize
    bounds = [(0, 1) for _ in range(n_models)]
    result = minimize(objective, initial_weights, bounds=bounds, method='L-BFGS-B')
    
    # Normalize
    optimal_weights = result.x / result.x.sum()
    
    # Print results
    print("\n  Optimal weights:")
    for name, weight in zip(predictions_dict.keys(), optimal_weights):
        print(f"    {name:15s}: {weight:.3f}")
    
    return dict(zip(predictions_dict.keys(), optimal_weights))


def main():
    """
    Main training pipeline
    """
    print("\n" + "=" * 80)
    print("📂 LOADING DATA")
    print("=" * 80)
    
    # Load data
    script_dir = Path(__file__).parent
    data_path = script_dir.parent / 'data' / 'processed' / 'techmania_with_weather_and_holidays.csv'
    
    df = pd.read_csv(data_path)
    print(f"  ✓ Loaded {len(df)} records")
    
    # Create features (V3 with event features!)
    df = create_features(df)
    
    # Prepare X, y
    remove_cols = ['date', 'total_visitors', 'school_visitors', 'public_visitors',
                   'extra', 'nazvy_svatek', 'school_break_type', 'season_exact', 'week_position']
    feature_cols = [col for col in df.columns if col not in remove_cols]
    numeric_features = [col for col in feature_cols if df[col].dtype in ['int64', 'float64', 'bool']]
    
    X = df[numeric_features]
    y = df['total_visitors']
    
    print(f"\n  📊 Feature matrix: {X.shape}")
    print(f"  📊 Numeric features: {len(numeric_features)}")
    
    # Split data (80/20)
    train_size = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_val = y.iloc[:train_size], y.iloc[train_size:]
    
    print(f"  ✓ Train: {len(X_train)} | Val: {len(X_val)}")
    
    # IMPROVEMENT #2: Calculate sample weights
    train_weights = calculate_sample_weights(y_train, threshold=800, penalty=5.0)
    
    # IMPROVEMENT #5: Train Google Trend predictor
    trend_result = train_google_trend_model(df)
    if trend_result is not None:
        trend_model, trend_features = trend_result
        
        # Predict trend for train/val
        pred_trend_train = trend_model.predict(df[trend_features].iloc[:train_size])
        pred_trend_val = trend_model.predict(df[trend_features].iloc[train_size:])
        
        # Add to features
        X_train['predicted_google_trend'] = pred_trend_train
        X_val['predicted_google_trend'] = pred_trend_val
        
        print(f"  ✓ Added predicted_google_trend feature")
    
    # IMPROVEMENT #4: Train models with Optuna
    print("\n" + "=" * 80)
    print("🚂 TRAINING OPTIMIZED MODELS")
    print("=" * 80)
    
    # LightGBM with Optuna
    lgb_model = train_lightgbm_optuna(X_train, y_train, X_val, y_val, 
                                       sample_weights=train_weights, n_trials=50)
    
    # XGBoost with Optuna
    xgb_model = train_xgboost_optuna(X_train, y_train, X_val, y_val, 
                                      sample_weights=train_weights, n_trials=50)
    
    # CatBoost with Optuna
    cat_model = train_catboost_optuna(X_train, y_train, X_val, y_val, 
                                       sample_weights=train_weights, n_trials=30)
    
    # IMPROVEMENT #3: TabNet - ODEBRÁNO (problémy s kompatibilitou)
    # tabnet_model = train_tabnet(X_train, y_train, X_val, y_val, sample_weights=train_weights)
    
    # Get predictions
    print("\n" + "=" * 80)
    print("📊 COLLECTING PREDICTIONS")
    print("=" * 80)
    
    predictions_dict = {}
    
    lgb_val_pred = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
    predictions_dict['LightGBM'] = lgb_val_pred
    print(f"  ✓ LightGBM MAE: {mean_absolute_error(y_val, lgb_val_pred):.2f}")
    
    dval = xgb.DMatrix(X_val)
    xgb_val_pred = xgb_model.predict(dval)
    predictions_dict['XGBoost'] = xgb_val_pred
    print(f"  ✓ XGBoost MAE: {mean_absolute_error(y_val, xgb_val_pred):.2f}")
    
    cat_val_pred = cat_model.predict(X_val)
    predictions_dict['CatBoost'] = cat_val_pred
    print(f"  ✓ CatBoost MAE: {mean_absolute_error(y_val, cat_val_pred):.2f}")
    
    # Optimize ensemble weights
    optimal_weights = optimize_ensemble_weights(predictions_dict, y_val)
    
    # Final ensemble prediction
    ensemble_pred = sum(predictions_dict[name] * optimal_weights[name] 
                        for name in predictions_dict.keys())
    
    # Final metrics
    ensemble_mae = mean_absolute_error(y_val, ensemble_pred)
    ensemble_r2 = r2_score(y_val, ensemble_pred)
    
    print("\n" + "=" * 80)
    print("🎯 FINAL ENSEMBLE RESULTS")
    print("=" * 80)
    print(f"  Overall MAE: {ensemble_mae:.2f}")
    print(f"  Overall R²: {ensemble_r2:.4f}")
    
    # Vypočítat MAE odděleně pro víkendy a všední dny
    print("\n" + "=" * 80)
    print("📊 MAE BREAKDOWN (Weekday vs Weekend)")
    print("=" * 80)
    
    # Split validační sadu na víkendy a všední dny
    is_weekend_val = X_val['is_weekend'] == 1
    weekday_mask = ~is_weekend_val
    weekend_mask = is_weekend_val
    
    # Vypočítat MAE pro každou skupinu
    if weekday_mask.sum() > 0:
        mae_weekday = mean_absolute_error(y_val[weekday_mask], ensemble_pred[weekday_mask])
        residuals_weekday = np.abs(y_val[weekday_mask].values - ensemble_pred[weekday_mask])
        print(f"  Weekday MAE: {mae_weekday:.2f} (n={weekday_mask.sum()})")
    else:
        mae_weekday = ensemble_mae
        residuals_weekday = np.array([])
        print("  ⚠️ No weekday data in validation set")
    
    if weekend_mask.sum() > 0:
        mae_weekend = mean_absolute_error(y_val[weekend_mask], ensemble_pred[weekend_mask])
        residuals_weekend = np.abs(y_val[weekend_mask].values - ensemble_pred[weekend_mask])
        print(f"  Weekend MAE: {mae_weekend:.2f} (n={weekend_mask.sum()})")
    else:
        mae_weekend = ensemble_mae
        residuals_weekend = np.array([])
        print("  ⚠️ No weekend data in validation set")
    
    # Uložit historická rezidua pro realistické CI
    historical_mae = {
        'weekday': float(mae_weekday),
        'weekend': float(mae_weekend),
        'overall': float(ensemble_mae),
        'residuals_weekday': residuals_weekday,
        'residuals_weekend': residuals_weekend
    }
    
    print(f"\n  ✓ Historical MAE saved for confidence intervals")
    
    if ensemble_mae < 100:
        print("\n  🎉 SUCCESS! MAE < 100 ACHIEVED!")
    else:
        print(f"\n  ⚠️ MAE still {ensemble_mae:.2f}, need {ensemble_mae - 100:.2f} more reduction")
    
    # Save models
    print("\n" + "=" * 80)
    print("💾 SAVING MODELS")
    print("=" * 80)
    
    models_dir = script_dir.parent / 'models'
    models_dir.mkdir(exist_ok=True)
    
    joblib.dump(lgb_model, models_dir / 'lightgbm_v3.pkl')
    joblib.dump(xgb_model, models_dir / 'xgboost_v3.pkl')
    joblib.dump(cat_model, models_dir / 'catboost_v3.pkl')
    joblib.dump(optimal_weights, models_dir / 'ensemble_weights_v3.pkl')
    joblib.dump(numeric_features, models_dir / 'feature_names_v3.pkl')
    joblib.dump(historical_mae, models_dir / 'historical_mae_v3.pkl')
    
    # TabNet odebráno
    # tabnet_model.save_model(str(models_dir / 'tabnet_v3'))

    
    if trend_result is not None:
        joblib.dump(trend_model, models_dir / 'google_trend_predictor_v3.pkl')
        joblib.dump(trend_features, models_dir / 'trend_feature_names_v3.pkl')
    
    print("  ✓ All models saved!")
    
    return ensemble_mae


if __name__ == "__main__":
    final_mae = main()
    print(f"\n{'='*80}")
    print(f"FINAL MAE: {final_mae:.2f}")
    print(f"{'='*80}")
