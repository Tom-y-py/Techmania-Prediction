# Ensemble Model - Plán implementace
## LightGBM + Prophet + Neural Network

---

## 📚 Co je Ensemble model?

**Ensemble learning** je technika strojového učení, která kombinuje **predikce z více různých modelů** do jednoho finálního výstupu. Myšlenka je jednoduchá: 

> **"Více hlav ví více než jedna"**

Každý model má své **silné a slabé stránky**. Ensemble kombinuje jejich přednosti a minimalizuje slabiny.

### **Typy Ensemble:**

1. **Bagging** (Bootstrap Aggregating)
   - Modely trénované na různých podmnožinách dat
   - Příklad: Random Forest

2. **Boosting**
   - Modely se učí postupně, každý opravuje chyby předchozího
   - Příklad: XGBoost, LightGBM

3. **Stacking**
   - Meta-model kombinuje predikce base modelů
   - Nejsložitější, ale často nejpřesnější

4. **Voting/Averaging** ⭐ (náš přístup)
   - Jednoduchý průměr nebo vážený průměr predikcí
   - Jednoduchý a efektivní

---

## 🎯 Proč Ensemble pro návštěvnost Techmanie?

### **1. LightGBM - "Detailista"**
**Co dělá dobře:**
- ✅ Zachycuje **komplexní nelineární vztahy** mezi features
- ✅ Skvělý s **kategorickými proměnnými** (den v týdnu, měsíc)
- ✅ **Feature interactions** (kombinace počasí + víkend + prázdniny)
- ✅ Robustní vůči **outlierům**

**Kde má slabiny:**
- ❌ Může "zapomenout" na **dlouhodobé trendy**
- ❌ Méně dobrý na **extrapolaci** (predikce mimo rozsah trénovacích dat)

---

### **2. Prophet - "Časovkář"**
**Co dělá dobře:**
- ✅ Specializovaný na **časové řady**
- ✅ Automaticky detekuje **sezónnost** (týdenní, měsíční, roční)
- ✅ Zachycuje **dlouhodobé trendy**
- ✅ Vynikající na **svátky a speciální události**
- ✅ Poskytuje **confidence intervals** (intervaly spolehlivosti)

**Kde má slabiny:**
- ❌ Jednodušší model, méně flexibilní
- ❌ Nezvládá dobře **komplexní interakce** mezi features
- ❌ Horší s **náhlými změnami** (např. speciální akce)

---

### **3. Neural Network (LSTM/Dense) - "Vzorář"**
**Co dělá dobře:**
- ✅ Objevuje **skryté vzory** v datech
- ✅ Zachycuje **sekvenční závislosti** (co bylo včera, předevčírem...)
- ✅ Velmi flexibilní, může se naučit **komplexní vztahy**
- ✅ Dobré na **krátkodobé předpovědi**

**Kde má slabiny:**
- ❌ Potřebuje **hodně dat** (máme ~3600 záznamů - OK, ale ne ideální)
- ❌ **Pomalé trénování**
- ❌ **Black box** - těžko interpretovatelné
- ❌ Náchylné k **overfittingu**

---

## 🧩 Jak to dáme dohromady?

### **Strategie kombinace:**

```
Finální predikce = w1 × LightGBM + w2 × Prophet + w3 × Neural Network
```

Kde `w1 + w2 + w3 = 1.0` (váhy musí dát dohromady 100%)

### **Doporučené váhy (výchozí):**
```python
weights = {
    'lightgbm': 0.45,   # 45% - nejsilnější model
    'prophet': 0.30,    # 30% - důležitá sezónnost
    'neural_net': 0.25  # 25% - zachytí speciální vzory
}
```

**Tyto váhy se optimalizují** na validačních datech!

---

## 📋 Implementační plán - Krok za krokem

### **Fáze 1: Příprava dat (společná pro všechny modely)**

#### **1.1 Feature Engineering**

```python
import pandas as pd
import numpy as np

def create_features(df):
    """
    Vytvoří všechny potřebné features pro ensemble
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # === ČASOVÉ FEATURES ===
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['quarter'] = df['date'].dt.quarter
    df['day_of_year'] = df['date'].dt.dayofyear
    
    # Víkend
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # === LAG FEATURES (historické hodnoty) ===
    for lag in [1, 7, 14, 30]:
        df[f'visitors_lag_{lag}'] = df['total_visitors'].shift(lag)
    
    # === ROLLING STATISTICS ===
    for window in [7, 14, 30]:
        df[f'visitors_rolling_mean_{window}'] = (
            df['total_visitors'].rolling(window=window, min_periods=1).mean()
        )
        df[f'visitors_rolling_std_{window}'] = (
            df['total_visitors'].rolling(window=window, min_periods=1).std()
        )
        df[f'visitors_rolling_min_{window}'] = (
            df['total_visitors'].rolling(window=window, min_periods=1).min()
        )
        df[f'visitors_rolling_max_{window}'] = (
            df['total_visitors'].rolling(window=window, min_periods=1).max()
        )
    
    # === SEZÓNNÍ FEATURES ===
    # Letní prázdniny (červenec + srpen)
    df['is_summer_holiday'] = df['month'].isin([7, 8]).astype(int)
    
    # Vánoční prázdniny (23.12 - 2.1)
    df['is_winter_holiday'] = (
        ((df['month'] == 12) & (df['day'] >= 23)) |
        ((df['month'] == 1) & (df['day'] <= 2))
    ).astype(int)
    
    # Školní rok vs prázdniny
    df['is_school_year'] = (~df['month'].isin([7, 8])).astype(int)
    
    # === SVÁTKY (z extra sloupce) ===
    df['is_holiday'] = df['extra'].notna().astype(int)
    
    # === DERIVED FEATURES ===
    # Poměr školní/veřejní návštěvníci (pokud existují)
    if 'school_visitors' in df.columns and 'public_visitors' in df.columns:
        df['school_ratio'] = df['school_visitors'] / (df['total_visitors'] + 1)
        df['public_ratio'] = df['public_visitors'] / (df['total_visitors'] + 1)
    
    # Otevírací doba v hodinách
    if 'opening_hours' in df.columns:
        df['opening_hours_numeric'] = df['opening_hours'].fillna(0)
    
    # Trend (lineární číslo dne)
    df['days_since_start'] = (df['date'] - df['date'].min()).dt.days
    
    return df
```

#### **1.2 Train/Validation/Test Split**

```python
def split_data(df, train_end='2023-12-31', val_end='2024-12-31'):
    """
    Chronologický split dat
    Train: 2016-2023
    Validation: 2024
    Test: 2025
    """
    train = df[df['date'] <= train_end].copy()
    val = df[(df['date'] > train_end) & (df['date'] <= val_end)].copy()
    test = df[df['date'] > val_end].copy()
    
    # Odstranit řádky s NaN (z lag features)
    train = train.dropna()
    val = val.dropna()
    test = test.dropna()
    
    return train, val, test
```

---

### **Fáze 2: Model 1 - LightGBM**

```python
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_lightgbm(X_train, y_train, X_val, y_val):
    """
    Trénuje LightGBM model
    """
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
            lgb.early_stopping(stopping_rounds=100),
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
    
    return model, val_pred
```

---

### **Fáze 3: Model 2 - Prophet**

```python
from prophet import Prophet

def train_prophet(train_df, val_df):
    """
    Trénuje Prophet model
    """
    # Prophet vyžaduje speciální formát: 'ds' (datum) a 'y' (target)
    prophet_train = pd.DataFrame({
        'ds': train_df['date'],
        'y': train_df['total_visitors']
    })
    
    # Přidání svátků
    holidays = train_df[train_df['is_holiday'] == 1][['date', 'extra']].copy()
    holidays.columns = ['ds', 'holiday']
    holidays = holidays.dropna()
    
    # Model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative',  # nebo 'additive'
        holidays=holidays,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10
    )
    
    # Přidání custom seasonality
    model.add_seasonality(
        name='monthly',
        period=30.5,
        fourier_order=5
    )
    
    # Trénování
    model.fit(prophet_train)
    
    # Predikce na validaci
    future = pd.DataFrame({'ds': val_df['date']})
    forecast = model.predict(future)
    val_pred = forecast['yhat'].values
    
    # Metriky
    print("\n=== Prophet Results ===")
    print(f"Val MAE: {mean_absolute_error(val_df['total_visitors'], val_pred):.2f}")
    print(f"Val RMSE: {np.sqrt(mean_squared_error(val_df['total_visitors'], val_pred)):.2f}")
    print(f"Val R²: {r2_score(val_df['total_visitors'], val_pred):.4f}")
    
    return model, val_pred
```

---

### **Fáze 4: Model 3 - Neural Network (LSTM)**

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler

def create_sequences(X, y, seq_length=7):
    """
    Vytvoří sekvence pro LSTM
    seq_length: kolik dnů zpět model vidí
    """
    Xs, ys = [], []
    for i in range(len(X) - seq_length):
        Xs.append(X[i:(i + seq_length)])
        ys.append(y[i + seq_length])
    return np.array(Xs), np.array(ys)

def train_neural_network(X_train, y_train, X_val, y_val, seq_length=7):
    """
    Trénuje Neural Network (LSTM) model
    """
    # Normalizace dat
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1)).flatten()
    
    # Vytvoření sekvencí
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, seq_length)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_scaled, seq_length)
    
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
    
    # Callbacks
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7
    )
    
    # Trénování
    history = model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=200,
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # Predikce
    val_pred_scaled = model.predict(X_val_seq)
    val_pred = scaler_y.inverse_transform(val_pred_scaled).flatten()
    
    # Upravit délku (kvůli sequences)
    y_val_adjusted = y_val.values[seq_length:]
    
    # Metriky
    print("\n=== Neural Network Results ===")
    print(f"Val MAE: {mean_absolute_error(y_val_adjusted, val_pred):.2f}")
    print(f"Val RMSE: {np.sqrt(mean_squared_error(y_val_adjusted, val_pred)):.2f}")
    print(f"Val R²: {r2_score(y_val_adjusted, val_pred):.4f}")
    
    return model, val_pred, scaler_X, scaler_y
```

---

### **Fáze 5: Ensemble - Kombinace modelů**

```python
from scipy.optimize import minimize

def optimize_weights(predictions_dict, y_true):
    """
    Najde optimální váhy pro ensemble pomocí optimalizace
    """
    def ensemble_mae(weights):
        # Weighted average
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
    """
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
        weights = [0.45, 0.30, 0.25]
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
    
    improvement = (
        min(
            mean_absolute_error(y_true, lgb_pred),
            mean_absolute_error(y_true, prophet_pred),
            mean_absolute_error(y_true, nn_pred)
        ) - mean_absolute_error(y_true, ensemble_pred)
    )
    print(f"\nImprovement over best single model: {improvement:.2f} visitors ({improvement/mean_absolute_error(y_true, ensemble_pred)*100:.1f}%)")
    
    return ensemble_pred, weights
```

---

### **Fáze 6: Hlavní pipeline**

```python
def main():
    """
    Hlavní pipeline pro ensemble model
    """
    # 1. Načíst data
    df = pd.read_csv('data/raw/techmania_cleaned_master.csv')
    
    # 2. Feature engineering
    df = create_features(df)
    
    # 3. Split data
    train, val, test = split_data(df)
    
    # Připravit X, y
    feature_cols = [col for col in df.columns if col not in 
                    ['date', 'total_visitors', 'school_visitors', 
                     'public_visitors', 'extra', 'day_of_week', 'opening_hours']]
    
    X_train = train[feature_cols]
    y_train = train['total_visitors']
    X_val = val[feature_cols]
    y_val = val['total_visitors']
    
    # 4. Trénovat modely
    print("=" * 50)
    print("Training LightGBM...")
    print("=" * 50)
    lgb_model, lgb_pred = train_lightgbm(X_train, y_train, X_val, y_val)
    
    print("\n" + "=" * 50)
    print("Training Prophet...")
    print("=" * 50)
    prophet_model, prophet_pred = train_prophet(train, val)
    
    print("\n" + "=" * 50)
    print("Training Neural Network...")
    print("=" * 50)
    nn_model, nn_pred, scaler_X, scaler_y = train_neural_network(
        X_train, y_train, X_val, y_val, seq_length=7
    )
    
    # 5. Ensemble
    print("\n" + "=" * 50)
    print("Creating Ensemble...")
    print("=" * 50)
    
    # NN má kratší predikce kvůli sequences
    y_val_adjusted = y_val.values[-len(nn_pred):]
    lgb_pred_adjusted = lgb_pred[-len(nn_pred):]
    prophet_pred_adjusted = prophet_pred[-len(nn_pred):]
    
    ensemble_pred, weights = create_ensemble(
        lgb_pred_adjusted,
        prophet_pred_adjusted,
        nn_pred,
        y_val_adjusted,
        optimize=True
    )
    
    # 6. Uložit modely
    import joblib
    
    joblib.dump(lgb_model, 'models/lightgbm_model.pkl')
    joblib.dump(prophet_model, 'models/prophet_model.pkl')
    nn_model.save('models/neural_network_model.h5')
    joblib.dump(scaler_X, 'models/scaler_X.pkl')
    joblib.dump(scaler_y, 'models/scaler_y.pkl')
    joblib.dump(weights, 'models/ensemble_weights.pkl')
    
    print("\n✅ Models saved successfully!")
    
    return {
        'lgb': lgb_model,
        'prophet': prophet_model,
        'nn': nn_model,
        'weights': weights,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y
    }

if __name__ == '__main__':
    models = main()
```

---

## 📊 Vizualizace výsledků

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_results(y_true, predictions_dict, ensemble_pred, dates):
    """
    Vizualizuje predikce všech modelů
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # 1. Time series plot
    ax1 = axes[0, 0]
    ax1.plot(dates, y_true, label='Actual', linewidth=2, color='black')
    ax1.plot(dates, predictions_dict['lightgbm'], label='LightGBM', alpha=0.7)
    ax1.plot(dates, predictions_dict['prophet'], label='Prophet', alpha=0.7)
    ax1.plot(dates, predictions_dict['neural_net'], label='Neural Net', alpha=0.7)
    ax1.plot(dates, ensemble_pred, label='Ensemble', linewidth=2, linestyle='--', color='red')
    ax1.set_title('Predictions Over Time')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Visitors')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Scatter plot - Predicted vs Actual
    ax2 = axes[0, 1]
    ax2.scatter(y_true, ensemble_pred, alpha=0.5)
    ax2.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
             'r--', linewidth=2, label='Perfect prediction')
    ax2.set_title('Ensemble: Predicted vs Actual')
    ax2.set_xlabel('Actual Visitors')
    ax2.set_ylabel('Predicted Visitors')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Residuals
    ax3 = axes[1, 0]
    residuals = y_true - ensemble_pred
    ax3.scatter(dates, residuals, alpha=0.5)
    ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax3.set_title('Ensemble Residuals')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Error (Actual - Predicted)')
    ax3.grid(True, alpha=0.3)
    
    # 4. MAE comparison
    ax4 = axes[1, 1]
    maes = {
        'LightGBM': mean_absolute_error(y_true, predictions_dict['lightgbm']),
        'Prophet': mean_absolute_error(y_true, predictions_dict['prophet']),
        'Neural Net': mean_absolute_error(y_true, predictions_dict['neural_net']),
        'Ensemble': mean_absolute_error(y_true, ensemble_pred)
    }
    colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold']
    ax4.bar(maes.keys(), maes.values(), color=colors)
    ax4.set_title('MAE Comparison')
    ax4.set_ylabel('Mean Absolute Error')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Přidat hodnoty na sloupcích
    for i, (model, mae) in enumerate(maes.items()):
        ax4.text(i, mae + 2, f'{mae:.1f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('ensemble_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n📊 Visualization saved as 'ensemble_results.png'")
```

---

## 🚀 Použití modelu pro predikci

```python
def predict_future(date, models_dict):
    """
    Predikuje návštěvnost pro konkrétní datum
    
    Args:
        date: datetime nebo string ve formátu 'YYYY-MM-DD'
        models_dict: dict s natrénovanými modely
    
    Returns:
        prediction: int - predikovaný počet návštěvníků
    """
    # Načíst historická data (potřebujeme pro lag features)
    df = pd.read_csv('data/raw/techmania_cleaned_master.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Přidat nový řádek pro predikci
    if isinstance(date, str):
        date = pd.to_datetime(date)
    
    new_row = pd.DataFrame({'date': [date], 'total_visitors': [np.nan]})
    df = pd.concat([df, new_row], ignore_index=True)
    
    # Feature engineering
    df = create_features(df)
    
    # Vybrat poslední řádek (náš prediction date)
    X_pred = df[df['date'] == date][feature_cols].dropna(axis=1)
    
    # === Predikce z každého modelu ===
    
    # 1. LightGBM
    lgb_model = models_dict['lgb']
    lgb_pred = lgb_model.predict(X_pred)[0]
    
    # 2. Prophet
    prophet_model = models_dict['prophet']
    future = pd.DataFrame({'ds': [date]})
    prophet_forecast = prophet_model.predict(future)
    prophet_pred = prophet_forecast['yhat'].values[0]
    
    # 3. Neural Network (složitější - potřebujeme sekvenci)
    nn_model = models_dict['nn']
    scaler_X = models_dict['scaler_X']
    scaler_y = models_dict['scaler_y']
    
    # Vytvořit sekvenci posledních 7 dnů
    last_7_days = df[df['date'] < date].tail(7)[feature_cols]
    last_7_days_scaled = scaler_X.transform(last_7_days)
    X_pred_seq = last_7_days_scaled.reshape(1, 7, -1)
    
    nn_pred_scaled = nn_model.predict(X_pred_seq, verbose=0)
    nn_pred = scaler_y.inverse_transform(nn_pred_scaled)[0][0]
    
    # === Ensemble ===
    weights = models_dict['weights']
    ensemble_pred = (
        weights[0] * lgb_pred +
        weights[1] * prophet_pred +
        weights[2] * nn_pred
    )
    
    # Zaokrouhlit na celé číslo
    ensemble_pred = int(round(ensemble_pred))
    
    print(f"\n🔮 Predikce pro {date.strftime('%Y-%m-%d')} ({date.strftime('%A')}):")
    print(f"  LightGBM: {int(lgb_pred)} návštěvníků")
    print(f"  Prophet: {int(prophet_pred)} návštěvníků")
    print(f"  Neural Network: {int(nn_pred)} návštěvníků")
    print(f"  ➡️ ENSEMBLE: {ensemble_pred} návštěvníků")
    
    return ensemble_pred

# Příklad použití:
# prediction = predict_future('2026-02-14', models_dict)
```

---

## ⚙️ Hyperparameter tuning

Pro ještě lepší výsledky můžete optimalizovat hyperparametry každého modelu:

```python
from sklearn.model_selection import RandomizedSearchCV
import optuna

# Pro LightGBM - Optuna
def objective_lgb(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
    }
    
    # Train a evaluate...
    # Return validation MAE
    
# study = optuna.create_study(direction='minimize')
# study.optimize(objective_lgb, n_trials=100)
```

---

## 📈 Očekávané výsledky

### **Jednotlivé modely:**
- **LightGBM**: MAE ~50-65 návštěvníků
- **Prophet**: MAE ~60-80 návštěvníků  
- **Neural Network**: MAE ~55-75 návštěvníků

### **Ensemble:**
- **MAE: ~35-50 návštěvníků** 
- **RMSE: ~50-70 návštěvníků**
- **R² score: ~0.85-0.92**
- **MAPE: ~10-15%**

### **Zlepšení oproti single model:**
- **15-30% lepší přesnost** než nejlepší jednotlivý model
- **Robustnější** - méně citlivý na outliers
- **Stabilnější** - méně variance v predikcích

---

## 💾 Struktura souborů

Po implementaci budete mít:

```
models/
├── lightgbm_model.pkl          # LightGBM model
├── prophet_model.pkl           # Prophet model
├── neural_network_model.h5     # Keras NN model
├── scaler_X.pkl                # Scaler pro features
├── scaler_y.pkl                # Scaler pro target
└── ensemble_weights.pkl        # Optimální váhy

src/
├── ensemble_model.py           # Hlavní implementace
├── feature_engineering.py      # Feature creation
└── predict.py                  # Prediction function

results/
├── ensemble_results.png        # Vizualizace
└── metrics.json                # Metriky všech modelů
```

---

## ⚠️ Důležité poznámky

### **1. Compute requirements:**
- **LightGBM**: Rychlý (minuty)
- **Prophet**: Střední (10-20 minut)
- **Neural Network**: Pomalý (30-60 minut, ideálně s GPU)
- **Celkem**: ~1-2 hodiny prvního trénování

### **2. Inference speed:**
- **Single prediction**: ~100-500ms
- **Batch predictions**: velmi rychlý

### **3. Memory:**
- **Modely dohromady**: ~100-200 MB
- **Training**: ~2-4 GB RAM

### **4. Maintenance:**
- **Retraining**: Doporučeno každý měsíc s novými daty
- **Monitoring**: Sledovat MAE na production datech

---

## 🎯 Závěr

Ensemble model kombinuje:
- ✅ **Sílu gradient boostingu** (LightGBM) pro komplexní vztahy
- ✅ **Časovou inteligenci** (Prophet) pro trendy a sezónnost
- ✅ **Deep learning** (NN) pro skryté vzory

**Výsledek:** Nejpřesnější možná predikce pro vaše data! 🎉

---

## 📚 Další kroky

1. ✅ Implementovat feature engineering
2. ✅ Natrénovat všechny tři modely
3. ✅ Optimalizovat váhy na validaci
4. ✅ Vyhodnotit na test setu
5. ✅ Přidat **confidence intervals**
6. 🔥 Vytvořit **API endpoint** (FastAPI/Flask)
7. 🔥 Vytvořit **monitoring dashboard**

---

**Vytvořeno pro projekt: Predikce návštěvnosti Techmanie**
**Datum: 9. ledna 2026**
