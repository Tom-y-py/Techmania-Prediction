# 🚀 Vylepšení Ensemble Modelu

## Problém
Předchozí ensemble model používal pouze **LightGBM** (váha 1.0), zatímco XGBoost a CatBoost měly váhu 0.0. Optimalizátor dal 100% váhu nejlepšímu modelu, což eliminovalo výhody ensemble přístupu.

## Řešení

### 1. **Minimální váha pro každý model (15%)**
- Každý model musí mít minimálně 15% váhu
- Zajišťuje diverzitu predikce
- Předchází overfittingu na jeden model

### 2. **Stacking Ensemble s Meta-modelem**
- Přidán druhý typ ensemble: **STACKING**
- Meta-model (Ridge Regression) se učí optimální kombinaci
- Automaticky vybírá lepší metodu (Weighted vs Stacking)

### 3. **Vylepšené hyperparametry pro diverzitu**
Každý model má nyní **jiné nastavení** pro zajištění různých typů chyb:

#### LightGBM (Balanced)
```python
- num_leaves: 31
- learning_rate: 0.02
- max_depth: 7
- feature_fraction: 0.75
- random_state: 42
```

#### XGBoost (Deep & Regularized)
```python
- max_depth: 8  # Hlubší než LightGBM
- learning_rate: 0.015  # Nižší než LightGBM
- colsample_bytree: 0.6  # Více randomizace
- reg_lambda: 0.8  # Vyšší L2
- random_state: 43  # JINÝ seed!
```

#### CatBoost (Aggressive Bagging)
```python
- depth: 8
- learning_rate: 0.025
- random_strength: 0.5  # Více randomizace
- bagging_temperature: 0.8  # Agresivnější
- bootstrap_type: 'Bayesian'  # Jiný typ bagging
- random_state: 44  # JINÝ seed!
```

### 4. **Lepší využití nových weather features**
- Vyšší kapacita modelů (více leaves, větší hloubka)
- Lepší zachycení weather interakcí
- Optimalizace pro 55+ features (včetně weather dat)

## Použití

### Trénování
```powershell
cd src
py ensemble_model.py
```

Model automaticky:
1. Natrénuje všechny 3 modely s novými hyperparametry
2. Vytvoří **Weighted Ensemble** (s min. váhou 15%)
3. Vytvoří **Stacking Ensemble** (meta-model)
4. Vybere lepší metodu
5. Uloží všechny modely včetně typu ensemble

### Predikce
```powershell
py predict.py 2026-01-15
```

Automaticky použije správný typ ensemble (weighted nebo stacking).

## Výhody

### ✅ Diverzita modelů
- 3 různé algoritmy s různými hyperparametry
- Různé random seeds
- Různé typy regularizace

### ✅ Všechny modely přispívají
- Minimální váha zajišťuje, že všechny modely jsou použity
- Ensemble zachytí různé vzory v datech

### ✅ Meta-learning
- Stacking může objevit nelineární kombinace
- Učí se, kdy kterému modelu důvěřovat

### ✅ Lepší využití weather features
- Modely mají dostatek kapacity
- Zachytí komplexní interakce (počasí × víkend, počasí × svátky)

## Očekávané výsledky

- **Všechny 3 modely budou použity** (váhy 15%+ každý)
- **Nižší validační MAE** než předchozí verze
- **Lepší generalizace** díky diverzitě
- **Robustnější predikce** při různých weather podmínkách

## Soubory

### Nové/Upravené soubory
- ✅ `src/ensemble_model.py` - Vylepšené hyperparametry + stacking
- ✅ `src/predict.py` - Podpora pro weighted i stacking ensemble

### Nové modely (po trénování)
- `models/ensemble_info.pkl` - Info o typu ensemble
- `models/meta_model.pkl` - Meta-model (pokud stacking vyhrál)
- `models/meta_weights.pkl` - Váhy meta-modelu (pokud lineární)

## Další možná vylepšení

1. **Neural Network** jako 4. model
2. **Feature selection** pro každý model jiný
3. **Temporal validation** - cross-validace na časových úsecích
4. **Ensemble pruning** - odstranit špatné predikce jednotlivých modelů

---
*Vytvořeno: 10.1.2026*
*Autor: GitHub Copilot*
