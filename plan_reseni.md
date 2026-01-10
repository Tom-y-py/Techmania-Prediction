# Plán řešení - Predikce návštěvnosti Techmanie

## 📊 Analýza zadání a dat

### Co máme k dispozici:
- **Dataset**: 3 653 záznamů (cca 10 let dat: 2016-2025)
- **Struktura dat**:
  - `date` - datum návštěvy
  - `day_of_week` - den v týdnu (česky)
  - `school_visitors` - návštěvníci ze škol
  - `public_visitors` - veřejní návštěvníci
  - `total_visitors` - celkový počet návštěvníků
  - `extra` - speciální události/svátky
  - `opening_hours` - otevírací doba

### Co potřebujeme predikovat:
- Počet návštěvníků pro konkrétní den nebo časové období
- Zohlednění faktorů: počasí, svátky, sezónnost, speciální akce

---

## 🎯 Doporučené řešení

### **Typ úlohy:** Regrese časových řad (Time Series Regression)

### **Vhodné technologie:**

#### 1. **Klasické ML modely (doporučeno pro začátek)**
- **Random Forest Regressor** ✨ (nejlepší pro začátečníky)
- **XGBoost / LightGBM** (silnější, ale složitější)
- **Linear Regression** (baseline model)

**Výhody:**
- Jednoduchá implementace
- Rychlé trénování
- Dobře interpretovatelné
- Funguje s menšími datasety

#### 2. **Časové řady specifické modely**
- **SARIMA** (Seasonal AutoRegressive Integrated Moving Average)
- **Prophet** (Facebook's forecasting tool)
- **LSTM** (Long Short-Term Memory - deep learning)

**Výhody:**
- Specificky navržené pro časové řady
- Zachycují sezónnost automaticky
- Prophet je velmi user-friendly

#### 3. **Hybridní přístup** (nejlepší výsledky)
- Kombinace více modelů (ensemble)

---

## 📋 Implementační plán

### **Fáze 1: Příprava dat (Feature Engineering)**

1. **Časové features:**
   - Den v týdnu (už máme)
   - Měsíc
   - Čtvrtletí
   - Týden v roce
   - Je víkend? (boolean)
   - Je svátek? (z `extra` sloupce)

2. **Lag features** (historická data)
   - Návštěvnost před 1 dnem
   - Návštěvnost před 7 dny (týden zpět)
   - Návštěvnost před 14 dny
   - Rolling average (klouzavý průměr za 7/14/30 dní)

3. **Sezónní features:**
   - Je prázdninové období?
   - Je školní rok?
   - Pololetní/vánoční prázdniny

4. **Externí data (optional - rozšíření):**
   - Data o počasí (z API - OpenWeatherMap, apod.)
   - Školní prázdniny oficiální kalendář
   - Státní svátky

5. **Odvozené features:**
   - Typ dne: pracovní/víkend/svátek
   - Otevírací doba (v hodinách)
   - Je zavřeno? (boolean)

### **Fáze 2: Exploratorní analýza (EDA)**

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Základní statistiky
# Trendy v čase
# Sezónnost (roční, týdenní)
# Korelace mezi features
# Detekce outlierů
# Rozdíly ve víkendové vs. pracovní návštěvnosti
```

### **Fáze 3: Modelování**

**Doporučený postup:**

1. **Train/Test split**
   - Chronologický split (ne náhodný!)
   - Train: 2016-2024
   - Test: 2025

2. **Baseline model**
   - Jednoduchý průměr
   - Naivní predikce (hodnota před týdnem)

3. **ML modely:**
   ```python
   # Random Forest (START HERE)
   from sklearn.ensemble import RandomForestRegressor
   
   # XGBoost (pokud RF nestačí)
   from xgboost import XGBRegressor
   
   # Prophet (alternativa)
   from prophet import Prophet
   ```

4. **Hyperparameter tuning**
   - GridSearchCV nebo RandomizedSearchCV

5. **Ensemble**
   - Kombinace nejlepších modelů

### **Fáze 4: Evaluace**

**Metriky:**
- **RMSE** (Root Mean Squared Error) - hlavní metrika
- **MAE** (Mean Absolute Error) - průměrná chyba
- **MAPE** (Mean Absolute Percentage Error) - chyba v %
- **R²** score

**Vizualizace:**
- Predicted vs. Actual graf
- Residuals (chyby) v čase
- Feature importance

### **Fáze 5: Aplikace/Dashboard**

**Možnosti:**

1. **Streamlit** ⭐ (nejjednodušší)
   ```bash
   pip install streamlit
   ```
   - Rychlý vývoj
   - Interaktivní
   - Snadné nasazení

2. **Flask/FastAPI** (API)
   - Pro integraci do jiných systémů

3. **Jupyter Dashboard**
   - Pro interní použití

4. **PowerBI/Tableau**
   - Vizualizace pro management

**Funkce aplikace:**
- Výběr data/období
- Zobrazení predikce
- Vizualizace trendů
- Export výsledků
- Confidence intervals (intervaly spolehlivosti)

---

## 🛠️ Technologický stack

### **Základní stack (Python):**

```bash
# Core
pandas          # Práce s daty
numpy           # Numerické výpočty
scikit-learn    # ML modely

# Vizualizace
matplotlib
seaborn
plotly          # Interaktivní grafy

# Časové řady
statsmodels     # SARIMA
prophet         # Facebook Prophet

# Pokročilé ML
xgboost
lightgbm

# Web aplikace
streamlit       # Dashboard
flask/fastapi   # API (optional)

# Utils
joblib          # Ukládání modelů
```

### **Instalace:**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly
pip install prophet xgboost lightgbm streamlit
```

## 🎯 Minimální funkční produkt (MVP)

### **Co musí umět:**
1. ✅ Načíst historická data
2. ✅ Vytvořit features
3. ✅ Natrénovat model (Random Forest)
4. ✅ Predikovat návštěvnost pro zadané datum
5. ✅ Zobrazit predikci v jednoduchém rozhraní
6. ✅ Ukázat přesnost modelu (RMSE, MAE)

### **Nice to have (rozšíření):**
- 🔥 Integrace počasí z API
- 🔥 Predikce pro celý měsíc najednou
- 🔥 Confidence intervals
- 🔥 Porovnání více modelů
- 🔥 Automatické reporty
- 🔥 Detekce anomálií

---

## 💡 Klíčové výzvy

1. **Chybějící data o počasí**
   - Řešení: Získat z historického API (např. Visual Crossing Weather)

2. **Outliers a speciální akce**
   - V sloupci `extra` jsou svátky, ale ne všechny speciální akce
   - Řešení: Detekce anomálií, ruční označení velkých akcí

3. **COVID období (2020-2021)**
   - Data budou zkreslená
   - Řešení: Možná vyřadit nebo označit

4. **Otevírací doba se mění**
   - Ovlivňuje potenciál návštěvnosti
   - Řešení: Normalizace nebo feature "hodiny otevřeno"

5. **Sezónnost**
   - Letní vs. zimní období
   - Prázdniny
   - Řešení: Seasonality features

---

## 📝 Doporučená struktura projektu

```
Techmania/
├── data/
│   ├── raw/
│   │   └── techmania_cleaned_master.csv
│   ├── processed/
│   │   └── techmania_features.csv
│   └── external/
│       └── weather_data.csv (optional)
├── src/
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── models.py
│   └── evaluation.py
├── models/
│   └── best_model.pkl
├── app/
│   ├── app.py
│   └── templates/
│       └── index.html
├── requirements.txt
├── README.md
└── plan_reseni.md (tento soubor)
```

---

## 🚀 Jak začít

### **Krok 1: Nastavit prostředí**
```bash
cd d:\sebik_programovani\Techmania
python -m venv venv
.\venv\Scripts\activate
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### **Krok 2: Exploratorní analýza**
```python
import pandas as pd
import matplotlib.pyplot as plt

# Načíst data
df = pd.read_csv('techmania_cleaned_master.csv')
df['date'] = pd.to_datetime(df['date'])

# Základní info
print(df.info())
print(df.describe())

# Plot trendů
df.set_index('date')['total_visitors'].plot(figsize=(15,5))
plt.title('Návštěvnost v čase')
plt.show()
```

### **Krok 3: Feature Engineering**
- Vytvořit časové features
- Vytvořit lag features
- Vytvořit rolling features

### **Krok 4: První model**
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Připravit X, y
# Split data
# Train model
# Evaluate
```

---

## 🎓 Závěr a doporučení

### **🎯 Doporučení pro start:**
1. Začněte s **Random Forest** - nejjednodušší a velmi efektivní
2. Vytvořte dobré **features** (časové, lag, rolling)
3. Vyhodnoťte přesnost pomocí **RMSE** a **MAE**
4. Vytvořte jednoduchý **Streamlit dashboard**
5. Pak případně rozšiřte o počasí a pokročilejší modely

### **📊 Očekávaná přesnost:**
- **Dobrý model**: MAE ~50-100 návštěvníků, MAPE ~15-25%
- **Velmi dobrý model**: MAE <50, MAPE <15%
- **Baseline**: MAE ~150-200

---

## 📚 Užitečné zdroje

- [Scikit-learn Time Series](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble)
- [Prophet Documentation](https://facebook.github.io/prophet/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Time Series Forecasting Tutorial](https://www.kaggle.com/learn/time-series)



📊 SUMMARY - WEEKLY PREDICTIONS WITH WEATHER
==============================================================================================================

Date         Day        Visitors  Weather                               Temp   Srážky
--------------------------------------------------------------------------------------------------------------
2026-01-10   Saturday        729  Sněhové přeháňky: slabé              -3.3°C  ❄️ 0.6mm
2026-01-11   Sunday          425  Polojasno                           -10.4°C   ☀️ 0mm
2026-01-12   Monday          365  Sněžení: slabé                       -9.3°C  ❄️ 1.3mm
2026-01-13   Tuesday         439  Neznámé                               1.6°C  🌧️ 0.4mm
2026-01-14   Wednesday       428  Neznámé                               0.9°C  🌧️ 0.3mm
2026-01-15   Thursday        461  Mlha                                  1.8°C   ☀️ 0mm
2026-01-16   Friday          451  Mlha                                 -0.1°C   ☀️ 0mm
--------------------------------------------------------------------------------------------------------------
TOTAL (7 days)             3298
AVERAGE/day                 471

==============================================================================================================
✅ TESTING COMPLETE!
==============================================================================================================