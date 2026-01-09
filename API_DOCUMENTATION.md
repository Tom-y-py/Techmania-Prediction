# 📚 Techmania Prediction API - Dokumentace

## 🎯 Přehled Pipeline

### Jak funguje predikční systém?

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐      ┌─────────────┐
│   Frontend  │ ───▶ │  FastAPI     │ ───▶ │  Feature    │ ───▶ │  Ensemble   │
│  (React)    │      │  Backend     │      │ Engineering │      │  Model      │
└─────────────┘      └──────────────┘      └─────────────┘      └─────────────┘
      ▲                                                                  │
      │                                                                  │
      └──────────────────────────────────────────────────────────────────┘
                              Predikce návštěvnosti
```

### 1️⃣ **Frontend (React + TypeScript + Vite)**
- Uživatelské rozhraní pro zadávání parametrů
- Vizualizace výsledků predikce
- Port: `5173` (výchozí pro Vite dev server)

### 2️⃣ **Backend (FastAPI)**
- RESTful API pro predikce
- Načítání natrénovaných modelů při startu
- Validace vstupních dat pomocí Pydantic
- Port: `8000` (doporučený pro FastAPI)

### 3️⃣ **Feature Engineering**
- Automatické vytváření features z minimálních vstupů
- Transformace data na 40+ features pro model
- Časové features, lag features, rolling statistics, seasonality

### 4️⃣ **Ensemble Model**
- **LightGBM** (gradient boosting)
- **XGBoost** (gradient boosting)
- **CatBoost** (gradient boosting)
- Vážený průměr predikcí podle výkonu na validačních datech

---

## 🔌 API Endpointy

### **Base URL:** `http://localhost:8000`

### 📍 `GET /`
Root endpoint s přehledem API

**Response:**
```json
{
  "name": "Techmania Prediction API",
  "version": "2.0.0",
  "message": "FastAPI backend pro predikci návštěvnosti Techmanie",
  "docs": "/docs",
  "endpoints": {...}
}
```

---

### 📍 `GET /docs`
Interaktivní Swagger UI dokumentace
- Automaticky generovaná z FastAPI
- Testování API přímo v prohlížeči

---

### 📍 `GET /health`
Health check - kontrola stavu API a modelů

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": {
    "lightgbm": true,
    "xgboost": true,
    "catboost": true
  },
  "features_count": 42
}
```

---

### 📍 `GET /models/info`
Informace o načtených modelech a jejich vahách

**Response:**
```json
{
  "models": ["lightgbm", "xgboost", "catboost"],
  "ensemble_weights": {
    "lightgbm": 0.35,
    "xgboost": 0.33,
    "catboost": 0.32
  },
  "features_count": 42,
  "feature_sample": ["year", "month", "day", "day_of_week", ...]
}
```

---

### 📍 `POST /predict`
**Predikce pro konkrétní datum**

#### Request Body:
```json
{
  "date": "2026-01-15",           // POVINNÉ: Datum ve formátu YYYY-MM-DD
  "is_holiday": false,             // VOLITELNÉ: Je svátek? (default: false)
  "opening_hours": "9-17"          // VOLITELNÉ: Otevírací doba (default: "9-17")
}
```

#### Response:
```json
{
  "date": "2026-01-15",
  "predicted_visitors": 287,
  "confidence_interval": {
    "lower": 244,                  // 85% předpovědi
    "upper": 330                   // 115% předpovědi
  },
  "model_info": {
    "type": "ensemble",
    "models": ["lightgbm", "xgboost", "catboost"],
    "weights": {
      "lightgbm": 0.35,
      "xgboost": 0.33,
      "catboost": 0.32
    }
  }
}
```

---

### 📍 `POST /predict/range`
**Predikce pro časové období**

#### Request Body:
```json
{
  "start_date": "2026-01-01",     // POVINNÉ: Začátek období
  "end_date": "2026-01-31"        // POVINNÉ: Konec období
}
```

#### Response:
```json
{
  "predictions": [
    {
      "date": "2026-01-01",
      "predicted_visitors": 245
    },
    {
      "date": "2026-01-02",
      "predicted_visitors": 312
    },
    // ... jeden záznam pro každý den v období
  ],
  "total_predicted": 8934,        // Součet návštěvníků za celé období
  "average_daily": 288.2,          // Průměr návštěvníků na den
  "period_days": 31                // Počet dní v období
}
```

---

## 📊 Jaké údaje lze zadat?

### **1. Datum (POVINNÉ)**
- **Formát:** `YYYY-MM-DD` (např. `2026-01-15`)
- **Rozsah:** Jakékoliv datum (model extrapoluje do budoucnosti)
- **Doporučení:** Nejlepší přesnost pro data podobná trénovacím (2017-2025)

### **2. Je svátek? (VOLITELNÉ)**
- **Typ:** Boolean (`true` / `false`)
- **Default:** `false`
- **Význam:** 
  - `true` = Státní/náboženský svátek (Vánoce, Velikonoce, 1. máj...)
  - `false` = Běžný pracovní/víkendový den
- **Vliv na predikci:** Svátky často mají vyšší návštěvnost

### **3. Otevírací doba (VOLITELNÉ)**
- **Typ:** String (např. `"9-17"`)
- **Default:** `"9-17"`
- **Formát:** `"hodina_otevření-hodina_zavření"`
- **Příklady:**
  - `"9-17"` - standardní pracovní den
  - `"10-18"` - víkend/prodloužená doba
  - `"9-20"` - speciální akce
- **Poznámka:** Zatím limitovaný vliv na predikci (lze rozšířit)

---

## 🧠 Jak model generuje predikci?

### Proces krok za krokem:

1. **Příjem dat z frontendu**
   - Uživatel zadá: datum, svátek?, otevírací doba
   - Frontend odešle JSON POST request na `/predict`

2. **Validace dat (FastAPI + Pydantic)**
   - Kontrola formátu data
   - Kontrola typů parametrů
   - Vrácení chyby 400 při nevalidních datech

3. **Feature Engineering (automatické)**
   Z minimálních vstupů se vytvoří **40+ features**:
   
   **Časové features:**
   - `year`, `month`, `day`, `day_of_week` (0=Po, 6=Ne)
   - `week_of_year`, `quarter`, `day_of_year`
   - `is_weekend` (0/1)
   
   **Sezónní features:**
   - `is_summer_holiday` (červenec + srpen)
   - `is_winter_holiday` (23.12 - 2.1)
   - `is_school_year` (ne prázdniny)
   
   **Cyklické features:** (zachycují periodicitu)
   - `day_of_week_sin/cos`
   - `month_sin/cos`
   
   **Lag features:** (historické hodnoty)
   - `visitors_lag_1` (včera)
   - `visitors_lag_7` (před týdnem)
   - `visitors_lag_14`, `visitors_lag_30`
   
   **Rolling statistics:**
   - `visitors_rolling_mean_7/14/30` (klouzavé průměry)
   - `visitors_rolling_std_7/14/30` (směrodatné odchylky)
   - `visitors_rolling_min/max_7/14/30`
   
   **Odvozené features:**
   - `days_since_start` (trend)
   - `is_closed` (z otevírací doby)

4. **Ensemble Predikce**
   - **LightGBM** predikuje: např. 285 návštěvníků
   - **XGBoost** predikuje: např. 290 návštěvníků
   - **CatBoost** predikuje: např. 287 návštěvníků
   
   **Vážený průměr:**
   ```
   prediction = 0.35 × 285 + 0.33 × 290 + 0.32 × 287
              = 99.75 + 95.7 + 91.84
              = 287 návštěvníků
   ```

5. **Confidence Interval**
   - **Lower bound:** 85% predikce = 244 návštěvníků
   - **Upper bound:** 115% predikce = 330 návštěvníků
   - Reprezentuje nejistotu modelu

6. **Response**
   - JSON s predikcí, intervalem spolehlivosti, info o modelu
   - Frontend zobrazí výsledky ve user-friendly UI

---

## 🔮 V jakém rozsahu lze generovat predikci?

### ✅ **Jedno datum** (`/predict`)
- **Minimum:** 1 den
- **Maximum:** 1 den
- **Rychlost:** ~100-200ms
- **Use case:** Detailní predikce pro konkrétní událost/datum

### ✅ **Období** (`/predict/range`)
- **Minimum:** 1 den
- **Maximum:** Neomezeno (prakticky do ~365 dní)
- **Rychlost:** 
  - 31 dní (měsíc): ~200-400ms
  - 365 dní (rok): ~1-2s
- **Use case:** Plánování kapacit, finanční projekce, trend analýza

### ⚠️ **Omezení:**
1. **Historické predikce:**
   - Model může předpovídat i pro historická data
   - Ale přesnost je optimalizovaná pro období 2017-2025+
   
2. **Velmi vzdálená budoucnost:**
   - Predikce pro rok 2030+ jsou méně spolehlivé
   - Model extrapoluje trendy, ale neví o budoucích změnách
   - Doporučeno: max 1-2 roky dopředu

3. **Lag features pro nová data:**
   - Pro predikci budoucnosti model používá natrénované patterny
   - Lag features se nahrazují průměry/mediány z trénovacích dat

---

## 🚀 Jak spustit API?

### 1. Instalace závislostí:
```bash
pip install -r requirements.txt
```

### 2. Spuštění FastAPI serveru:
```bash
# Z root složky projektu
cd app
python app.py

# Nebo přímo s uvicorn
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Testování:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **Health check:** http://localhost:8000/health

### 4. Frontend:
```bash
cd frontend
npm install
npm run dev
# Frontend na http://localhost:5173
```

---

## 📝 Příklady použití

### cURL:
```bash
# Jednoduchá predikce
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"date": "2026-01-15", "is_holiday": false}'

# Predikce pro období
curl -X POST "http://localhost:8000/predict/range" \
  -H "Content-Type: application/json" \
  -d '{"start_date": "2026-01-01", "end_date": "2026-01-31"}'
```

### Python:
```python
import requests

# Jednoduchá predikce
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "date": "2026-01-15",
        "is_holiday": False,
        "opening_hours": "9-17"
    }
)
print(response.json())

# Predikce pro období
response = requests.post(
    "http://localhost:8000/predict/range",
    json={
        "start_date": "2026-01-01",
        "end_date": "2026-01-31"
    }
)
print(response.json())
```

### JavaScript (Frontend):
```javascript
// Již implementováno v frontend/src/api/client.ts
const result = await api.predict({
  date: '2026-01-15',
  is_holiday: false,
  opening_hours: '9-17'
});
console.log(result);
```

---

## 🎨 Frontend - Uživatelské rozhraní

### Vstupní formulář:
1. **Date picker** - Výběr data z kalendáře
2. **Select** - Je svátek? (Ano/Ne)
3. **Text input** - Otevírací doba (např. "9-17")
4. **Button** - "Předpovědět návštěvnost"

### Výstup:
- **Karta s výsledky:**
  - Datum predikce
  - Předpověděný počet návštěvníků (velké číslo)
  - Interval spolehlivosti (rozpětí)
  - Vizuálně atraktivní design s gradienty

---

## 🔒 Bezpečnost & Produkce

### Aktuálně (Development):
- CORS: Povoleno pro všechny domény (`allow_origins=["*"]`)
- Port: 8000 (lokální)
- Debug mode: Zapnutý

### Pro Produkci (TODO):
- [ ] CORS: Omezit na konkrétní domény
- [ ] HTTPS: SSL certifikát
- [ ] Rate limiting: Ochrana proti DDoS
- [ ] API klíče: Autentizace požadavků
- [ ] Monitoring: Logování, metriky
- [ ] Load balancing: Více instancí API
- [ ] Caching: Redis pro časté dotazy

---

## 📈 Výkon a Metriky

### Rychlost API:
- **Single prediction:** ~100-200ms
- **Range prediction (31 days):** ~200-400ms
- **Range prediction (365 days):** ~1-2s

### Přesnost modelu:
- **MAE (Mean Absolute Error):** ~40-60 návštěvníků
- **R² Score:** ~0.75-0.85
- **MAPE (Mean Absolute % Error):** ~15-25%

---

## 🛠️ Troubleshooting

### API nereaguje:
1. Zkontroluj, že server běží: `http://localhost:8000/health`
2. Zkontroluj port (8000 vs 5000)
3. Zkontroluj CORS nastavení

### Modely nejsou načteny:
1. Zkontroluj existenci souborů v `models/`:
   - `lightgbm_model.pkl`
   - `xgboost_model.pkl`
   - `catboost_model.pkl`
   - `ensemble_weights.pkl`
   - `feature_columns.pkl`
2. Zkontroluj konzoli při startu API

### Chyba při predikci:
1. Zkontroluj formát data (YYYY-MM-DD)
2. Zkontroluj JSON struktur request
3. Podívej se na error message v response

---

**🎉 Hotovo! Máte kompletní dokumentaci k Techmania Prediction API.**
