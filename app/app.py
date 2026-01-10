"""
FastAPI backend pro predikci návštěvnosti Techmanie.
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import date, datetime
import pandas as pd
import numpy as np
import joblib
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy.orm import Session

# Načíst proměnné prostředí
load_dotenv()

# Import databázových komponent
try:
    from database import (
        get_db, init_db, Prediction, HistoricalData, TemplateData,
        get_next_version, validate_future_date, mark_template_complete,
        get_complete_template_records, get_latest_prediction
    )
    DATABASE_ENABLED = True
except ImportError as e:
    print(f"⚠️ Database module not available: {e}")
    DATABASE_ENABLED = False

# Přidat src do path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from feature_engineering import create_features
from services import holiday_service, weather_service

# Konfigurace z .env
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', '5000'))
CORS_ORIGINS = os.getenv('CORS_ORIGINS', 'http://localhost:3000').split(',')
API_TITLE = os.getenv('API_TITLE', 'Techmania Prediction API')
API_VERSION = os.getenv('API_VERSION', '2.0.0')
DEBUG = os.getenv('DEBUG', 'true').lower() == 'true'

# Nastavení cest podle prostředí
if ENVIRONMENT == 'production':
    # Cesty v Docker kontejneru
    BASE_DIR = Path('/app')
    MODELS_DIR = BASE_DIR / 'models'
    DATA_DIR = BASE_DIR / 'data' / 'raw'
else:
    # Lokální cesty pro development
    BASE_DIR = Path(__file__).parent.parent
    MODELS_DIR = BASE_DIR / 'models'
    DATA_DIR = BASE_DIR / 'data' / 'raw'

print(f"🔧 Prostředí: {ENVIRONMENT}")
print(f"📁 Adresář modelů: {MODELS_DIR}")
print(f"📁 Adresář dat: {DATA_DIR}")

# Inicializace FastAPI
app = FastAPI(
    title=API_TITLE,
    description="API pro predikci návštěvnosti Techmanie pomocí ensemble modelu",
    version=API_VERSION,
    debug=DEBUG
)

# CORS middleware s konfigurací podle prostředí
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in CORS_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globální proměnné pro modely
models = {}
feature_columns = None
ensemble_weights = None
ensemble_info = None  # Nová: informace o typu ensemble (weighted/stacking/single_lgb)
meta_model = None  # Nová: meta-model pro stacking
historical_data = None  # Pro ukládání historických dat

# Pydantic modely pro request/response
class PredictionRequest(BaseModel):
    date: str = Field(..., description="Datum ve formátu YYYY-MM-DD", example="2026-01-15")
    is_holiday: Optional[bool] = Field(None, description="Je svátek? (None = auto-detekce)")
    opening_hours: Optional[str] = Field("9-17", description="Otevírací doba")

class WeatherInfo(BaseModel):
    temperature_mean: float
    precipitation: float
    weather_description: str
    is_nice_weather: bool

class HolidayInfo(BaseModel):
    is_holiday: bool
    holiday_name: Optional[str]

class PredictionResponse(BaseModel):
    date: str
    predicted_visitors: int
    confidence_interval: Dict[str, int]
    model_info: Dict[str, Any]
    holiday_info: HolidayInfo
    weather_info: WeatherInfo

class RangePredictionRequest(BaseModel):
    start_date: str = Field(..., description="Počáteční datum", example="2026-01-01")
    end_date: str = Field(..., description="Konečné datum", example="2026-01-31")

class DayPrediction(BaseModel):
    date: str
    predicted_visitors: int
    confidence_interval: Dict[str, int]
    holiday_info: HolidayInfo
    weather_info: WeatherInfo
    day_of_week: str
    is_weekend: bool

class RangePredictionResponse(BaseModel):
    predictions: List[DayPrediction]
    total_predicted: int
    average_daily: float
    period_days: int

class HealthResponse(BaseModel):
    status: str
    models_loaded: Dict[str, bool]
    features_count: Optional[int]

class StatsResponse(BaseModel):
    total_visitors: int
    avg_daily_visitors: float
    peak_day: str
    peak_visitors: int
    trend: float
    data_start_date: str
    data_end_date: str

class HistoricalDataPoint(BaseModel):
    date: str
    visitors: int

class HistoricalDataResponse(BaseModel):
    data: List[HistoricalDataPoint]
    start_date: str
    end_date: str
    total_days: int

class PredictionVersion(BaseModel):
    version: int
    predicted_visitors: int
    created_at: str
    model_name: str
    temperature_mean: Optional[float]
    precipitation: Optional[float]
    is_nice_weather: Optional[int]
    notes: Optional[str]

class PredictionHistoryResponse(BaseModel):
    date: str
    versions: List[PredictionVersion]
    total_versions: int

# Nové modely pro PATCH endpoint
class TemplateDataUpdate(BaseModel):
    """Data pro aktualizaci template záznamu s reálnými hodnotami"""
    date: str = Field(..., description="Datum ve formátu YYYY-MM-DD", example="2026-01-15")
    total_visitors: Optional[int] = Field(None, description="Celkový počet návštěvníků")
    school_visitors: Optional[int] = Field(None, description="Počet školních návštěvníků")
    public_visitors: Optional[int] = Field(None, description="Počet veřejných návštěvníků")
    extra: Optional[str] = Field(None, description="Extra poznámky")
    opening_hours: Optional[str] = Field(None, description="Otevírací doba")

class TemplateDataPatchResponse(BaseModel):
    """Odpověď z PATCH endpointu"""
    success: bool
    message: str
    date: str
    was_complete: bool
    is_complete: bool
    updated_fields: List[str]

class TemplateDataBatchUpdate(BaseModel):
    """Batch aktualizace více template záznamů najednou"""
    updates: List[TemplateDataUpdate] = Field(..., description="Seznam aktualizací")

class TemplateDataBatchResponse(BaseModel):
    """Odpověď z batch update"""
    success: bool
    total_processed: int
    successful_updates: int
    failed_updates: int
    details: List[TemplateDataPatchResponse]

# Načtení modelů při startu
@app.on_event("startup")
async def load_models():
    """Načte všechny natrénované modely a historická data."""
    global models, feature_columns, ensemble_weights, ensemble_info, meta_model, historical_data
    
    # Inicializovat databázi pokud je dostupná
    if DATABASE_ENABLED:
        try:
            init_db()
            print("✅ Database initialized")
        except Exception as e:
            print(f"⚠️ Database initialization failed: {e}")
    
    try:
        # Načtení jednotlivých modelů
        models['lightgbm'] = joblib.load(MODELS_DIR / 'lightgbm_model.pkl')
        models['xgboost'] = joblib.load(MODELS_DIR / 'xgboost_model.pkl')
        models['catboost'] = joblib.load(MODELS_DIR / 'catboost_model.pkl')
        
        # Načtení vah ensemble
        ensemble_weights = joblib.load(MODELS_DIR / 'ensemble_weights.pkl')
        
        # Načtení informace o typu ensemble (nové modely)
        ensemble_info_path = MODELS_DIR / 'ensemble_info.pkl'
        if ensemble_info_path.exists():
            ensemble_info = joblib.load(ensemble_info_path)
            print(f"   - Ensemble type: {ensemble_info.get('type', 'weighted').upper()}")
            print(f"   - Ensemble MAE: {ensemble_info.get('mae', 'N/A')}")
            
            # Načíst meta-model pokud je stacking
            if ensemble_info.get('type') == 'stacking':
                meta_model_path = MODELS_DIR / 'meta_model.pkl'
                if meta_model_path.exists():
                    meta_model = joblib.load(meta_model_path)
                    print(f"   - Meta-model loaded: ✅")
                else:
                    print(f"   ⚠️ Meta-model not found, falling back to weighted")
                    ensemble_info['type'] = 'weighted'
        else:
            # Starší modely bez ensemble_info = weighted
            ensemble_info = {'type': 'weighted', 'mae': None}
            print(f"   - Ensemble type: WEIGHTED (legacy)")
        
        # Načtení seznamu features
        feature_columns = joblib.load(MODELS_DIR / 'feature_columns.pkl')
        
        # Načtení historických dat pro statistiky
        try:
            # 1. Načíst historická data (do 2025)
            historical_data = pd.read_csv(DATA_DIR / 'techmania_cleaned_master.csv')
            historical_data['date'] = pd.to_datetime(historical_data['date'])
            print(f"   - Historická data: {len(historical_data)} záznamů (do {historical_data['date'].max().date()})")
            
            # 2. Načíst template pro 2026 (s předvyplněnými holiday features)
            template_2026_path = DATA_DIR / 'techmania_2026_template.csv'
            if template_2026_path.exists():
                df_2026 = pd.read_csv(template_2026_path)
                df_2026['date'] = pd.to_datetime(df_2026['date'])
                
                # Spojit s historickými daty (pokud už tam nejsou data z 2026)
                max_historical_date = historical_data['date'].max()
                df_2026_filtered = df_2026[df_2026['date'] > max_historical_date]
                
                if len(df_2026_filtered) > 0:
                    # Filtrovat jen řádky s návštěvností (pro statistiky)
                    # Pro predikce použijeme i řádky bez návštěvnosti
                    historical_data = pd.concat([historical_data, df_2026_filtered], ignore_index=True)
                    print(f"   - 2026 template: {len(df_2026_filtered)} řádků (holiday features předvyplněny)")
            else:
                print(f"   ⚠️ 2026 template nenalezen: {template_2026_path}")
        except Exception as e:
            print(f"   ⚠️ Historická data nenačtena: {e}")
            historical_data = None
        
        print("✅ Všechny modely úspěšně načteny")
        print(f"   - LightGBM: načten")
        print(f"   - XGBoost: načten")
        print(f"   - CatBoost: načten")
        print(f"   - Features: {len(feature_columns)} sloupců")
        print(f"   - Ensemble weights: {ensemble_weights}")
        
    except Exception as e:
        print(f"❌ Chyba při načítání modelů: {e}")
        raise

def make_ensemble_prediction(df: pd.DataFrame) -> np.ndarray:
    """
    Provede ensemble predikci podle typu ensemble.
    Podporuje: weighted, stacking, single_lgb
    """
    import xgboost as xgb
    
    # Predikce z každého modelu
    lgb_pred = models['lightgbm'].predict(df[feature_columns])
    
    # XGBoost potřebuje DMatrix
    dmatrix = xgb.DMatrix(df[feature_columns])
    xgb_pred = models['xgboost'].predict(dmatrix)
    
    cat_pred = models['catboost'].predict(df[feature_columns])
    
    # Rozhodnout podle typu ensemble
    ensemble_type = ensemble_info.get('type', 'weighted') if ensemble_info else 'weighted'
    
    if ensemble_type == 'single_lgb':
        # SINGLE: Použít pouze LightGBM
        ensemble_pred = lgb_pred
        print(f"   🎯 Using SINGLE LightGBM model")
        
    elif ensemble_type == 'stacking' and meta_model is not None:
        # STACKING: Použít meta-model
        meta_features = np.column_stack([lgb_pred, xgb_pred, cat_pred])
        ensemble_pred = meta_model.predict(meta_features)
        print(f"   🧠 Using STACKING ensemble with meta-model")
        
    else:
        # WEIGHTED: Vážený průměr (default)
        ensemble_pred = (
            ensemble_weights[0] * lgb_pred +
            ensemble_weights[1] * xgb_pred +
            ensemble_weights[2] * cat_pred
        )
        print(f"   ⚖️ Using WEIGHTED ensemble (weights: {ensemble_weights})")
    
    return ensemble_pred

# API Endpointy
@app.get("/", tags=["Info"])
async def root():
    """Root endpoint - informace o API."""
    return {
        "name": "Techmania Prediction API",
        "version": "2.0.0",
        "message": "FastAPI backend pro predikci návštěvnosti Techmanie",
        "docs": "/docs",
        "endpoints": {
            "/": "Tento endpoint",
            "/docs": "Interaktivní dokumentace (Swagger UI)",
            "/redoc": "Alternativní dokumentace (ReDoc)",
            "/health": "GET - Health check",
            "/predict": "POST - Predikce pro konkrétní datum",
            "/predict/range": "POST - Predikce pro období",
            "/models/info": "GET - Informace o modelech"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": {
            "lightgbm": "lightgbm" in models,
            "xgboost": "xgboost" in models,
            "catboost": "catboost" in models,
        },
        "features_count": len(feature_columns) if feature_columns is not None else None
    }

@app.get("/models/info", tags=["Info"])
async def models_info():
    """Informace o načtených modelech."""
    if not models:
        raise HTTPException(status_code=503, detail="Modely nejsou načteny")
    
    ensemble_type = ensemble_info.get('type', 'weighted') if ensemble_info else 'weighted'
    ensemble_mae = ensemble_info.get('mae') if ensemble_info else None
    
    response = {
        "models": list(models.keys()),
        "ensemble_type": ensemble_type.upper(),
        "ensemble_weights": {
            "lightgbm": float(ensemble_weights[0]),
            "xgboost": float(ensemble_weights[1]),
            "catboost": float(ensemble_weights[2])
        } if ensemble_weights is not None and len(ensemble_weights) >= 3 else None,
        "features_count": len(feature_columns) if feature_columns else 0,
        "feature_sample": feature_columns[:10] if feature_columns else []
    }
    
    if ensemble_mae is not None:
        response["validation_mae"] = float(ensemble_mae)
    
    if ensemble_type == 'stacking':
        response["meta_model"] = "Ridge Regression" if meta_model is not None else "Not loaded"
    
    return response

@app.get("/stats", response_model=StatsResponse, tags=["Statistics"])
async def get_statistics():
    """
    Získá statistiky z historických dat.
    """
    if historical_data is None:
        raise HTTPException(status_code=503, detail="Historická data nejsou dostupná")
    
    try:
        # Odfiltrovat NaN hodnoty
        clean_data = historical_data.dropna(subset=['total_visitors'])
        
        if len(clean_data) == 0:
            raise HTTPException(status_code=503, detail="Žádná platná data nejsou k dispozici")
        
        # Výpočet statistik
        total_visitors = int(clean_data['total_visitors'].sum())
        avg_daily = float(clean_data['total_visitors'].mean())
        
        # Najít den s nejvyšší návštěvností
        peak_idx = clean_data['total_visitors'].idxmax()
        peak_day = clean_data.loc[peak_idx, 'date'].strftime('%d. %B %Y')
        peak_visitors = int(clean_data.loc[peak_idx, 'total_visitors'])
        
        # Vypočítat trend (poslední měsíc vs předchozí měsíc)
        last_month = clean_data.tail(30)
        prev_month = clean_data.iloc[-60:-30] if len(clean_data) >= 60 else clean_data.head(30)
        
        if len(prev_month) > 0 and prev_month['total_visitors'].mean() > 0:
            trend = ((last_month['total_visitors'].mean() - prev_month['total_visitors'].mean()) / 
                    prev_month['total_visitors'].mean() * 100)
            trend = float(trend) if not np.isnan(trend) else 0.0
        else:
            trend = 0.0
        
        return {
            "total_visitors": total_visitors,
            "avg_daily_visitors": avg_daily,
            "peak_day": peak_day,
            "peak_visitors": peak_visitors,
            "trend": round(trend, 1),
            "data_start_date": clean_data['date'].min().strftime('%Y-%m-%d'),
            "data_end_date": clean_data['date'].max().strftime('%Y-%m-%d')
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chyba při výpočtu statistik: {str(e)}")

@app.get("/historical", response_model=HistoricalDataResponse, tags=["Statistics"])
async def get_historical_data(days: int = 30):
    """
    Získá historická data za poslední N dní.
    """
    if historical_data is None:
        raise HTTPException(status_code=503, detail="Historická data nejsou dostupná")
    
    try:
        # Odfiltrovat NaN hodnoty
        clean_data = historical_data.dropna(subset=['total_visitors'])
        
        if len(clean_data) == 0:
            raise HTTPException(status_code=503, detail="Žádná platná data nejsou k dispozici")
        
        # Získat poslední N dní
        recent_data = clean_data.tail(days).copy()
        
        data_points = []
        for _, row in recent_data.iterrows():
            visitors = row['total_visitors']
            # Další kontrola pro jistotu
            if pd.notna(visitors):
                data_points.append({
                    "date": row['date'].strftime('%Y-%m-%d'),
                    "visitors": int(visitors)
                })
        
        if len(data_points) == 0:
            raise HTTPException(status_code=503, detail="Žádná platná data pro zadané období")
        
        return {
            "data": data_points,
            "start_date": recent_data['date'].min().strftime('%Y-%m-%d'),
            "end_date": recent_data['date'].max().strftime('%Y-%m-%d'),
            "total_days": len(data_points)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chyba při načítání dat: {str(e)}")

@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(request: PredictionRequest, db: Session = Depends(get_db) if DATABASE_ENABLED else None):
    """
    Predikce návštěvnosti pro konkrétní datum.
    
    Použije ensemble model (LightGBM + XGBoost + CatBoost) pro predikci.
    Automaticky detekuje svátky a získává informace o počasí.
    Uloží predikci do databáze s verzováním.
    
    DŮLEŽITÉ: Nepřijímá predikce do minulosti (pouze budoucí data).
    """
    if not models:
        raise HTTPException(status_code=503, detail="Modely nejsou načteny")
    
    try:
        # Parsování data
        pred_date = pd.to_datetime(request.date).date()
        
        # ========== VALIDACE: ZAKÁZAT PREDIKCE DO MINULOSTI ==========
        if DATABASE_ENABLED and not validate_future_date(pred_date):
            today = date.today()
            raise HTTPException(
                status_code=400,
                detail=f"Nelze vytvořit predikci do minulosti. Požadované datum: {pred_date}, Dnešní datum: {today}. "
                       f"Predikce jsou povoleny pouze pro budoucí data."
            )
        
        # Zkusit najít datum v historických datech (může obsahovat předvyplněné holiday features)
        existing_row = None
        if historical_data is not None:
            existing_row_df = historical_data[historical_data['date'] == pd.to_datetime(pred_date)]
            if not existing_row_df.empty:
                existing_row = existing_row_df.iloc[0].to_dict()
                print(f"   ℹ️ Datum {pred_date} nalezeno v datech (použiji předvyplněné holiday features)")
        
        # Auto-detekce svátku (pokud není zadán A není v datech)
        if request.is_holiday is None:
            if existing_row and 'is_holiday' in existing_row:
                # Použít hodnotu z CSV
                is_holiday = bool(existing_row['is_holiday'])
                holiday_name = existing_row.get('extra') if pd.notna(existing_row.get('extra')) else None
                print(f"   ✓ Holiday info z CSV: is_holiday={is_holiday}")
            else:
                # Fallback na holiday_service
                holiday_info = holiday_service.get_holiday_info(pred_date)
                is_holiday = holiday_info['is_holiday']
                holiday_name = holiday_info['holiday_name']
                print(f"   ✓ Holiday info z holiday_service: is_holiday={is_holiday}")
        else:
            is_holiday = request.is_holiday
            holiday_name = None if not is_holiday else "Uživatelem zadaný svátek"
        
        # Získat informace o počasí
        weather_data = weather_service.get_weather(pred_date)
        
        # Zkontrolovat, že máme všechna potřebná data o počasí
        required_weather_fields = ['temperature_max', 'temperature_min', 'temperature_mean', 'precipitation']
        missing_fields = [field for field in required_weather_fields if field not in weather_data or weather_data[field] is None]
        
        if missing_fields:
            raise HTTPException(
                status_code=503,
                detail=f"Weather data incomplete: missing fields {missing_fields}. Cannot make prediction without real weather data."
            )
        
        # Vytvoření DataFrame pro predikci
        # Pokud máme existující řádek z CSV, použijeme ho jako základ
        if existing_row:
            # Použít existující řádek a přepsat jen weather data a opening_hours
            df_pred = pd.DataFrame([existing_row])
            df_pred['date'] = pd.to_datetime(df_pred['date'])
            
            # Aktualizovat weather data z API
            for k, v in weather_data.items():
                df_pred[k] = v
            
            # Aktualizovat opening_hours
            df_pred['opening_hours'] = request.opening_hours
            
            print(f"   ✓ Použity předvyplněné holiday features z CSV")
        else:
            # Vytvořit nový řádek (fallback pro data mimo 2026 template)
            df_pred = pd.DataFrame({
                'date': [pd.to_datetime(pred_date)],
                'total_visitors': [np.nan],  # NaN = neznámá hodnota (predikce)
                'school_visitors': [np.nan],
                'public_visitors': [np.nan],
                'extra': [holiday_name],
                'opening_hours': [request.opening_hours],
                # Všechna weather data z API (rozbalíme dictionary)
                **{k: [v] for k, v in weather_data.items()}
            })
            print(f"   ⚠️ Datum nenalezeno v CSV, vytvářím nový řádek")
        
        # create_features přidá časové features, školní prázdniny, odvozené features atd.
        df_pred = create_features(df_pred)
        
        # Vybrat pouze features, které model očekává
        available_features = [col for col in feature_columns if col in df_pred.columns]
        X_pred = df_pred[available_features].copy()
        
        # Doplnit chybějící features (např. některé weather features mohou chybět)
        missing_features = [col for col in feature_columns if col not in df_pred.columns]
        if missing_features:
            print(f"   ⚠️ Warning: Missing features: {missing_features[:10]}{'...' if len(missing_features) > 10 else ''}")
            # Doplníme nulami nebo mediány
            for col in missing_features:
                X_pred[col] = 0
        
        # Nahradit NaN hodnotami nulou
        X_pred = X_pred.fillna(0)
        
        # Ujistit se, že máme správné pořadí sloupců
        X_pred = X_pred[feature_columns]
        
        # Ensemble predikce
        prediction = make_ensemble_prediction(X_pred)[0]
        
        # Zaokrouhlení na celé číslo
        prediction = int(np.round(prediction))
        
        # Uložit predikci do databáze s verzováním
        if DATABASE_ENABLED and db is not None:
            try:
                # Získat další verzi
                version = get_next_version(db, pred_date)
                
                # Získat den v týdnu v češtině
                day_names = ['pondělí', 'úterý', 'středa', 'čtvrtek', 'pátek', 'sobota', 'neděle']
                day_of_week_cz = day_names[pred_date.weekday()]
                
                # Vytvořit nový záznam predikce
                db_prediction = Prediction(
                    prediction_date=pred_date,
                    predicted_visitors=prediction,
                    temperature_mean=weather_data.get('temperature_mean'),
                    precipitation=weather_data.get('precipitation'),
                    wind_speed_max=weather_data.get('wind_speed_max'),
                    is_rainy=1 if weather_data.get('is_rainy', False) else 0,
                    is_snowy=1 if weather_data.get('is_snowy', False) else 0,
                    is_nice_weather=1 if weather_data.get('is_nice_weather', False) else 0,
                    day_of_week=day_of_week_cz,
                    is_weekend=1 if pred_date.weekday() >= 5 else 0,
                    is_holiday=1 if is_holiday else 0,
                    model_name="ensemble",
                    confidence_lower=int(prediction * 0.85),
                    confidence_upper=int(prediction * 1.15),
                    version=version,
                    created_by="api"
                )
                db.add(db_prediction)
                db.commit()
                print(f"✅ Prediction saved to database: {pred_date} (version {version})")
            except Exception as e:
                print(f"⚠️ Failed to save prediction to database: {e}")
                db.rollback()
        
        return {
            "date": pred_date.strftime('%Y-%m-%d'),
            "predicted_visitors": prediction,
            "confidence_interval": {
                "lower": int(prediction * 0.85),
                "upper": int(prediction * 1.15)
            },
            "model_info": {
                "type": ensemble_info.get('type', 'weighted').upper() if ensemble_info else "WEIGHTED",
                "models": list(models.keys()),
                "weights": {
                    "lightgbm": float(ensemble_weights[0]),
                    "xgboost": float(ensemble_weights[1]),
                    "catboost": float(ensemble_weights[2])
                } if ensemble_weights is not None and len(ensemble_weights) >= 3 else None
            },
            "holiday_info": {
                "is_holiday": is_holiday,
                "holiday_name": holiday_name
            },
            "weather_info": {
                "temperature_mean": float(weather_data['temperature_mean']),
                "precipitation": float(weather_data['precipitation']),
                "weather_description": weather_data.get('weather_description', 'N/A'),
                "is_nice_weather": bool(weather_data.get('is_nice_weather', False))
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chyba při predikci: {str(e)}")

@app.post("/predict/range", response_model=RangePredictionResponse, tags=["Predictions"])
async def predict_range(request: RangePredictionRequest):
    """
    Predikce návštěvnosti pro časové období.
    
    Vytvoří predikce pro každý den v zadaném období.
    Automaticky stahuje weather data pro každý den z Open-Meteo API.
    
    DŮLEŽITÉ: Nepřijímá predikce do minulosti (pouze budoucí data).
    """
    if not models:
        raise HTTPException(status_code=503, detail="Modely nejsou načteny")
    
    try:
        from predict import predict_date_range
        
        start_date = pd.to_datetime(request.start_date)
        end_date = pd.to_datetime(request.end_date)
        
        if start_date > end_date:
            raise HTTPException(status_code=400, detail="start_date musí být před end_date")
        
        # ========== VALIDACE: ZAKÁZAT PREDIKCE DO MINULOSTI ==========
        if DATABASE_ENABLED:
            today = date.today()
            if start_date.date() <= today:
                raise HTTPException(
                    status_code=400,
                    detail=f"Nelze vytvořit predikci do minulosti. Start datum: {start_date.date()}, Dnešní datum: {today}. "
                           f"Predikce jsou povoleny pouze pro budoucí data. Použijte start_date > {today}."
                )
        
        # Použít funkci z predict.py která automaticky stahuje weather data
        models_dict = {
            'lgb': models['lightgbm'],
            'xgb': models['xgboost'],
            'cat': models['catboost'],
            'weights': ensemble_weights,
            'feature_cols': feature_columns,
            'ensemble_type': ensemble_info.get('type', 'weighted') if ensemble_info else 'weighted',
            'meta_model': meta_model
        }
        
        results_df = predict_date_range(start_date, end_date, models_dict)
        
        # Formátování výstupu s detailními informacemi
        predictions = []
        for _, row in results_df.iterrows():
            pred_date = row['date'].date()
            prediction_value = int(row['prediction'])
            
            # Získat informace o svátku
            holiday_info_data = holiday_service.get_holiday_info(pred_date)
            
            # Získat informace o počasí
            weather_data = weather_service.get_weather(pred_date)
            
            # Den v týdnu
            day_name = row['date'].strftime('%A')
            day_name_cs = {
                'Monday': 'Pondělí',
                'Tuesday': 'Úterý',
                'Wednesday': 'Středa',
                'Thursday': 'Čtvrtek',
                'Friday': 'Pátek',
                'Saturday': 'Sobota',
                'Sunday': 'Neděle'
            }.get(day_name, day_name)
            
            predictions.append({
                "date": row['date'].strftime('%Y-%m-%d'),
                "predicted_visitors": prediction_value,
                "confidence_interval": {
                    "lower": int(prediction_value * 0.85),
                    "upper": int(prediction_value * 1.15)
                },
                "holiday_info": {
                    "is_holiday": holiday_info_data['is_holiday'],
                    "holiday_name": holiday_info_data['holiday_name']
                },
                "weather_info": {
                    "temperature_mean": float(weather_data['temperature_mean']),
                    "precipitation": float(weather_data['precipitation']),
                    "weather_description": weather_data.get('weather_description', 'N/A'),
                    "is_nice_weather": bool(weather_data.get('is_nice_weather', False))
                },
                "day_of_week": day_name_cs,
                "is_weekend": row['date'].dayofweek >= 5
            })
        
        total = int(results_df['prediction'].sum())
        
        return {
            "predictions": predictions,
            "total_predicted": total,
            "average_daily": float(results_df['prediction'].mean()),
            "period_days": len(results_df)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"❌ Error in predict_range: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Chyba při predikci: {str(e)}")

@app.get("/predictions/history/{date_str}", response_model=PredictionHistoryResponse, tags=["Predictions"])
async def get_prediction_history(date_str: str, db: Session = Depends(get_db) if DATABASE_ENABLED else None):
    """
    Získá všechny verze predikce pro dané datum.
    
    Umožňuje vidět, jak se predikce měnila v čase.
    """
    if not DATABASE_ENABLED or db is None:
        raise HTTPException(status_code=503, detail="Database není dostupná")
    
    try:
        pred_date = pd.to_datetime(date_str).date()
        
        # Načíst všechny verze predikce pro toto datum
        predictions = db.query(Prediction)\
            .filter(Prediction.prediction_date == pred_date)\
            .order_by(Prediction.version.desc())\
            .all()
        
        if not predictions:
            raise HTTPException(status_code=404, detail=f"Žádné predikce pro datum {date_str}")
        
        versions = []
        for pred in predictions:
            versions.append({
                "version": pred.version,
                "predicted_visitors": pred.predicted_visitors,
                "created_at": pred.created_at.isoformat(),
                "model_name": pred.model_name,
                "temperature_mean": pred.temperature_mean,
                "precipitation": pred.precipitation,
                "is_nice_weather": pred.is_nice_weather,
                "notes": pred.notes
            })
        
        return {
            "date": date_str,
            "versions": versions,
            "total_versions": len(versions)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chyba při načítání historie: {str(e)}")

@app.get("/predictions/latest", tags=["Predictions"])
async def get_latest_predictions(limit: int = 20, db: Session = Depends(get_db) if DATABASE_ENABLED else None):
    """
    Získá nejnovější predikce (poslední verze pro každé datum).
    """
    if not DATABASE_ENABLED or db is None:
        raise HTTPException(status_code=503, detail="Database není dostupná")
    
    try:
        from sqlalchemy import func
        
        # Získat nejnovější verzi pro každé datum
        subquery = db.query(
            Prediction.prediction_date,
            func.max(Prediction.version).label('max_version')
        ).group_by(Prediction.prediction_date).subquery()
        
        predictions = db.query(Prediction)\
            .join(
                subquery,
                (Prediction.prediction_date == subquery.c.prediction_date) &
                (Prediction.version == subquery.c.max_version)
            )\
            .order_by(Prediction.created_at.desc())\
            .limit(limit)\
            .all()
        
        results = []
        for pred in predictions:
            results.append({
                "date": pred.prediction_date.isoformat(),
                "predicted_visitors": pred.predicted_visitors,
                "version": pred.version,
                "created_at": pred.created_at.isoformat(),
                "model_name": pred.model_name,
                "temperature_mean": pred.temperature_mean,
                "precipitation": pred.precipitation,
                "is_nice_weather": pred.is_nice_weather,
                "confidence_interval": {
                    "lower": pred.confidence_lower,
                    "upper": pred.confidence_upper
                }
            })
        
        return {
            "predictions": results,
            "count": len(results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chyba při načítání predikcí: {str(e)}")

@app.get("/data/historical", tags=["Data"])
async def get_historical_from_db(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 100,
    db: Session = Depends(get_db) if DATABASE_ENABLED else None
):
    """
    Získá historická data z databáze.
    
    Pokud není databáze dostupná, použije se fallback na CSV.
    """
    if DATABASE_ENABLED and db is not None:
        try:
            query = db.query(HistoricalData)
            
            if start_date:
                start = pd.to_datetime(start_date).date()
                query = query.filter(HistoricalData.date >= start)
            
            if end_date:
                end = pd.to_datetime(end_date).date()
                query = query.filter(HistoricalData.date <= end)
            
            records = query.order_by(HistoricalData.date.desc()).limit(limit).all()
            
            results = []
            for record in records:
                results.append({
                    "date": record.date.isoformat(),
                    "visitors": record.total_visitors,
                    "school_visitors": record.school_visitors,
                    "public_visitors": record.public_visitors,
                    "day_of_week": record.day_of_week,
                    "temperature_mean": record.temperature_mean,
                    "precipitation": record.precipitation,
                    "is_weekend": record.is_weekend,
                    "is_holiday": record.is_holiday,
                    "is_nice_weather": record.is_nice_weather
                })
            
            return {
                "source": "database",
                "data": results,
                "count": len(results)
            }
        except Exception as e:
            print(f"⚠️ Database query failed: {e}")
            # Fallback na CSV
    
    # Fallback pokud databáze není dostupná
    if historical_data is not None:
        df = historical_data.copy()
        
        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]
        
        df = df.tail(limit)
        
        results = []
        for _, row in df.iterrows():
            results.append({
                "date": row['date'].strftime('%Y-%m-%d'),
                "visitors": int(row['total_visitors'])
            })
        
        return {
            "source": "csv",
            "data": results,
            "count": len(results)
        }
    
    raise HTTPException(status_code=503, detail="Historická data nejsou dostupná")


# ========== NOVÉ ENDPOINTY PRO TEMPLATE DATA A PATCH ==========

@app.patch("/template/update", response_model=TemplateDataPatchResponse, tags=["Template"])
async def update_template_record(
    update_data: TemplateDataUpdate,
    db: Session = Depends(get_db)
):
    """
    PATCH endpoint pro aktualizaci template záznamu s reálnými daty.
    
    Když Techmania přidá reálná data do Excelu (např. návštěvnost pro 1.1.2026),
    tento endpoint detekuje změnu, aktualizuje záznam v DB a nastaví flag is_complete=True.
    
    Kompletní záznamy se pak mohou použít pro grafy a statistiky.
    """
    if not DATABASE_ENABLED:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        # Parse date
        record_date = pd.to_datetime(update_data.date).date()
        
        # Find template record
        template_record = db.query(TemplateData)\
            .filter(TemplateData.date == record_date)\
            .first()
        
        if not template_record:
            raise HTTPException(
                status_code=404,
                detail=f"Template záznam pro datum {update_data.date} nebyl nalezen"
            )
        
        # Track what was updated
        was_complete = template_record.is_complete
        updated_fields = []
        
        # Update visitor data if provided
        if update_data.total_visitors is not None:
            template_record.total_visitors = update_data.total_visitors
            updated_fields.append("total_visitors")
        
        if update_data.school_visitors is not None:
            template_record.school_visitors = update_data.school_visitors
            updated_fields.append("school_visitors")
        
        if update_data.public_visitors is not None:
            template_record.public_visitors = update_data.public_visitors
            updated_fields.append("public_visitors")
        
        if update_data.extra is not None:
            template_record.extra = update_data.extra
            updated_fields.append("extra")
        
        if update_data.opening_hours is not None:
            template_record.opening_hours = update_data.opening_hours
            updated_fields.append("opening_hours")
        
        # If total_visitors was added, mark as complete
        if update_data.total_visitors is not None and not was_complete:
            template_record.is_complete = True
            template_record.completed_at = datetime.utcnow()
            updated_fields.append("is_complete")
        
        template_record.updated_at = datetime.utcnow()
        db.commit()
        
        return TemplateDataPatchResponse(
            success=True,
            message=f"Záznam pro {update_data.date} byl úspěšně aktualizován" + 
                   (" a označen jako kompletní" if not was_complete and template_record.is_complete else ""),
            date=update_data.date,
            was_complete=was_complete,
            is_complete=template_record.is_complete,
            updated_fields=updated_fields
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Chyba při aktualizaci: {str(e)}")


@app.patch("/template/batch-update", response_model=TemplateDataBatchResponse, tags=["Template"])
async def batch_update_template_records(
    batch_data: TemplateDataBatchUpdate,
    db: Session = Depends(get_db)
):
    """
    Batch PATCH endpoint pro aktualizaci více template záznamů najednou.
    
    Umožňuje nahrát celý Excel soubor s více dny najednou.
    """
    if not DATABASE_ENABLED:
        raise HTTPException(status_code=503, detail="Database not available")
    
    results = []
    successful = 0
    failed = 0
    
    for update_data in batch_data.updates:
        try:
            # Parse date
            record_date = pd.to_datetime(update_data.date).date()
            
            # Find template record
            template_record = db.query(TemplateData)\
                .filter(TemplateData.date == record_date)\
                .first()
            
            if not template_record:
                results.append(TemplateDataPatchResponse(
                    success=False,
                    message=f"Záznam pro {update_data.date} nebyl nalezen",
                    date=update_data.date,
                    was_complete=False,
                    is_complete=False,
                    updated_fields=[]
                ))
                failed += 1
                continue
            
            # Track changes
            was_complete = template_record.is_complete
            updated_fields = []
            
            # Update fields
            if update_data.total_visitors is not None:
                template_record.total_visitors = update_data.total_visitors
                updated_fields.append("total_visitors")
            
            if update_data.school_visitors is not None:
                template_record.school_visitors = update_data.school_visitors
                updated_fields.append("school_visitors")
            
            if update_data.public_visitors is not None:
                template_record.public_visitors = update_data.public_visitors
                updated_fields.append("public_visitors")
            
            if update_data.extra is not None:
                template_record.extra = update_data.extra
                updated_fields.append("extra")
            
            if update_data.opening_hours is not None:
                template_record.opening_hours = update_data.opening_hours
                updated_fields.append("opening_hours")
            
            # Mark as complete if total_visitors was added
            if update_data.total_visitors is not None and not was_complete:
                template_record.is_complete = True
                template_record.completed_at = datetime.utcnow()
                updated_fields.append("is_complete")
            
            template_record.updated_at = datetime.utcnow()
            
            results.append(TemplateDataPatchResponse(
                success=True,
                message=f"Úspěšně aktualizováno",
                date=update_data.date,
                was_complete=was_complete,
                is_complete=template_record.is_complete,
                updated_fields=updated_fields
            ))
            successful += 1
            
        except Exception as e:
            results.append(TemplateDataPatchResponse(
                success=False,
                message=f"Chyba: {str(e)}",
                date=update_data.date,
                was_complete=False,
                is_complete=False,
                updated_fields=[]
            ))
            failed += 1
    
    # Commit all changes at once
    try:
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Chyba při ukládání: {str(e)}")
    
    return TemplateDataBatchResponse(
        success=successful > 0,
        total_processed=len(batch_data.updates),
        successful_updates=successful,
        failed_updates=failed,
        details=results
    )


@app.get("/template/status", tags=["Template"])
async def get_template_status(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    complete_only: bool = False,
    db: Session = Depends(get_db)
):
    """
    Získá status template záznamů - které mají kompletní data a které ne.
    
    Args:
        start_date: Filtr od data (YYYY-MM-DD)
        end_date: Filtr do data (YYYY-MM-DD)
        complete_only: Vrátit pouze kompletní záznamy
    """
    if not DATABASE_ENABLED:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        query = db.query(TemplateData)
        
        if start_date:
            start = pd.to_datetime(start_date).date()
            query = query.filter(TemplateData.date >= start)
        
        if end_date:
            end = pd.to_datetime(end_date).date()
            query = query.filter(TemplateData.date <= end)
        
        if complete_only:
            query = query.filter(TemplateData.is_complete == True)
        
        records = query.order_by(TemplateData.date).all()
        
        results = []
        for record in records:
            results.append({
                "date": record.date.isoformat(),
                "is_complete": record.is_complete,
                "completed_at": record.completed_at.isoformat() if record.completed_at else None,
                "total_visitors": record.total_visitors,
                "has_visitor_data": record.total_visitors is not None,
                "day_of_week": record.day_of_week,
                "is_weekend": record.is_weekend,
                "is_holiday": record.is_holiday
            })
        
        complete_count = sum(1 for r in results if r["is_complete"])
        
        return {
            "total_records": len(results),
            "complete_records": complete_count,
            "incomplete_records": len(results) - complete_count,
            "data": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chyba při načítání statusu: {str(e)}")


@app.get("/template/complete", tags=["Template"])
async def get_complete_template_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Získá všechny kompletní template záznamy (s reálnými daty).
    Tyto záznamy se mohou použít pro grafy a statistiky.
    """
    if not DATABASE_ENABLED:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        start = pd.to_datetime(start_date).date() if start_date else None
        end = pd.to_datetime(end_date).date() if end_date else None
        
        records = get_complete_template_records(db, start, end)
        
        results = []
        for record in records:
            results.append({
                "date": record.date.isoformat(),
                "total_visitors": record.total_visitors,
                "school_visitors": record.school_visitors,
                "public_visitors": record.public_visitors,
                "day_of_week": record.day_of_week,
                "temperature_mean": record.temperature_mean,
                "precipitation": record.precipitation,
                "is_weekend": record.is_weekend,
                "is_holiday": record.is_holiday,
                "is_nice_weather": record.is_nice_weather,
                "completed_at": record.completed_at.isoformat() if record.completed_at else None
            })
        
        return {
            "source": "template_data",
            "complete_records": len(results),
            "data": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chyba při načítání dat: {str(e)}")


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
