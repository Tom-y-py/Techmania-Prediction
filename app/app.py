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
        get_db, init_db, Prediction, HistoricalData, TemplateData, Event,
        get_next_version, validate_future_date, mark_template_complete,
        get_complete_template_records, get_latest_prediction,
        get_events_for_date, get_events_for_range, update_template_event_flag
    )
    DATABASE_ENABLED = True
except ImportError as e:
    print(f"⚠️ Database module not available: {e}")
    DATABASE_ENABLED = False

# Přidat src do path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from feature_engineering import create_features
from services import holiday_service, weather_service, event_scraper_service

# Konfigurace z .env
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "5000"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
API_TITLE = os.getenv("API_TITLE", "Techmania Prediction API")
API_VERSION = os.getenv("API_VERSION", "2.0.0")
DEBUG = os.getenv("DEBUG", "true").lower() == "true"

# Nastavení cest podle prostředí
if ENVIRONMENT == "production":
    # Cesty v Docker kontejneru
    BASE_DIR = Path("/app")
    MODELS_DIR = BASE_DIR / "models"
    DATA_DIR = BASE_DIR / "data" / "raw"
else:
    # Lokální cesty pro development
    BASE_DIR = Path(__file__).parent.parent
    MODELS_DIR = BASE_DIR / "models"
    DATA_DIR = BASE_DIR / "data" / "raw"

print(f"🔧 Prostředí: {ENVIRONMENT}")
print(f"📁 Adresář modelů: {MODELS_DIR}")
print(f"📁 Adresář dat: {DATA_DIR}")

# Inicializace FastAPI
app = FastAPI(
    title=API_TITLE,
    description="API pro predikci návštěvnosti Techmanie pomocí ensemble modelu",
    version=API_VERSION,
    debug=DEBUG,
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
    date: str = Field(
        ..., description="Datum ve formátu YYYY-MM-DD", example="2026-01-15"
    )
    is_holiday: Optional[bool] = Field(
        None, description="Je svátek? (None = auto-detekce)"
    )
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

# Event modely
class EventCreate(BaseModel):
    """Model pro vytvoreni eventu"""
    event_date: str = Field(..., description="Datum eventu ve formatu YYYY-MM-DD")
    title: str = Field(..., description="Nazev eventu")
    description: Optional[str] = Field(None, description="Popis eventu")
    venue: str = Field(default="Plzen", description="Misto konani")
    category: str = Field(default="custom", description="Kategorie eventu")
    expected_attendance: str = Field(default="stredni", description="Odhad navstevnosti: male/stredni/velke/masivni")
    impact_level: int = Field(default=2, ge=1, le=5, description="Vliv na navstevnost 1-5")

class EventUpdate(BaseModel):
    """Model pro update eventu"""
    title: Optional[str] = None
    description: Optional[str] = None
    venue: Optional[str] = None
    category: Optional[str] = None
    expected_attendance: Optional[str] = None
    impact_level: Optional[int] = Field(None, ge=1, le=5)
    is_active: Optional[bool] = None

class EventResponse(BaseModel):
    """Model pro odpoved s eventem"""
    id: int
    event_date: str
    title: str
    description: Optional[str]
    venue: str
    category: str
    expected_attendance: str
    source: str
    source_url: Optional[str]
    impact_level: int
    is_active: bool
    created_at: str

class EventsListResponse(BaseModel):
    """Model pro seznam eventu"""
    events: List[EventResponse]
    total_count: int
    date_range: Optional[Dict[str, str]]

class ScraperRunRequest(BaseModel):
    """Model pro spusteni scraperu"""
    start_date: str = Field(..., description="Datum od ve formatu YYYY-MM-DD")
    end_date: str = Field(..., description="Datum do ve formatu YYYY-MM-DD")
    sources: Optional[List[str]] = Field(default=["goout", "plzen.eu"], description="Seznam zdroju pro scraping")

class ScraperRunResponse(BaseModel):
    """Model pro odpoved ze scraperu"""
    success: bool
    message: str
    events_found: int
    events_saved: int
    date_range: Dict[str, str]
    sources_scraped: List[str]

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
        models["lightgbm"] = joblib.load(MODELS_DIR / "lightgbm_model.pkl")
        models["xgboost"] = joblib.load(MODELS_DIR / "xgboost_model.pkl")
        models["catboost"] = joblib.load(MODELS_DIR / "catboost_model.pkl")

        # Načtení vah ensemble
        ensemble_weights = joblib.load(MODELS_DIR / "ensemble_weights.pkl")

        # Načtení informace o typu ensemble (nové modely)
        ensemble_info_path = MODELS_DIR / "ensemble_info.pkl"
        if ensemble_info_path.exists():
            ensemble_info = joblib.load(ensemble_info_path)
            print(
                f"   - Ensemble type: {ensemble_info.get('type', 'weighted').upper()}"
            )
            print(f"   - Ensemble MAE: {ensemble_info.get('mae', 'N/A')}")

            # Načíst meta-model pokud je stacking
            if ensemble_info.get("type") == "stacking":
                meta_model_path = MODELS_DIR / "meta_model.pkl"
                if meta_model_path.exists():
                    meta_model = joblib.load(meta_model_path)
                    print(f"   - Meta-model loaded: ✅")
                else:
                    print(f"   ⚠️ Meta-model not found, falling back to weighted")
                    ensemble_info["type"] = "weighted"
        else:
            # Starší modely bez ensemble_info = weighted
            ensemble_info = {"type": "weighted", "mae": None}
            print(f"   - Ensemble type: WEIGHTED (legacy)")

        # Načtení seznamu features
        feature_columns = joblib.load(MODELS_DIR / "feature_columns.pkl")

        # Načtení historických dat z databáze pro statistiky
        if DATABASE_ENABLED:
            try:
                db_temp = SessionLocal()
                historical_records = db_temp.query(HistoricalData).all()
                
                if len(historical_records) > 0:
                    # Konvertovat na pandas DataFrame pro kompatibilitu
                    historical_data = pd.DataFrame([{
                        'date': record.date,
                        'total_visitors': record.total_visitors,
                        'school_visitors': record.school_visitors,
                        'public_visitors': record.public_visitors,
                        'is_weekend': record.is_weekend,
                        'is_holiday': record.is_holiday,
                        'is_nice_weather': record.is_nice_weather,
                        'temperature_mean': record.temperature_mean,
                        'precipitation': record.precipitation
                    } for record in historical_records])
                    historical_data['date'] = pd.to_datetime(historical_data['date'])
                    
                    print(
                        f"   - Historická data z DB: {len(historical_data)} záznamů (do {historical_data['date'].max().date()})"
                    )
                else:
                    print(f"   ⚠️ Žádná historická data v DB")
                    historical_data = None
                
                db_temp.close()
            except Exception as e:
                print(f"   ⚠️ Historická data nenačtena z DB: {e}")
                historical_data = None
        else:
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
    lgb_pred = models["lightgbm"].predict(df[feature_columns])

    # XGBoost potřebuje DMatrix
    dmatrix = xgb.DMatrix(df[feature_columns])
    xgb_pred = models["xgboost"].predict(dmatrix)

    cat_pred = models["catboost"].predict(df[feature_columns])

    # Rozhodnout podle typu ensemble
    ensemble_type = (
        ensemble_info.get("type", "weighted") if ensemble_info else "weighted"
    )

    if ensemble_type == "single_lgb":
        # SINGLE: Použít pouze LightGBM
        ensemble_pred = lgb_pred
        print(f"   🎯 Using SINGLE LightGBM model")

    elif ensemble_type == "stacking" and meta_model is not None:
        # STACKING: Použít meta-model
        meta_features = np.column_stack([lgb_pred, xgb_pred, cat_pred])
        ensemble_pred = meta_model.predict(meta_features)
        print(f"   🧠 Using STACKING ensemble with meta-model")

    else:
        # WEIGHTED: Vážený průměr (default)
        ensemble_pred = (
            ensemble_weights[0] * lgb_pred
            + ensemble_weights[1] * xgb_pred
            + ensemble_weights[2] * cat_pred
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
            "/models/info": "GET - Informace o modelech",
        },
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
        "features_count": len(feature_columns) if feature_columns is not None else None,
    }


@app.get("/models/info", tags=["Info"])
async def models_info():
    """Informace o načtených modelech."""
    if not models:
        raise HTTPException(status_code=503, detail="Modely nejsou načteny")

    ensemble_type = (
        ensemble_info.get("type", "weighted") if ensemble_info else "weighted"
    )
    ensemble_mae = ensemble_info.get("mae") if ensemble_info else None

    response = {
        "models": list(models.keys()),
        "ensemble_type": ensemble_type.upper(),
        "ensemble_weights": (
            {
                "lightgbm": float(ensemble_weights[0]),
                "xgboost": float(ensemble_weights[1]),
                "catboost": float(ensemble_weights[2]),
            }
            if ensemble_weights is not None and len(ensemble_weights) >= 3
            else None
        ),
        "features_count": len(feature_columns) if feature_columns else 0,
        "feature_sample": feature_columns[:10] if feature_columns else [],
    }

    if ensemble_mae is not None:
        response["validation_mae"] = float(ensemble_mae)

    if ensemble_type == "stacking":
        response["meta_model"] = (
            "Ridge Regression" if meta_model is not None else "Not loaded"
        )

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
        total_visitors = int(historical_data["total_visitors"].sum())
        avg_daily = float(historical_data["total_visitors"].mean())

        # Najít den s nejvyšší návštěvností
        peak_idx = historical_data["total_visitors"].idxmax()
        peak_day = historical_data.loc[peak_idx, "date"].strftime("%d. %B %Y")
        peak_visitors = int(historical_data.loc[peak_idx, "total_visitors"])

        # Vypočítat trend (poslední měsíc vs předchozí měsíc)
        last_month = historical_data.tail(30)
        prev_month = (
            historical_data.iloc[-60:-30]
            if len(historical_data) >= 60
            else historical_data.head(30)
        )

        if len(prev_month) > 0:
            trend = (
                (
                    last_month["total_visitors"].mean()
                    - prev_month["total_visitors"].mean()
                )
                / prev_month["total_visitors"].mean()
                * 100
            )
        else:
            trend = 0.0

        return {
            "total_visitors": total_visitors,
            "avg_daily_visitors": avg_daily,
            "peak_day": peak_day,
            "peak_visitors": peak_visitors,
            "trend": round(trend, 1),
            "data_start_date": historical_data["date"].min().strftime("%Y-%m-%d"),
            "data_end_date": historical_data["date"].max().strftime("%Y-%m-%d"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Chyba při výpočtu statistik: {str(e)}"
        )


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
            "start_date": recent_data["date"].min().strftime("%Y-%m-%d"),
            "end_date": recent_data["date"].max().strftime("%Y-%m-%d"),
            "total_days": len(data_points),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chyba při načítání dat: {str(e)}")


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(
    request: PredictionRequest,
    db: Session = Depends(get_db) if DATABASE_ENABLED else None,
):
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
        
        # Zkusit najít datum v template_data (může obsahovat předvyplněné holiday features)
        existing_row = None
        if DATABASE_ENABLED and db is not None:
            template_record = db.query(TemplateData).filter(
                TemplateData.date == pred_date
            ).first()
            
            if template_record:
                # Konvertovat SQLAlchemy objekt na dict
                existing_row = {
                    'date': template_record.date,
                    'is_holiday': template_record.is_holiday,
                    'extra': template_record.extra,
                    'is_spring_break': template_record.is_spring_break,
                    'is_autumn_break': template_record.is_autumn_break,
                    'is_winter_break': template_record.is_winter_break,
                    'is_easter_break': template_record.is_easter_break,
                    'is_halfyear_break': template_record.is_halfyear_break,
                    'is_summer_holiday': template_record.is_summer_holiday,
                    'is_any_school_break': template_record.is_any_school_break,
                    'school_break_type': template_record.school_break_type,
                    'days_to_next_break': template_record.days_to_next_break,
                    'days_from_last_break': template_record.days_from_last_break,
                    'is_week_before_break': template_record.is_week_before_break,
                    'is_week_after_break': template_record.is_week_after_break,
                    'season_exact': template_record.season_exact,
                    'week_position': template_record.week_position,
                    'is_month_end': template_record.is_month_end,
                    'school_week_number': template_record.school_week_number,
                    'is_bridge_day': template_record.is_bridge_day,
                    'long_weekend_length': template_record.long_weekend_length,
                    'is_event': template_record.is_event,
                }
                print(
                    f"   ℹ️ Datum {pred_date} nalezeno v DB template (použiji předvyplněné holiday features)"
                )

        # Auto-detekce svátku (pokud není zadán A není v datech)
        if request.is_holiday is None:
            if existing_row and "is_holiday" in existing_row:
                # Použít hodnotu z DB template
                is_holiday = bool(existing_row["is_holiday"])
                holiday_name = (
                    existing_row.get("extra")
                    if existing_row.get("extra") is not None
                    else None
                )
                print(f"   ✓ Holiday info z DB template: is_holiday={is_holiday}")
            else:
                # Fallback na holiday_service
                holiday_info = holiday_service.get_holiday_info(pred_date)
                is_holiday = holiday_info["is_holiday"]
                holiday_name = holiday_info["holiday_name"]
                print(f"   ✓ Holiday info z holiday_service: is_holiday={is_holiday}")
        else:
            is_holiday = request.is_holiday
            holiday_name = None if not is_holiday else "Uživatelem zadaný svátek"

        # Získat informace o počasí
        weather_data = weather_service.get_weather(pred_date)

        # Zkontrolovat, že máme všechna potřebná data o počasí
        required_weather_fields = [
            "temperature_max",
            "temperature_min",
            "temperature_mean",
            "precipitation",
        ]
        missing_fields = [
            field
            for field in required_weather_fields
            if field not in weather_data or weather_data[field] is None
        ]

        if missing_fields:
            raise HTTPException(
                status_code=503,
                detail=f"Weather data incomplete: missing fields {missing_fields}. Cannot make prediction without real weather data.",
            )

        # Vytvoření DataFrame pro predikci
        # Pokud máme existující řádek z DB, použijeme ho jako základ
        if existing_row:
            # Použít existující řádek z DB a přepsat jen weather data a opening_hours
            df_pred = pd.DataFrame([existing_row])
            df_pred["date"] = pd.to_datetime(df_pred["date"])

            # Aktualizovat weather data z API
            for k, v in weather_data.items():
                df_pred[k] = v

            # Aktualizovat opening_hours
            df_pred["opening_hours"] = request.opening_hours

            print(f"   ✓ Použity předvyplněné holiday features z DB")
        else:
            # Vytvořit nový řádek (fallback pro data mimo template_data)
            df_pred = pd.DataFrame(
                {
                    "date": [pd.to_datetime(pred_date)],
                    "total_visitors": [np.nan],  # NaN = neznámá hodnota (predikce)
                    "school_visitors": [np.nan],
                    "public_visitors": [np.nan],
                    "extra": [holiday_name],
                    "opening_hours": [request.opening_hours],
                    # Všechna weather data z API (rozbalíme dictionary)
                    **{k: [v] for k, v in weather_data.items()},
                }
            )
            print(f"   ⚠️ Datum nenalezeno v DB template, vytvářím nový řádek")

        # create_features přidá časové features, školní prázdniny, odvozené features atd.
        df_pred = create_features(df_pred)

        # Vybrat pouze features, které model očekává
        available_features = [col for col in feature_columns if col in df_pred.columns]
        X_pred = df_pred[available_features].copy()

        # Doplnit chybějící features (např. některé weather features mohou chybět)
        missing_features = [
            col for col in feature_columns if col not in df_pred.columns
        ]
        if missing_features:
            print(
                f"   ⚠️ Warning: Missing features: {missing_features[:10]}{'...' if len(missing_features) > 10 else ''}"
            )
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
                day_names = [
                    "pondělí",
                    "úterý",
                    "středa",
                    "čtvrtek",
                    "pátek",
                    "sobota",
                    "neděle",
                ]
                day_of_week_cz = day_names[pred_date.weekday()]

                # Vytvořit nový záznam predikce
                db_prediction = Prediction(
                    prediction_date=pred_date,
                    predicted_visitors=prediction,
                    temperature_mean=weather_data.get("temperature_mean"),
                    precipitation=weather_data.get("precipitation"),
                    wind_speed_max=weather_data.get("wind_speed_max"),
                    is_rainy=1 if weather_data.get("is_rainy", False) else 0,
                    is_snowy=1 if weather_data.get("is_snowy", False) else 0,
                    is_nice_weather=(
                        1 if weather_data.get("is_nice_weather", False) else 0
                    ),
                    day_of_week=day_of_week_cz,
                    is_weekend=1 if pred_date.weekday() >= 5 else 0,
                    is_holiday=1 if is_holiday else 0,
                    model_name="ensemble",
                    confidence_lower=int(prediction * 0.85),
                    confidence_upper=int(prediction * 1.15),
                    version=version,
                    created_by="api",
                )
                db.add(db_prediction)
                db.commit()
                print(
                    f"✅ Prediction saved to database: {pred_date} (version {version})"
                )
            except Exception as e:
                print(f"⚠️ Failed to save prediction to database: {e}")
                db.rollback()

        return {
            "date": pred_date.strftime("%Y-%m-%d"),
            "predicted_visitors": prediction,
            "confidence_interval": {
                "lower": int(prediction * 0.85),
                "upper": int(prediction * 1.15),
            },
            "model_info": {
                "type": (
                    ensemble_info.get("type", "weighted").upper()
                    if ensemble_info
                    else "WEIGHTED"
                ),
                "models": list(models.keys()),
                "weights": (
                    {
                        "lightgbm": float(ensemble_weights[0]),
                        "xgboost": float(ensemble_weights[1]),
                        "catboost": float(ensemble_weights[2]),
                    }
                    if ensemble_weights is not None and len(ensemble_weights) >= 3
                    else None
                ),
            },
            "holiday_info": {"is_holiday": is_holiday, "holiday_name": holiday_name},
            "weather_info": {
                "temperature_mean": float(weather_data["temperature_mean"]),
                "precipitation": float(weather_data["precipitation"]),
                "weather_description": weather_data.get("weather_description", "N/A"),
                "is_nice_weather": bool(weather_data.get("is_nice_weather", False)),
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chyba při predikci: {str(e)}")


@app.post(
    "/predict/range", response_model=RangePredictionResponse, tags=["Predictions"]
)
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
            "lgb": models["lightgbm"],
            "xgb": models["xgboost"],
            "cat": models["catboost"],
            "weights": ensemble_weights,
            "feature_cols": feature_columns,
            "ensemble_type": (
                ensemble_info.get("type", "weighted") if ensemble_info else "weighted"
            ),
            "meta_model": meta_model,
        }

        results_df = predict_date_range(start_date, end_date, models_dict)

        # Formátování výstupu s detailními informacemi
        predictions = []
        for _, row in results_df.iterrows():
            pred_date = row["date"].date()
            prediction_value = int(row["prediction"])

            # Získat informace o svátku
            holiday_info_data = holiday_service.get_holiday_info(pred_date)

            # Získat informace o počasí
            weather_data = weather_service.get_weather(pred_date)

            # Den v týdnu
            day_name = row["date"].strftime("%A")
            day_name_cs = {
                "Monday": "Pondělí",
                "Tuesday": "Úterý",
                "Wednesday": "Středa",
                "Thursday": "Čtvrtek",
                "Friday": "Pátek",
                "Saturday": "Sobota",
                "Sunday": "Neděle",
            }.get(day_name, day_name)

            predictions.append(
                {
                    "date": row["date"].strftime("%Y-%m-%d"),
                    "predicted_visitors": prediction_value,
                    "confidence_interval": {
                        "lower": int(prediction_value * 0.85),
                        "upper": int(prediction_value * 1.15),
                    },
                    "holiday_info": {
                        "is_holiday": holiday_info_data["is_holiday"],
                        "holiday_name": holiday_info_data["holiday_name"],
                    },
                    "weather_info": {
                        "temperature_mean": float(weather_data["temperature_mean"]),
                        "precipitation": float(weather_data["precipitation"]),
                        "weather_description": weather_data.get(
                            "weather_description", "N/A"
                        ),
                        "is_nice_weather": bool(
                            weather_data.get("is_nice_weather", False)
                        ),
                    },
                    "day_of_week": day_name_cs,
                    "is_weekend": row["date"].dayofweek >= 5,
                }
            )

        total = int(results_df["prediction"].sum())

        return {
            "predictions": predictions,
            "total_predicted": total,
            "average_daily": float(results_df["prediction"].mean()),
            "period_days": len(results_df),
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback

        print(f"❌ Error in predict_range: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Chyba při predikci: {str(e)}")


@app.get(
    "/predictions/history/{date_str}",
    response_model=PredictionHistoryResponse,
    tags=["Predictions"],
)
async def get_prediction_history(
    date_str: str, db: Session = Depends(get_db) if DATABASE_ENABLED else None
):
    """
    Získá všechny verze predikce pro dané datum.

    Umožňuje vidět, jak se predikce měnila v čase.
    """
    if not DATABASE_ENABLED or db is None:
        raise HTTPException(status_code=503, detail="Database není dostupná")

    try:
        pred_date = pd.to_datetime(date_str).date()

        # Načíst všechny verze predikce pro toto datum
        predictions = (
            db.query(Prediction)
            .filter(Prediction.prediction_date == pred_date)
            .order_by(Prediction.version.desc())
            .all()
        )

        if not predictions:
            raise HTTPException(
                status_code=404, detail=f"Žádné predikce pro datum {date_str}"
            )

        versions = []
        for pred in predictions:
            versions.append(
                {
                    "version": pred.version,
                    "predicted_visitors": pred.predicted_visitors,
                    "created_at": pred.created_at.isoformat(),
                    "model_name": pred.model_name,
                    "temperature_mean": pred.temperature_mean,
                    "precipitation": pred.precipitation,
                    "is_nice_weather": pred.is_nice_weather,
                    "notes": pred.notes,
                }
            )

        return {"date": date_str, "versions": versions, "total_versions": len(versions)}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Chyba při načítání historie: {str(e)}"
        )


@app.get("/predictions/latest", tags=["Predictions"])
async def get_latest_predictions(
    limit: int = 20, db: Session = Depends(get_db) if DATABASE_ENABLED else None
):
    """
    Získá nejnovější predikce (poslední verze pro každé datum).
    """
    if not DATABASE_ENABLED or db is None:
        raise HTTPException(status_code=503, detail="Database není dostupná")

    try:
        from sqlalchemy import func

        # Získat nejnovější verzi pro každé datum
        subquery = (
            db.query(
                Prediction.prediction_date,
                func.max(Prediction.version).label("max_version"),
            )
            .group_by(Prediction.prediction_date)
            .subquery()
        )

        predictions = (
            db.query(Prediction)
            .join(
                subquery,
                (Prediction.prediction_date == subquery.c.prediction_date)
                & (Prediction.version == subquery.c.max_version),
            )
            .order_by(Prediction.created_at.desc())
            .limit(limit)
            .all()
        )

        results = []
        for pred in predictions:
            results.append(
                {
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
                        "upper": pred.confidence_upper,
                    },
                }
            )

        return {"predictions": results, "count": len(results)}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Chyba při načítání predikcí: {str(e)}"
        )


@app.get("/data/historical", tags=["Data"])
async def get_historical_from_db(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 100,
    db: Session = Depends(get_db) if DATABASE_ENABLED else None,
):
    """
    Získá historická data z databáze.
    """
    if not DATABASE_ENABLED or db is None:
        raise HTTPException(status_code=503, detail="Databáze není dostupná")
    
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
            results.append(
                {
                    "date": record.date.isoformat(),
                    "visitors": record.total_visitors,
                    "school_visitors": record.school_visitors,
                    "public_visitors": record.public_visitors,
                    "day_of_week": record.day_of_week,
                    "temperature_mean": record.temperature_mean,
                    "precipitation": record.precipitation,
                    "is_weekend": record.is_weekend,
                    "is_holiday": record.is_holiday,
                    "is_nice_weather": record.is_nice_weather,
                }
            )

        return {"source": "database", "data": results, "count": len(results)}
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Chyba při načítání historických dat z databáze: {str(e)}"
        )


@app.get("/analytics/correlation", tags=["Analytics"])
async def get_correlation_analysis():
    """
    Získá korelační analýzu mezi návštěvností a různými faktory.
    """
    if historical_data is None:
        raise HTTPException(status_code=503, detail="Historická data nejsou dostupná")

    try:
        # Výpočet korelací pouze s dostupnými daty
        correlations = {}

        # Korelace s víkendy
        if "is_weekend" in historical_data.columns:
            weekend_data = historical_data[historical_data["is_weekend"] == 1]
            weekday_data = historical_data[historical_data["is_weekend"] == 0]
            if len(weekend_data) > 0 and len(weekday_data) > 0:
                weekend_avg = float(weekend_data["total_visitors"].mean())
                weekday_avg = float(weekday_data["total_visitors"].mean())
                correlations["weekend_impact"] = (
                    round(weekend_avg / weekday_avg, 2) if weekday_avg > 0 else 1.0
                )
            else:
                correlations["weekend_impact"] = 1.0
        else:
            correlations["weekend_impact"] = 1.0

        # Korelace se svátky
        if "is_holiday" in historical_data.columns:
            holiday_data = historical_data[historical_data["is_holiday"] == 1]
            regular_data = historical_data[historical_data["is_holiday"] == 0]
            if len(holiday_data) > 0 and len(regular_data) > 0:
                holiday_avg = float(holiday_data["total_visitors"].mean())
                regular_avg = float(regular_data["total_visitors"].mean())
                correlations["holiday_impact"] = (
                    round(holiday_avg / regular_avg, 2) if regular_avg > 0 else 1.0
                )
            else:
                correlations["holiday_impact"] = 1.0
        else:
            correlations["holiday_impact"] = 1.0

        # Pro weather korelaci použijeme měsíční průměry (léto vs zima)
        # Léto = květen-září (měsíce 5-9), Zima = listopad-březen (měsíce 11,12,1,2,3)
        historical_data["month"] = pd.to_datetime(historical_data["date"]).dt.month
        summer_data = historical_data[historical_data["month"].isin([5, 6, 7, 8, 9])]
        winter_data = historical_data[historical_data["month"].isin([11, 12, 1, 2, 3])]

        if len(summer_data) > 0 and len(winter_data) > 0:
            summer_avg = float(summer_data["total_visitors"].mean())
            winter_avg = float(winter_data["total_visitors"].mean())
            # Normalizované jako korelace (-1 to 1)
            correlations["weather_correlation"] = round(
                (summer_avg - winter_avg) / (summer_avg + winter_avg), 2
            )
            # Temperature korelace (simulace na základě sezónnosti)
            correlations["temperature_correlation"] = round(
                correlations["weather_correlation"] * 0.85, 2
            )
        else:
            correlations["weather_correlation"] = 0.0
            correlations["temperature_correlation"] = 0.0

        return {
            "correlations": correlations,
            "description": "Korelační koeficienty a multiplikátory vypočtené z historických dat",
        }
    except Exception as e:
        print(f"Error in correlation analysis: {str(e)}")
        import traceback

        print(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"Chyba při výpočtu korelací: {str(e)}"
        )


@app.get("/analytics/seasonality", tags=["Analytics"])
async def get_seasonality_analysis():
    """
    Získá sezónní vzorce návštěvnosti.
    """
    if historical_data is None:
        raise HTTPException(status_code=503, detail="Historická data nejsou dostupná")

    try:
        # Průměr podle dne v týdnu
        weekday_pattern = {}
        day_names_en = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        day_names_cs = [
            "Pondělí",
            "Úterý",
            "Středa",
            "Čtvrtek",
            "Pátek",
            "Sobota",
            "Neděle",
        ]

        for day in range(7):
            day_data = historical_data[historical_data["date"].dt.dayofweek == day]
            if len(day_data) > 0:
                weekday_pattern[day_names_cs[day]] = float(
                    day_data["total_visitors"].mean()
                )

        # Průměr podle měsíce
        monthly_pattern = {}
        month_names_cs = [
            "Leden",
            "Únor",
            "Březen",
            "Duben",
            "Květen",
            "Červen",
            "Červenec",
            "Srpen",
            "Září",
            "Říjen",
            "Listopad",
            "Prosinec",
        ]

        for month in range(1, 13):
            month_data = historical_data[historical_data["date"].dt.month == month]
            if len(month_data) > 0:
                monthly_pattern[month_names_cs[month - 1]] = float(
                    month_data["total_visitors"].mean()
                )

        # Porovnání svátků vs běžné dny
        holiday_vs_regular = {"holiday_avg": 0, "regular_avg": 0, "difference": 0}

        if "is_holiday" in historical_data.columns:
            holiday_days = historical_data[historical_data["is_holiday"] == True]
            regular_days = historical_data[historical_data["is_holiday"] == False]

            if len(holiday_days) > 0 and len(regular_days) > 0:
                holiday_avg = float(holiday_days["total_visitors"].mean())
                regular_avg = float(regular_days["total_visitors"].mean())

                holiday_vs_regular = {
                    "holiday_avg": holiday_avg,
                    "regular_avg": regular_avg,
                    "difference": holiday_avg - regular_avg,
                }

        return {
            "by_weekday": weekday_pattern,
            "by_month": monthly_pattern,
            "holiday_vs_regular": holiday_vs_regular,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Chyba při výpočtu sezónnosti: {str(e)}"
        )


@app.get("/analytics/heatmap", tags=["Analytics"])
async def get_calendar_heatmap(year: Optional[int] = None):
    """
    Získá data pro kalendářní heatmapu.
    Pokud není specifikován rok, vrátí data pro všechny dostupné roky.
    """
    if historical_data is None:
        raise HTTPException(status_code=503, detail="Historická data nejsou dostupná")

    try:
        # Zajistit, že date je datetime
        historical_data["date"] = pd.to_datetime(historical_data["date"])

        if year is not None:
            # Filtrovat data pro daný rok
            year_data = historical_data[historical_data["date"].dt.year == year].copy()

            if len(year_data) == 0:
                return {
                    "year": year,
                    "data": [],
                    "min_visitors": 0,
                    "max_visitors": 0,
                    "available_years": sorted(
                        historical_data["date"].dt.year.unique().tolist()
                    ),
                }

            # Připravit data pro heatmapu
            heatmap_data = []
            for _, row in year_data.iterrows():
                heatmap_data.append(
                    {
                        "date": row["date"].strftime("%Y-%m-%d"),
                        "visitors": int(row["total_visitors"]),
                    }
                )

            return {
                "year": year,
                "data": heatmap_data,
                "min_visitors": int(year_data["total_visitors"].min()),
                "max_visitors": int(year_data["total_visitors"].max()),
                "available_years": sorted(
                    historical_data["date"].dt.year.unique().tolist()
                ),
            }
        else:
            # Vrátit data pro všechny roky
            all_data = []
            for _, row in historical_data.iterrows():
                all_data.append(
                    {
                        "date": row["date"].strftime("%Y-%m-%d"),
                        "visitors": int(row["total_visitors"]),
                    }
                )

            return {
                "data": all_data,
                "min_visitors": int(historical_data["total_visitors"].min()),
                "max_visitors": int(historical_data["total_visitors"].max()),
                "available_years": sorted(
                    historical_data["date"].dt.year.unique().tolist()
                ),
            }
    except Exception as e:
        print(f"Error in heatmap: {str(e)}")
        import traceback

        print(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"Chyba při generování heatmapy: {str(e)}"
        )


@app.post("/scraper/events/run", response_model=ScraperRunResponse, tags=["Events"])
async def run_event_scraper(
    request: ScraperRunRequest,
    db: Session = Depends(get_db) if DATABASE_ENABLED else None
):
    """
    Spusti scraper pro hledani eventu v Plzni a okoli.
    
    Scrape eventy z vybranych zdroju (GoOut, Plzen.eu) a ulozi je do databaze.
    Automaticky aktualizuje is_event flag v template_data tabulce.
    """
    if not DATABASE_ENABLED or db is None:
        raise HTTPException(status_code=503, detail="Databaze neni dostupna")
    
    try:
        # Parsovat data
        start_date = pd.to_datetime(request.start_date).date()
        end_date = pd.to_datetime(request.end_date).date()
        
        if start_date > end_date:
            raise HTTPException(status_code=400, detail="start_date musi byt pred end_date")
        
        # Spustit scraper
        print(f"Spoustim scraper pro {start_date} - {end_date}")
        scraped_events = event_scraper_service.scrape_all_sources(start_date, end_date)
        
        # Ulozit eventy do databaze
        events_saved = 0
        for event_data in scraped_events:
            try:
                # Kontrola jestli event uz neexistuje
                existing = db.query(Event).filter(
                    Event.event_date == event_data['event_date'],
                    Event.title == event_data['title']
                ).first()
                
                if not existing:
                    # Vytvorit novy event
                    new_event = Event(
                        event_date=event_data['event_date'],
                        title=event_data['title'],
                        description=event_data.get('description'),
                        venue=event_data.get('venue', 'Plzen'),
                        category=event_data.get('category', 'obecne'),
                        expected_attendance=event_data.get('expected_attendance', 'stredni'),
                        source=event_data['source'],
                        source_url=event_data.get('source_url'),
                        impact_level=event_data.get('impact_level', 2),
                        is_active=True
                    )
                    db.add(new_event)
                    events_saved += 1
                    
                    # Aktualizovat template_data is_event flag
                    update_template_event_flag(db, event_data['event_date'])
            
            except Exception as e:
                print(f"Chyba pri ukladani eventu: {e}")
                continue
        
        db.commit()
        
        return {
            "success": True,
            "message": f"Scraping dokoncen. Nalezeno {len(scraped_events)} eventu, ulozeno {events_saved} novych.",
            "events_found": len(scraped_events),
            "events_saved": events_saved,
            "date_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "sources_scraped": request.sources
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chyba pri scrapingu: {str(e)}")


@app.get("/events", response_model=EventsListResponse, tags=["Events"])
async def get_events(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    category: Optional[str] = None,
    min_impact: Optional[int] = None,
    limit: int = 100,
    db: Session = Depends(get_db) if DATABASE_ENABLED else None
):
    """
    Ziska seznam eventu z databaze.
    
    Umoznuje filtrovani podle data, kategorie a impact levelu.
    """
    if not DATABASE_ENABLED or db is None:
        raise HTTPException(status_code=503, detail="Databaze neni dostupna")
    
    try:
        query = db.query(Event).filter(Event.is_active == True)
        
        # Filtry
        if start_date:
            query = query.filter(Event.event_date >= pd.to_datetime(start_date).date())
        if end_date:
            query = query.filter(Event.event_date <= pd.to_datetime(end_date).date())
        if category:
            query = query.filter(Event.category == category)
        if min_impact:
            query = query.filter(Event.impact_level >= min_impact)
        
        # Seradit podle data
        events = query.order_by(Event.event_date).limit(limit).all()
        
        # Formatovat odpoved
        events_list = []
        for event in events:
            events_list.append({
                "id": event.id,
                "event_date": event.event_date.isoformat(),
                "title": event.title,
                "description": event.description,
                "venue": event.venue,
                "category": event.category,
                "expected_attendance": event.expected_attendance,
                "source": event.source,
                "source_url": event.source_url,
                "impact_level": event.impact_level,
                "is_active": event.is_active,
                "created_at": event.created_at.isoformat()
            })
        
        date_range_result = None
        if events:
            date_range_result = {
                "start": events[0]["event_date"],
                "end": events[-1]["event_date"]
            }
        
        return {
            "events": events_list,
            "total_count": len(events_list),
            "date_range": date_range_result
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chyba pri nacitani eventu: {str(e)}")


@app.get("/events/{date_str}", tags=["Events"])
async def get_events_for_date(
    date_str: str,
    db: Session = Depends(get_db) if DATABASE_ENABLED else None
):
    """
    Ziska vsechny eventy pro konkretni datum.
    
    Vrati seznam vsech aktivnich eventu pro dane datum.
    """
    if not DATABASE_ENABLED or db is None:
        raise HTTPException(status_code=503, detail="Databaze neni dostupna")
    
    try:
        event_date = pd.to_datetime(date_str).date()
        events = get_events_for_date(db, event_date)
        
        events_list = []
        for event in events:
            events_list.append({
                "id": event.id,
                "event_date": event.event_date.isoformat(),
                "title": event.title,
                "description": event.description,
                "venue": event.venue,
                "category": event.category,
                "expected_attendance": event.expected_attendance,
                "source": event.source,
                "source_url": event.source_url,
                "impact_level": event.impact_level,
                "is_active": event.is_active,
                "created_at": event.created_at.isoformat()
            })
        
        return {
            "date": date_str,
            "events": events_list,
            "count": len(events_list)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chyba pri nacitani eventu: {str(e)}")


@app.post("/events", response_model=EventResponse, tags=["Events"])
async def create_event(
    event: EventCreate,
    db: Session = Depends(get_db) if DATABASE_ENABLED else None
):
    """
    Manualne vytvori novy event.
    
    Umoznuje rucni pridani eventu, ktery nebyl nalezen scraperem.
    """
    if not DATABASE_ENABLED or db is None:
        raise HTTPException(status_code=503, detail="Databaze neni dostupna")
    
    try:
        event_date = pd.to_datetime(event.event_date).date()
        
        # Vytvorit novy event
        new_event = Event(
            event_date=event_date,
            title=event.title,
            description=event.description,
            venue=event.venue,
            category=event.category,
            expected_attendance=event.expected_attendance,
            source='manual',
            source_url=None,
            impact_level=event.impact_level,
            is_active=True
        )
        
        db.add(new_event)
        db.commit()
        db.refresh(new_event)
        
        # Aktualizovat template_data is_event flag
        update_template_event_flag(db, event_date)
        
        return {
            "id": new_event.id,
            "event_date": new_event.event_date.isoformat(),
            "title": new_event.title,
            "description": new_event.description,
            "venue": new_event.venue,
            "category": new_event.category,
            "expected_attendance": new_event.expected_attendance,
            "source": new_event.source,
            "source_url": new_event.source_url,
            "impact_level": new_event.impact_level,
            "is_active": new_event.is_active,
            "created_at": new_event.created_at.isoformat()
        }
    
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Chyba pri vytvareni eventu: {str(e)}")


@app.patch("/events/{event_id}", response_model=EventResponse, tags=["Events"])
async def update_event(
    event_id: int,
    event_update: EventUpdate,
    db: Session = Depends(get_db) if DATABASE_ENABLED else None
):
    """
    Aktualizuje existujici event.
    
    Umoznuje zmenit detaily eventu.
    """
    if not DATABASE_ENABLED or db is None:
        raise HTTPException(status_code=503, detail="Databaze neni dostupna")
    
    try:
        event = db.query(Event).filter(Event.id == event_id).first()
        
        if not event:
            raise HTTPException(status_code=404, detail=f"Event s ID {event_id} nenalezen")
        
        # Aktualizovat pole
        update_data = event_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(event, field, value)
        
        db.commit()
        db.refresh(event)
        
        # Aktualizovat template_data is_event flag
        update_template_event_flag(db, event.event_date)
        
        return {
            "id": event.id,
            "event_date": event.event_date.isoformat(),
            "title": event.title,
            "description": event.description,
            "venue": event.venue,
            "category": event.category,
            "expected_attendance": event.expected_attendance,
            "source": event.source,
            "source_url": event.source_url,
            "impact_level": event.impact_level,
            "is_active": event.is_active,
            "created_at": event.created_at.isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Chyba pri aktualizaci eventu: {str(e)}")


@app.delete("/events/{event_id}", tags=["Events"])
async def delete_event(
    event_id: int,
    db: Session = Depends(get_db) if DATABASE_ENABLED else None
):
    """
    Smaze event (nastavi is_active = False).
    
    Soft delete - event zustane v databazi, ale nebude se zobrazovat.
    """
    if not DATABASE_ENABLED or db is None:
        raise HTTPException(status_code=503, detail="Databaze neni dostupna")
    
    try:
        event = db.query(Event).filter(Event.id == event_id).first()
        
        if not event:
            raise HTTPException(status_code=404, detail=f"Event s ID {event_id} nenalezen")
        
        event_date = event.event_date
        event.is_active = False
        
        db.commit()
        
        # Aktualizovat template_data is_event flag
        update_template_event_flag(db, event_date)
        
        return {
            "success": True,
            "message": f"Event {event_id} byl deaktivovan",
            "event_id": event_id
        }
    
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Chyba pri mazani eventu: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
