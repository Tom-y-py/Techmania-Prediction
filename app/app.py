"""
FastAPI backend pro predikci návštěvnosti Techmanie.
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
import joblib
import sys
import os
import json
import math
from pathlib import Path
from sqlalchemy.orm import Session

# Custom JSON encoder pro NaN hodnoty
class NaNSafeJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return super().default(obj)

def sanitize_for_json(obj):
    """Rekurzivně nahradí NaN a Inf hodnoty None pro JSON serializaci."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, (np.floating, np.integer)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    return obj

# Import centrální konfigurace (načítá .env automaticky)
from config import (
    config, ENVIRONMENT, HOST, PORT, CORS_ORIGINS,
    API_TITLE, API_VERSION, DEBUG, BASE_DIR, MODELS_DIR, DATA_DIR
)

# Vypsat info o konfiguraci
config.print_info()

# Import databázových komponent
try:
    from database import (
        get_db, init_db, SessionLocal, Prediction, HistoricalData, TemplateData, Event,
        get_next_version, validate_future_date, mark_template_complete,
        get_complete_template_records, get_latest_prediction,
        get_events_for_date, get_events_for_range, update_template_event_flag
    )
    from init_db import load_historical_data, load_template_data
    DATABASE_ENABLED = True
except ImportError as e:
    print(f"⚠️ Database module not available: {e}")
    DATABASE_ENABLED = False

# Přidat src do path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Import V3 feature engineering (s event features!)
try:
    from feature_engineering_v3 import create_features
    FEATURE_ENGINEERING_V3 = True
    print("✅ Using feature_engineering_v3 (with event features)")
except ImportError:
    from feature_engineering import create_features
    FEATURE_ENGINEERING_V3 = False
    print("⚠️ Fallback to legacy feature_engineering")

from services import holiday_service, weather_service, event_scraper_service

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


# ==================== AI CHAT ====================

class ChatMessage(BaseModel):
    message: str = Field(..., description="Zpráva od uživatele")
    history: Optional[List[Dict[str, str]]] = Field(default=None, description="Historie konverzace")


class ChatResponse(BaseModel):
    response: str
    context_used: bool = True


@app.post("/chat", tags=["AI Chat"])
async def chat_endpoint(
    request: ChatMessage,
    db: Session = Depends(get_db) if DATABASE_ENABLED else None
):
    """
    AI Chat endpoint pro dotazy na data o návštěvnosti.
    Vrací odpověď jako streaming (Server-Sent Events).
    """
    try:
        from chat import chat_stream
        
        def generate():
            try:
                for chunk in chat_stream(request.message, db, request.history):
                    # SSE formát
                    yield f"data: {json.dumps({'content': chunk})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'content': f'❌ Chyba: {str(e)}'})}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Chat modul není dostupný: {str(e)}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Chyba chatu: {str(e)}")


@app.post("/chat/sync", response_model=ChatResponse, tags=["AI Chat"])
async def chat_sync_endpoint(
    request: ChatMessage,
    db: Session = Depends(get_db) if DATABASE_ENABLED else None
):
    """
    Synchronní verze AI chatu (bez streamování).
    Vrací celou odpověď najednou.
    """
    try:
        from chat import chat_sync
        
        response = chat_sync(request.message, db, request.history)
        return ChatResponse(response=response)
    except ImportError:
        raise HTTPException(status_code=500, detail="Chat modul není dostupný")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chyba chatu: {str(e)}")


@app.get("/chat/tools", tags=["AI Chat"])
async def get_chat_tools():
    """
    Vrátí seznam dostupných MCP nástrojů pro AI chat.
    """
    try:
        from mcp_tools import MCP_TOOLS
        
        tools_summary = []
        for tool in MCP_TOOLS:
            func = tool.get("function", {})
            tools_summary.append({
                "name": func.get("name"),
                "description": func.get("description"),
                "parameters": list(func.get("parameters", {}).get("properties", {}).keys())
            })
        
        return {
            "tools_count": len(MCP_TOOLS),
            "tools": tools_summary
        }
    except ImportError:
        raise HTTPException(status_code=500, detail="MCP tools modul není dostupný")

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
        # Načtení V3 modelů (nové modely s event features a 85 features)
        v3_models_exist = (
            (MODELS_DIR / "lightgbm_v3.pkl").exists() and
            (MODELS_DIR / "xgboost_v3.pkl").exists() and
            (MODELS_DIR / "catboost_v3.pkl").exists() and
            (MODELS_DIR / "feature_names_v3.pkl").exists()
        )
        
        if v3_models_exist:
            # Načíst V3 modely
            models["lightgbm"] = joblib.load(MODELS_DIR / "lightgbm_v3.pkl")
            models["xgboost"] = joblib.load(MODELS_DIR / "xgboost_v3.pkl")
            models["catboost"] = joblib.load(MODELS_DIR / "catboost_v3.pkl")
            ensemble_weights = joblib.load(MODELS_DIR / "ensemble_weights_v3.pkl")
            feature_columns = joblib.load(MODELS_DIR / "feature_names_v3.pkl")
            
            # Načíst Google Trend predictor pokud existuje
            trend_predictor_path = MODELS_DIR / "google_trend_predictor_v3.pkl"
            if trend_predictor_path.exists():
                models["google_trend_predictor"] = joblib.load(trend_predictor_path)
                models["trend_features"] = joblib.load(MODELS_DIR / "trend_feature_names_v3.pkl")
                print(f"   - Google Trend Predictor: načten ✅")
            
            ensemble_info = {"type": "weighted", "mae": None}
            mae_path = MODELS_DIR / "historical_mae_v3.pkl"
            if mae_path.exists():
                mae_value = joblib.load(mae_path)
                # MAE může být buď float nebo dict
                if isinstance(mae_value, dict):
                    ensemble_info["mae"] = mae_value.get("mae", None)
                else:
                    ensemble_info["mae"] = mae_value
            
            print(f"   - Ensemble type: WEIGHTED V3")
            if ensemble_info.get("mae") is not None:
                print(f"   - Ensemble MAE: {ensemble_info['mae']:.2f}")
        else:
            # Fallback na staré modely
            print("   ⚠️ V3 modely nenalezeny, používám staré modely")
            models["lightgbm"] = joblib.load(MODELS_DIR / "lightgbm_model.pkl")
            models["xgboost"] = joblib.load(MODELS_DIR / "xgboost_model.pkl")
            models["catboost"] = joblib.load(MODELS_DIR / "catboost_model.pkl")
            ensemble_weights = joblib.load(MODELS_DIR / "ensemble_weights.pkl")
            feature_columns = joblib.load(MODELS_DIR / "feature_columns.pkl")

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

        # Načtení historických dat z databáze pro statistiky
        if DATABASE_ENABLED:
            try:
                db_temp = SessionLocal()
                historical_records = db_temp.query(HistoricalData).all()
                
                if len(historical_records) > 0:
                    # Konvertovat na pandas DataFrame pro kompatibilitu - VŠECHNY SLOUPCE!
                    historical_data = pd.DataFrame([{
                        column.name: getattr(record, column.name)
                        for column in HistoricalData.__table__.columns
                        if column.name not in ['id', 'created_at']  # Vynechat jen metadata
                    } for record in historical_records])
                    historical_data['date'] = pd.to_datetime(historical_data['date'])
                    
                    print(
                        f"   - Historická data z DB: {len(historical_data)} záznamů (do {historical_data['date'].max().date()})"
                    )
                else:
                    print(f"   ⚠️ Žádná historická data v DB - načítám z CSV...")
                    db_temp.close()
                    
                    # Automaticky načíst data z CSV
                    try:
                        csv_path = str(BASE_DIR / "data" / "processed" / "techmania_with_weather_and_holidays.csv")
                        if not os.path.exists(csv_path):
                            csv_path = "../data/processed/techmania_with_weather_and_holidays.csv"
                        
                        load_historical_data(csv_path, auto_skip_if_exists=True)
                        print("   ✅ Historická data úspěšně načtena z CSV")
                        
                        # Znovu načíst data z DB
                        db_temp = SessionLocal()
                        historical_records = db_temp.query(HistoricalData).all()
                        
                        if len(historical_records) > 0:
                            # Konvertovat na pandas DataFrame - VŠECHNY SLOUPCE!
                            historical_data = pd.DataFrame([{
                                column.name: getattr(record, column.name)
                                for column in HistoricalData.__table__.columns
                                if column.name not in ['id', 'created_at']  # Vynechat jen metadata
                            } for record in historical_records])
                            historical_data['date'] = pd.to_datetime(historical_data['date'])
                            print(f"   - Historická data z DB: {len(historical_data)} záznamů (do {historical_data['date'].max().date()})")
                        else:
                            historical_data = None
                    except Exception as csv_error:
                        print(f"   ❌ Chyba při automatickém načítání z CSV: {csv_error}")
                        historical_data = None
                
                db_temp.close()
                
                # Kontrola a načtení template dat pro 2026
                try:
                    db_temp = SessionLocal()
                    template_records_count = db_temp.query(TemplateData).count()
                    
                    if template_records_count == 0:
                        print(f"   ⚠️ Žádná template data v DB - načítám z CSV...")
                        db_temp.close()
                        
                        try:
                            template_csv_path = str(BASE_DIR / "data" / "raw" / "techmania_2026_template.csv")
                            if not os.path.exists(template_csv_path):
                                template_csv_path = "../data/raw/techmania_2026_template.csv"
                            
                            load_template_data(template_csv_path, auto_skip_if_exists=True)
                            print("   ✅ Template data úspěšně načtena z CSV")
                            
                            # Zobrazit statistiku
                            db_temp = SessionLocal()
                            template_count = db_temp.query(TemplateData).count()
                            if template_count > 0:
                                print(f"   - Template data v DB: {template_count} záznamů pro rok 2026")
                            db_temp.close()
                        except Exception as template_error:
                            print(f"   ❌ Chyba při automatickém načítání template dat: {template_error}")
                    else:
                        print(f"   - Template data v DB: {template_records_count} záznamů")
                        db_temp.close()
                except Exception as template_check_error:
                    print(f"   ⚠️ Chyba při kontrole template dat: {template_check_error}")
                
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


# make_ensemble_prediction - REMOVED (replaced by model_predictor utilities)


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


@app.get("/today", tags=["Predictions"])
async def get_today_prediction():
    """
    Vrátí predikci nebo historická data pro dnešní den (REFACTORED).
    """
    today = date.today()
    
    # Zkusit najít historická data pro dnešek
    if historical_data is not None:
        today_data = historical_data[historical_data['date'].dt.date == today]
        if len(today_data) > 0:
            row = today_data.iloc[0]
            visitors_value = row['total_visitors']
            if pd.notna(visitors_value):
                return {
                    "date": today.isoformat(),
                    "visitors": int(visitors_value),
                    "is_historical": True,
                    "day_of_week": today.strftime("%A"),
                    "is_weekend": today.weekday() >= 5,
                    "is_holiday": holiday_service.is_holiday(today)[0]
                }
    
    # Jinak použít refaktorovaný predict_single_date
    try:
        from predict import predict_single_date
        
        models_dict = {
            'lgb': models['lightgbm'],
            'xgb': models['xgboost'],
            'cat': models['catboost'],
            'weights': ensemble_weights,
            'feature_cols': feature_columns,
            'google_trend_predictor': models.get('google_trend_predictor'),
            'historical_mae': ensemble_info.get('mae') if ensemble_info else None,
            'ensemble_type': ensemble_info.get('type', 'weighted') if ensemble_info else 'weighted'
        }
        
        result = predict_single_date(today, models_dict, historical_df=historical_data)
        
        return {
            "date": today.isoformat(),
            "visitors": result['ensemble_prediction'],
            "is_historical": False,
            "day_of_week": today.strftime("%A"),
            "is_weekend": today.weekday() >= 5,
            "is_holiday": False,  # TODO: extract from result
            "weather": {
                "temperature_mean": result['weather']['temperature'],
                "precipitation": result['weather']['precipitation'],
                "weather_description": result['weather']['description'],
            }
        }
    except Exception as e:
        print(f"❌ Chyba při predikci pro dnešek: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Chyba při predikci: {str(e)}")


@app.get("/stats", response_model=StatsResponse, tags=["Statistics"])
async def get_statistics():
    """
    Získá statistiky z historických dat.
    """
    if historical_data is None:
        raise HTTPException(status_code=503, detail="Historická data nejsou dostupná")

    try:
        # Odfiltrovat NaN hodnoty - pracujeme pouze s čistými daty
        clean_data = historical_data.dropna(subset=['total_visitors'])
        
        if len(clean_data) == 0:
            raise HTTPException(status_code=503, detail="Žádná platná data nejsou k dispozici")
        
        # Výpočet statistik z čistých dat
        total_visitors = int(clean_data["total_visitors"].sum())
        avg_daily = float(clean_data["total_visitors"].mean())

        # Najít den s nejvyšší návštěvností
        peak_idx = clean_data["total_visitors"].idxmax()
        peak_day = clean_data.loc[peak_idx, "date"].strftime("%d. %B %Y")
        peak_visitors = int(clean_data.loc[peak_idx, "total_visitors"])

        # Vypočítat trend (poslední měsíc vs předchozí měsíc) - pouze z čistých dat
        last_month = clean_data.tail(30)
        prev_month = (
            clean_data.iloc[-60:-30]
            if len(clean_data) >= 60
            else clean_data.head(30)
        )

        trend = 0.0
        if len(prev_month) > 0 and len(last_month) > 0:
            last_avg = last_month["total_visitors"].mean()
            prev_avg = prev_month["total_visitors"].mean()
            if pd.notna(last_avg) and pd.notna(prev_avg) and prev_avg > 0:
                trend = ((last_avg - prev_avg) / prev_avg) * 100

        return sanitize_for_json({
            "total_visitors": total_visitors,
            "avg_daily_visitors": avg_daily,
            "peak_day": peak_day,
            "peak_visitors": peak_visitors,
            "trend": round(trend, 1),
            "data_start_date": clean_data["date"].min().strftime("%Y-%m-%d"),
            "data_end_date": clean_data["date"].max().strftime("%Y-%m-%d"),
        })
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


# Pydantic modely pro kalendář
class CalendarEvent(BaseModel):
    date: str
    name: str
    type: str  # holiday, vacation, high_traffic
    predicted_visitors: Optional[int] = None
    day_of_week: Optional[str] = None


class CalendarEventsResponse(BaseModel):
    events: List[CalendarEvent]
    month: int
    year: int
    total_events: int


@app.get("/calendar/events", response_model=CalendarEventsResponse, tags=["Calendar"])
async def get_calendar_events(month: int = None, year: int = None):
    """
    Získá události (svátky, prázdniny) pro daný měsíc.
    Pokud month/year nejsou zadány, použije aktuální měsíc.
    """
    from datetime import date as date_type
    import calendar as cal_module
    
    # Default na aktuální měsíc
    today = date_type.today()
    if month is None:
        month = today.month
    if year is None:
        year = today.year
    
    # Validace
    if month < 1 or month > 12:
        raise HTTPException(status_code=400, detail="Měsíc musí být mezi 1 a 12")
    if year < 2020 or year > 2030:
        raise HTTPException(status_code=400, detail="Rok musí být mezi 2020 a 2030")
    
    events = []
    day_of_week_names = {
        0: 'Pondělí', 1: 'Úterý', 2: 'Středa', 3: 'Čtvrtek',
        4: 'Pátek', 5: 'Sobota', 6: 'Neděle'
    }
    vacation_names = {
        'winter': 'Vánoční prázdniny',
        'halfyear': 'Pololetní prázdniny',
        'spring': 'Jarní prázdniny',
        'easter': 'Velikonoční prázdniny',
        'summer': 'Letní prázdniny',
        'autumn': 'Podzimní prázdniny',
    }
    
    try:
        _, num_days = cal_module.monthrange(year, month)
        
        # Načíst data z databáze (TemplateData pro 2026)
        if DATABASE_ENABLED:
            db = next(get_db())
            try:
                start_str = f"{year}-{month:02d}-01"
                end_str = f"{year}-{month:02d}-{num_days:02d}"
                
                template_records = db.query(TemplateData).filter(
                    TemplateData.date >= start_str,
                    TemplateData.date <= end_str
                ).all()
                
                for record in template_records:
                    record_date = date_type.fromisoformat(str(record.date))
                    day_of_week = day_of_week_names.get(record_date.weekday(), '')
                    
                    # Svátky
                    if record.is_holiday == 1 and record.nazvy_svatek:
                        events.append(CalendarEvent(
                            date=str(record.date),
                            name=record.nazvy_svatek,
                            type='holiday',
                            day_of_week=day_of_week
                        ))
                    
                    # Prázdniny
                    if record.school_break_type and record.school_break_type.strip():
                        vacation_name = vacation_names.get(record.school_break_type, record.school_break_type)
                        events.append(CalendarEvent(
                            date=str(record.date),
                            name=vacation_name,
                            type='vacation',
                            day_of_week=day_of_week
                        ))
                    
                    # Extra události (pokud nejsou stejné jako svátek)
                    if record.extra and record.extra.strip() and record.extra != record.nazvy_svatek:
                        events.append(CalendarEvent(
                            date=str(record.date),
                            name=record.extra,
                            type='event',
                            day_of_week=day_of_week
                        ))
            finally:
                db.close()
        
        # Fallback - použít holiday_service
        if not events:
            for day in range(1, num_days + 1):
                current_date = date_type(year, month, day)
                is_holiday, holiday_name = holiday_service.is_holiday(current_date)
                
                if is_holiday and holiday_name:
                    events.append(CalendarEvent(
                        date=current_date.isoformat(),
                        name=holiday_name,
                        type='holiday',
                        day_of_week=day_of_week_names.get(current_date.weekday(), '')
                    ))
        
        # Deduplikovat
        seen = set()
        unique_events = []
        for event in events:
            key = (event.date, event.type, event.name)
            if key not in seen:
                seen.add(key)
                unique_events.append(event)
        
        unique_events.sort(key=lambda x: x.date)
        
        return CalendarEventsResponse(
            events=unique_events,
            month=month,
            year=year,
            total_events=len(unique_events)
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Chyba při načítání událostí: {str(e)}")


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(
    request: PredictionRequest,
    db: Session = Depends(get_db) if DATABASE_ENABLED else None,
):
    """
    Predikce návštěvnosti pro konkrétní datum (REFACTORED).

    Použije refaktorovaný predict_single_date.
    Automaticky detekuje svátky a získává informace o počasí.
    Uloží predikci do databáze s verzováním.
    
    DŮLEŽITÉ: Nepřijímá predikce do minulosti (pouze budoucí data).
    MAX 16 DNÍ DOPŘEDU (limit Weather API).
    """
    if not models:
        raise HTTPException(status_code=503, detail="Modely nejsou načteny")

    try:
        from predict import predict_single_date
        
        # Parsování data
        pred_date = pd.to_datetime(request.date).date()
        today = date.today()
        
        # Validace: zakázat predikce do minulosti
        if DATABASE_ENABLED and not validate_future_date(pred_date):
            raise HTTPException(
                status_code=400,
                detail=f"Nelze vytvořit predikci do minulosti. Požadované datum: {pred_date}, Dnešní datum: {today}."
            )
        
        # Validace: max 16 dní dopředu
        days_ahead = (pred_date - today).days
        if days_ahead > 16:
            raise HTTPException(
                status_code=400,
                detail=f"Nelze vytvořit predikci více než 16 dní dopředu (Weather API limit). Maximální datum: {today + timedelta(days=16)}"
            )
        
        # Připravit models_dict
        models_dict = {
            'lgb': models['lightgbm'],
            'xgb': models['xgboost'],
            'cat': models['catboost'],
            'weights': ensemble_weights,
            'feature_cols': feature_columns,
            'google_trend_predictor': models.get('google_trend_predictor'),
            'historical_mae': ensemble_info.get('mae') if ensemble_info else None,
            'ensemble_type': ensemble_info.get('type', 'weighted') if ensemble_info else 'weighted'
        }
        
        # Zavolat refaktorovaný predict_single_date
        result = predict_single_date(pred_date, models_dict, historical_df=historical_data)
        
        prediction = result['ensemble_prediction']
        confidence_interval = result['confidence_interval']
        weather_info = result['weather']

        # Uložit predikci do databáze
        if DATABASE_ENABLED and db is not None:
            try:
                version = get_next_version(db, pred_date)
                day_names = ["pondělí", "úterý", "středa", "čtvrtek", "pátek", "sobota", "neděle"]
                day_of_week_cz = day_names[pred_date.weekday()]

                db_prediction = Prediction(
                    prediction_date=pred_date,
                    predicted_visitors=prediction,
                    temperature_mean=weather_info.get("temperature"),
                    precipitation=weather_info.get("precipitation"),
                    wind_speed_max=weather_info.get("rain", 0),
                    is_rainy=1 if weather_info.get("rain", 0) > 0 else 0,
                    is_snowy=1 if weather_info.get("snowfall", 0) > 0 else 0,
                    is_nice_weather=0,
                    day_of_week=day_of_week_cz,
                    is_weekend=1 if pred_date.weekday() >= 5 else 0,
                    is_holiday=0,
                    model_name="ensemble",
                    confidence_lower=confidence_interval[0],
                    confidence_upper=confidence_interval[1],
                    version=version,
                    created_by="api",
                )
                db.add(db_prediction)
                db.commit()
                print(f"✅ Prediction saved to database: {pred_date} (version {version})")
            except Exception as e:
                print(f"⚠️ Failed to save prediction to database: {e}")
                db.rollback()

        return {
            "date": pred_date.strftime("%Y-%m-%d"),
            "predicted_visitors": prediction,
            "confidence_interval": {
                "lower": confidence_interval[0],
                "upper": confidence_interval[1],
            },
            "model_info": {
                "type": result.get("ensemble_type", "WEIGHTED").upper(),
                "models": list(result['individual_predictions'].keys()),
                "weights": result.get("model_weights", {}),
            },
            "holiday_info": {"is_holiday": False, "holiday_name": None},
            "weather_info": {
                "temperature_mean": float(weather_info["temperature"]),
                "precipitation": float(weather_info["precipitation"]),
                "weather_description": weather_info.get("description", "N/A"),
                "is_nice_weather": False,
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chyba při predikci: {str(e)}")


@app.post(
    "/predict/range", response_model=RangePredictionResponse, tags=["Predictions"]
)
async def predict_range(
    request: RangePredictionRequest, 
    backtest: bool = False,
    db: Session = Depends(get_db) if DATABASE_ENABLED else None
):
    """
    Predikce návštěvnosti pro časové období (REFACTORED).

    Použije refaktorovaný predict_date_range.
    
    Parametry:
    - backtest: Pokud True, povolí predikce pro historická data (pro testování přesnosti modelu)
    
    DŮLEŽITÉ: 
    - Bez backtest=True nepřijímá predikce do minulosti
    - MAX 16 DNÍ DOPŘEDU pro budoucí data (limit Weather API forecast)
    """
    if not models:
        raise HTTPException(status_code=503, detail="Modely nejsou načteny")

    try:
        from predict import predict_date_range

        start_date = pd.to_datetime(request.start_date)
        end_date = pd.to_datetime(request.end_date)
        today = date.today()

        if start_date > end_date:
            raise HTTPException(status_code=400, detail="start_date musí být před end_date")
        
        # Validace pro backtest vs normální predikce
        if backtest:
            if end_date.date() > today:
                raise HTTPException(
                    status_code=400,
                    detail=f"Backtest je pouze pro historická data. End datum: {end_date.date()} je v budoucnosti."
                )
            print(f"🔬 BACKTEST MODE: {start_date.date()} - {end_date.date()}")
        else:
            if DATABASE_ENABLED and start_date.date() < today:
                raise HTTPException(
                    status_code=400,
                    detail=f"Nelze vytvořit predikci do minulosti. Použijte parametr backtest=true"
                )
            
            max_date = today + timedelta(days=16)
            if end_date.date() > max_date:
                raise HTTPException(
                    status_code=400,
                    detail=f"Nelze vytvořit predikci více než 16 dní dopředu. Maximální datum: {max_date}"
                )
        
        # Připravit models_dict
        models_dict = {
            "lgb": models["lightgbm"],
            "xgb": models["xgboost"],
            "cat": models["catboost"],
            "weights": ensemble_weights,
            "feature_cols": feature_columns,
            "google_trend_predictor": models.get('google_trend_predictor'),
            "historical_mae": ensemble_info.get('mae') if ensemble_info else None,
            "ensemble_type": ensemble_info.get("type", "weighted") if ensemble_info else "weighted",
            "meta_model": meta_model,
        }

        # Zavolat refaktorovaný predict_date_range
        results_df = predict_date_range(start_date, end_date, models_dict)
        
        # Odstranit duplikáty - vzít pouze první výskyt každého data
        results_df = results_df.drop_duplicates(subset=['date'], keep='first')

        # Formátování výstupu
        predictions = []
        for _, row in results_df.iterrows():
            pred_date = row["date"].date()
            prediction_value = int(row["prediction"])

            # Pro backtest: načíst data z databáze, jinak ze služeb
            if backtest and DATABASE_ENABLED and db is not None:
                # Načíst historická data z databáze
                hist_record = db.query(HistoricalData).filter(
                    HistoricalData.date == pred_date
                ).first()
                
                if hist_record:
                    # Použít skutečná data z databáze
                    holiday_name = hist_record.nazvy_svatek
                    if holiday_name in [None, '', '0', 0]:
                        holiday_name = None
                    
                    holiday_info_data = {
                        "is_holiday": bool(hist_record.is_holiday),
                        "holiday_name": holiday_name
                    }
                    
                    # Interpretovat weather_code na popis
                    weather_description = "N/A"
                    if hist_record.weather_code is not None:
                        weather_description = weather_service._interpret_weather_code(int(hist_record.weather_code))
                    
                    weather_data = {
                        "temperature_mean": hist_record.temperature_mean or 0.0,
                        "precipitation": hist_record.precipitation or 0.0,
                        "weather_description": weather_description,
                        "is_nice_weather": bool(hist_record.is_nice_weather) if hist_record.is_nice_weather is not None else False,
                        "wind_speed_max": hist_record.wind_speed or 0.0,
                        "is_rainy": bool(hist_record.is_rainy) if hist_record.is_rainy is not None else False,
                        "is_snowy": bool(hist_record.is_snowy) if hist_record.is_snowy is not None else False,
                    }
                else:
                    # Fallback na služby
                    holiday_info_data = holiday_service.get_holiday_info(pred_date)
                    weather_data = weather_service.get_weather(pred_date)
            else:
                # Budoucí predikce - použít služby
                holiday_info_data = holiday_service.get_holiday_info(pred_date)
                weather_data = weather_service.get_weather(pred_date)

            day_name = row["date"].strftime("%A")
            day_name_cs = {
                "Monday": "Pondělí", "Tuesday": "Úterý", "Wednesday": "Středa",
                "Thursday": "Čtvrtek", "Friday": "Pátek", "Saturday": "Sobota", "Sunday": "Neděle"
            }.get(day_name, day_name)

            predictions.append({
                "date": row["date"].strftime("%Y-%m-%d"),
                "predicted_visitors": prediction_value,
                "confidence_interval": {
                    "lower": int(row["lower_bound"]),
                    "upper": int(row["upper_bound"]),
                },
                "holiday_info": {
                    "is_holiday": holiday_info_data["is_holiday"],
                    "holiday_name": holiday_info_data["holiday_name"],
                },
                "weather_info": {
                    "temperature_mean": float(weather_data["temperature_mean"]),
                    "precipitation": float(weather_data["precipitation"]),
                    "weather_description": weather_data.get("weather_description", "N/A"),
                    "is_nice_weather": bool(weather_data.get("is_nice_weather", False)),
                },
                "day_of_week": day_name_cs,
                "is_weekend": row["date"].dayofweek >= 5,
            })
            
            # Uložit do databáze
            if DATABASE_ENABLED and db is not None:
                try:
                    version = get_next_version(db, pred_date)
                    db_prediction = Prediction(
                        prediction_date=pred_date,
                        predicted_visitors=prediction_value,
                        temperature_mean=weather_data.get("temperature_mean"),
                        precipitation=weather_data.get("precipitation"),
                        wind_speed_max=weather_data.get("wind_speed_max"),
                        is_rainy=1 if weather_data.get("is_rainy", False) else 0,
                        is_snowy=1 if weather_data.get("is_snowy", False) else 0,
                        is_nice_weather=1 if weather_data.get("is_nice_weather", False) else 0,
                        day_of_week=day_name_cs,
                        is_weekend=1 if row["date"].dayofweek >= 5 else 0,
                        is_holiday=1 if holiday_info_data["is_holiday"] else 0,
                        model_name="ensemble",
                        confidence_lower=int(row["lower_bound"]),
                        confidence_upper=int(row["upper_bound"]),
                        version=version,
                        created_by="api_range" if not backtest else "api_backtest",
                    )
                    db.add(db_prediction)
                    db.flush()  # FIXED: Flush immediately to ensure next get_next_version() sees this record
                except Exception as e:
                    print(f"⚠️ Failed to save prediction for {pred_date}: {e}")
                    db.rollback()
        
        if DATABASE_ENABLED and db is not None:
            try:
                db.commit()
                print(f"✅ Saved {len(predictions)} predictions to database")
            except Exception as e:
                print(f"⚠️ Failed to commit predictions: {e}")
                db.rollback()

        return {
            "predictions": predictions,
            "total_predicted": int(results_df["prediction"].sum()),
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


@app.get("/predictions/history", tags=["Predictions"])
async def get_predictions_history_range(
    days: int = 30,
    include_metrics: bool = False,
    db: Session = Depends(get_db) if DATABASE_ENABLED else None
):
    """
    Získá historii predikcí pro posledních N dní s možností porovnání se skutečností.
    """
    if not DATABASE_ENABLED or db is None:
        raise HTTPException(status_code=503, detail="Database není dostupná")

    try:
        from sqlalchemy import func
        from datetime import timedelta

        end_date = date.today()
        start_date = end_date - timedelta(days=days)

        # Získat nejnovější verzi predikce pro každé datum v rozsahu
        from sqlalchemy import func, desc
        
        subquery = (
            db.query(
                Prediction.prediction_date,
                func.max(Prediction.version).label("max_version"),
            )
            .filter(Prediction.prediction_date >= start_date)
            .filter(Prediction.prediction_date <= end_date)
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
            .order_by(Prediction.prediction_date.asc(), Prediction.id.desc())
            .all()
        )
        
        # Deduplikovat podle prediction_date (vzít pouze první záznam pro každé datum)
        seen_dates = set()
        unique_predictions = []
        for pred in predictions:
            if pred.prediction_date not in seen_dates:
                seen_dates.add(pred.prediction_date)
                unique_predictions.append(pred)
        
        predictions = unique_predictions

        history = []
        for pred in predictions:
            pred_data = {
                "date": pred.prediction_date.isoformat(),
                "predicted": pred.predicted_visitors,
                "predicted_visitors": pred.predicted_visitors,  # Zpětná kompatibilita
                "is_future": pred.prediction_date > date.today(),
                "confidence_lower": pred.confidence_lower,
                "confidence_upper": pred.confidence_upper,
                "confidence_interval": {
                    "lower": pred.confidence_lower,
                    "upper": pred.confidence_upper,
                }
            }

            # Pokud máme historická data, přidat skutečnou návštěvnost
            if pred.prediction_date <= date.today():
                historical = db.query(HistoricalData).filter(
                    HistoricalData.date == pred.prediction_date
                ).first()
                
                if historical and historical.total_visitors is not None:
                    pred_data["actual_visitors"] = int(historical.total_visitors)
                    error = abs(pred.predicted_visitors - historical.total_visitors)
                    pred_data["error"] = int(error)
                    pred_data["error_percentage"] = round(abs(
                        (pred.predicted_visitors - historical.total_visitors) / historical.total_visitors * 100
                    ), 2) if historical.total_visitors > 0 else 0
                    
                    # Přidat metriky pouze pokud include_metrics
                    if include_metrics:
                        pred_data["accuracy_metrics"] = {
                            "within_10_percent": pred_data["error_percentage"] <= 10,
                            "within_20_percent": pred_data["error_percentage"] <= 20,
                        }

            history.append(pred_data)

        # Vypočítat summary pokud include_metrics
        summary = None
        if include_metrics:
            past_predictions = [h for h in history if not h.get('is_future', True) and 'actual_visitors' in h]
            
            if past_predictions:
                errors = [h['error'] for h in past_predictions]
                error_percentages = [h['error_percentage'] for h in past_predictions]
                
                # Přesnost do 10% a 20%
                within_10_percent = sum(1 for e in error_percentages if e <= 10)
                within_20_percent = sum(1 for e in error_percentages if e <= 20)
                
                summary = {
                    'valid_comparisons': len(past_predictions),
                    'avg_error': sum(errors) / len(errors) if errors else 0,
                    'avg_error_percent': sum(error_percentages) / len(error_percentages) if error_percentages else 0,
                    'accuracy_10_percent': (within_10_percent / len(past_predictions) * 100) if past_predictions else 0,
                    'accuracy_20_percent': (within_20_percent / len(past_predictions) * 100) if past_predictions else 0,
                }

        return {
            "history": history,
            "summary": summary,
            "count": len(history),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Chyba při načítání historie: {str(e)}"
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

        response_data = {
            "correlations": correlations,
            "description": "Korelační koeficienty a multiplikátory vypočtené z historických dat",
        }
        return sanitize_for_json(response_data)
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

        response_data = {
            "by_weekday": weekday_pattern,
            "by_month": monthly_pattern,
            "holiday_vs_regular": holiday_vs_regular,
        }
        return sanitize_for_json(response_data)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Chyba při výpočtu sezónnosti: {str(e)}"
        )


@app.get("/analytics/prediction-history", tags=["Analytics"])
async def get_prediction_history(
    days: int = 30,
    include_future: bool = True,
    db: Session = Depends(get_db) if DATABASE_ENABLED else None
):
    """
    Získá historii predikcí s porovnáním skutečných hodnot.
    Umožňuje sledovat, jak přesné byly predikce oproti realitě.
    
    - days: počet dní do minulosti
    - include_future: zda zahrnout i budoucí predikce (bez porovnání)
    """
    if not DATABASE_ENABLED or db is None:
        raise HTTPException(status_code=503, detail="Databáze není dostupná")
    
    try:
        from datetime import timedelta
        
        today = date.today()
        cutoff_date = today - timedelta(days=days)
        
        # Získat predikce - buď jen historické nebo včetně budoucích
        if include_future:
            predictions = db.query(Prediction)\
                .filter(Prediction.prediction_date >= cutoff_date)\
                .order_by(Prediction.prediction_date.desc(), Prediction.version.desc())\
                .all()
        else:
            predictions = db.query(Prediction)\
                .filter(Prediction.prediction_date >= cutoff_date)\
                .filter(Prediction.prediction_date <= today)\
                .order_by(Prediction.prediction_date.desc(), Prediction.version.desc())\
                .all()
        
        if not predictions:
            return {
                "history": [],
                "summary": {
                    "total_predictions": 0,
                    "avg_error": None,
                    "avg_error_percent": None,
                    "predictions_within_10_percent": 0,
                    "predictions_within_20_percent": 0
                }
            }
        
        # Získat skutečné hodnoty z historických dat
        history = []
        total_error = 0
        total_error_percent = 0
        within_10 = 0
        within_20 = 0
        valid_comparisons = 0
        
        for pred in predictions:
            pred_date = pred.prediction_date
            is_future = pred_date > today
            
            # Najít skutečnou hodnotu (pouze pro minulé/dnešní datum)
            actual_value = None
            if not is_future and historical_data is not None:
                actual_row = historical_data[historical_data['date'].dt.date == pred_date]
                if len(actual_row) > 0 and pd.notna(actual_row.iloc[0]['total_visitors']):
                    actual_value = int(actual_row.iloc[0]['total_visitors'])
            
            # Vypočítat chybu
            error = None
            error_percent = None
            if actual_value is not None:
                error = pred.predicted_visitors - actual_value
                error_percent = (error / actual_value * 100) if actual_value > 0 else 0
                
                total_error += abs(error)
                total_error_percent += abs(error_percent)
                valid_comparisons += 1
                
                if abs(error_percent) <= 10:
                    within_10 += 1
                if abs(error_percent) <= 20:
                    within_20 += 1
            
            history.append({
                "date": pred_date.isoformat(),
                "predicted": pred.predicted_visitors,
                "actual": actual_value,
                "error": error,
                "error_percent": round(error_percent, 1) if error_percent is not None else None,
                "version": pred.version,
                "created_at": pred.created_at.isoformat() if pred.created_at else None,
                "confidence_lower": pred.confidence_lower,
                "confidence_upper": pred.confidence_upper,
                "within_confidence": (
                    actual_value is not None and 
                    pred.confidence_lower is not None and 
                    pred.confidence_upper is not None and
                    pred.confidence_lower <= actual_value <= pred.confidence_upper
                ),
                "is_future": is_future
            })
        
        # Souhrn
        summary = {
            "total_predictions": len(predictions),
            "valid_comparisons": valid_comparisons,
            "avg_error": round(total_error / valid_comparisons, 1) if valid_comparisons > 0 else None,
            "avg_error_percent": round(total_error_percent / valid_comparisons, 1) if valid_comparisons > 0 else None,
            "predictions_within_10_percent": within_10,
            "predictions_within_20_percent": within_20,
            "accuracy_10_percent": round(within_10 / valid_comparisons * 100, 1) if valid_comparisons > 0 else None,
            "accuracy_20_percent": round(within_20 / valid_comparisons * 100, 1) if valid_comparisons > 0 else None
        }
        
        return sanitize_for_json({
            "history": history,
            "summary": summary
        })
        
    except Exception as e:
        print(f"Error in prediction history: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Chyba při získávání historie predikcí: {str(e)}"
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
        
        # Filtrovat pouze řádky s platnými hodnotami visitors (bez NaN)
        clean_data = historical_data.dropna(subset=['total_visitors'])

        if year is not None:
            # Filtrovat data pro daný rok
            year_data = clean_data[clean_data["date"].dt.year == year].copy()

            if len(year_data) == 0:
                return {
                    "year": year,
                    "data": [],
                    "min_visitors": 0,
                    "max_visitors": 0,
                    "available_years": sorted(
                        clean_data["date"].dt.year.unique().tolist()
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
                    clean_data["date"].dt.year.unique().tolist()
                ),
            }
        else:
            # Vrátit data pro všechny roky (bez NaN)
            all_data = []
            for _, row in clean_data.iterrows():
                all_data.append(
                    {
                        "date": row["date"].strftime("%Y-%m-%d"),
                        "visitors": int(row["total_visitors"]),
                    }
                )

            return {
                "data": all_data,
                "min_visitors": int(clean_data["total_visitors"].min()),
                "max_visitors": int(clean_data["total_visitors"].max()),
                "available_years": sorted(
                    clean_data["date"].dt.year.unique().tolist()
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
    uvicorn.run(app, host="0.0.0.0", port=8000)