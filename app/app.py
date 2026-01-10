"""
FastAPI backend pro predikci návštěvnosti Techmanie.
"""

from fastapi import FastAPI, HTTPException
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

# Načíst proměnné prostředí
load_dotenv()

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

# Načtení modelů při startu
@app.on_event("startup")
async def load_models():
    """Načte všechny natrénované modely a historická data."""
    global models, feature_columns, ensemble_weights, historical_data
    
    # Cesty v Docker kontejneru
    models_dir = Path('/app/models')
    data_dir = Path('/app/data/raw')
    
    try:
        # Načtení jednotlivých modelů
        models['lightgbm'] = joblib.load(models_dir / 'lightgbm_model.pkl')
        models['xgboost'] = joblib.load(models_dir / 'xgboost_model.pkl')
        models['catboost'] = joblib.load(models_dir / 'catboost_model.pkl')
        
        # Načtení vah ensemble
        ensemble_weights = joblib.load(models_dir / 'ensemble_weights.pkl')
        
        # Načtení seznamu features
        feature_columns = joblib.load(models_dir / 'feature_columns.pkl')
        
        # Načtení historických dat pro statistiky
        try:
            historical_data = pd.read_csv(data_dir / 'techmania_cleaned_master.csv')
            historical_data['date'] = pd.to_datetime(historical_data['date'])
            print(f"   - Historická data: {len(historical_data)} záznamů")
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
    """Provede ensemble predikci."""
    predictions = {}
    
    # Predikce z každého modelu
    for model_name, model in models.items():
        predictions[model_name] = model.predict(df[feature_columns])
    
    # Vážený průměr
    ensemble_pred = np.zeros(len(df))
    for model_name, weight in ensemble_weights.items():
        ensemble_pred += weight * predictions[model_name]
    
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
    
    return {
        "models": list(models.keys()),
        "ensemble_weights": ensemble_weights,
        "features_count": len(feature_columns) if feature_columns else 0,
        "feature_sample": feature_columns[:10] if feature_columns else []
    }

@app.get("/stats", response_model=StatsResponse, tags=["Statistics"])
async def get_statistics():
    """
    Získá statistiky z historických dat.
    """
    if historical_data is None:
        raise HTTPException(status_code=503, detail="Historická data nejsou dostupná")
    
    try:
        # Výpočet statistik
        total_visitors = int(historical_data['total_visitors'].sum())
        avg_daily = float(historical_data['total_visitors'].mean())
        
        # Najít den s nejvyšší návštěvností
        peak_idx = historical_data['total_visitors'].idxmax()
        peak_day = historical_data.loc[peak_idx, 'date'].strftime('%d. %B %Y')
        peak_visitors = int(historical_data.loc[peak_idx, 'total_visitors'])
        
        # Vypočítat trend (poslední měsíc vs předchozí měsíc)
        last_month = historical_data.tail(30)
        prev_month = historical_data.iloc[-60:-30] if len(historical_data) >= 60 else historical_data.head(30)
        
        if len(prev_month) > 0:
            trend = ((last_month['total_visitors'].mean() - prev_month['total_visitors'].mean()) / 
                    prev_month['total_visitors'].mean() * 100)
        else:
            trend = 0.0
        
        return {
            "total_visitors": total_visitors,
            "avg_daily_visitors": avg_daily,
            "peak_day": peak_day,
            "peak_visitors": peak_visitors,
            "trend": round(trend, 1),
            "data_start_date": historical_data['date'].min().strftime('%Y-%m-%d'),
            "data_end_date": historical_data['date'].max().strftime('%Y-%m-%d')
        }
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
        # Získat poslední N dní
        recent_data = historical_data.tail(days).copy()
        
        data_points = []
        for _, row in recent_data.iterrows():
            data_points.append({
                "date": row['date'].strftime('%Y-%m-%d'),
                "visitors": int(row['total_visitors'])
            })
        
        return {
            "data": data_points,
            "start_date": recent_data['date'].min().strftime('%Y-%m-%d'),
            "end_date": recent_data['date'].max().strftime('%Y-%m-%d'),
            "total_days": len(data_points)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chyba při načítání dat: {str(e)}")

@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(request: PredictionRequest):
    """
    Predikce návštěvnosti pro konkrétní datum.
    
    Použije ensemble model (LightGBM + XGBoost + CatBoost) pro predikci.
    Automaticky detekuje svátky a získává informace o počasí.
    """
    if not models:
        raise HTTPException(status_code=503, detail="Modely nejsou načteny")
    
    try:
        # Parsování data
        pred_date = pd.to_datetime(request.date).date()
        
        # Auto-detekce svátku (pokud není zadán)
        if request.is_holiday is None:
            holiday_info = holiday_service.get_holiday_info(pred_date)
            is_holiday = holiday_info['is_holiday']
            holiday_name = holiday_info['holiday_name']
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
        df_pred = pd.DataFrame({
            'date': [pd.to_datetime(pred_date)],
            'total_visitors': [0],  # Potřebné pro create_features (bude ignorováno při predikci)
            'is_holiday': [is_holiday],
            'opening_hours': [request.opening_hours],
            'temperature_max': [weather_data['temperature_max']],
            'temperature_min': [weather_data['temperature_min']],
            'temperature_mean': [weather_data['temperature_mean']],
            'precipitation': [weather_data['precipitation']],
            'is_rainy': [weather_data.get('is_rainy', weather_data['precipitation'] > 1.0)],
            'is_snowy': [weather_data.get('is_snowy', False)],
            'is_nice_weather': [weather_data.get('is_nice_weather', False)],
        })
        
        # Vytvoření features
        df_pred = create_features(df_pred)
        
        # Odstranit lag a rolling features (budou NaN pro single prediction)
        # Nahradit NaN hodnotami střední hodnotou nebo 0
        df_pred = df_pred.fillna(0)
        
        # Vybrat pouze features, které model očekává
        df_pred = df_pred[feature_columns]
        
        # Ensemble predikce
        prediction = make_ensemble_prediction(df_pred)[0]
        
        # Zaokrouhlení na celé číslo
        prediction = int(np.round(prediction))
        
        return {
            "date": pred_date.strftime('%Y-%m-%d'),
            "predicted_visitors": prediction,
            "confidence_interval": {
                "lower": int(prediction * 0.85),
                "upper": int(prediction * 1.15)
            },
            "model_info": {
                "type": "ensemble",
                "models": list(models.keys()),
                "weights": ensemble_weights
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
    """
    if not models:
        raise HTTPException(status_code=503, detail="Modely nejsou načteny")
    
    try:
        from predict import predict_date_range
        
        start_date = pd.to_datetime(request.start_date)
        end_date = pd.to_datetime(request.end_date)
        
        if start_date > end_date:
            raise HTTPException(status_code=400, detail="start_date musí být před end_date")
        
        # Použít funkci z predict.py která automaticky stahuje weather data
        models_dict = {
            'lgb': models['lightgbm'],
            'xgb': models['xgboost'],
            'cat': models['catboost'],
            'weights': ensemble_weights,
            'feature_cols': feature_columns
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

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
