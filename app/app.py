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
from pathlib import Path

# Přidat src do path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from feature_engineering import create_all_features
from services import holiday_service, weather_service

# Inicializace FastAPI
app = FastAPI(
    title="Techmania Prediction API",
    description="API pro predikci návštěvnosti Techmanie pomocí ensemble modelu",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # V produkci nastavit konkrétní domény
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globální proměnné pro modely
models = {}
feature_columns = None
ensemble_weights = None

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

class RangePredictionResponse(BaseModel):
    predictions: List[DayPrediction]
    total_predicted: int
    average_daily: float
    period_days: int

class HealthResponse(BaseModel):
    status: str
    models_loaded: Dict[str, bool]
    features_count: Optional[int]

# Načtení modelů při startu
@app.on_event("startup")
async def load_models():
    """Načte všechny natrénované modely."""
    global models, feature_columns, ensemble_weights
    
    models_dir = Path(__file__).parent.parent / 'models'
    
    try:
        # Načtení jednotlivých modelů
        models['lightgbm'] = joblib.load(models_dir / 'lightgbm_model.pkl')
        models['xgboost'] = joblib.load(models_dir / 'xgboost_model.pkl')
        models['catboost'] = joblib.load(models_dir / 'catboost_model.pkl')
        
        # Načtení vah ensemble
        ensemble_weights = joblib.load(models_dir / 'ensemble_weights.pkl')
        
        # Načtení seznamu features
        feature_columns = joblib.load(models_dir / 'feature_columns.pkl')
        
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
        
        # Vytvoření DataFrame pro predikci
        df_pred = pd.DataFrame({
            'date': [pd.to_datetime(pred_date)],
            'is_holiday': [is_holiday],
            'opening_hours': [request.opening_hours],
            # Přidat počasí do features
            'temperature_max': [weather_data.get('temperature_max', 15.0)],
            'temperature_min': [weather_data.get('temperature_min', 5.0)],
            'temperature_mean': [weather_data.get('temperature_mean', 10.0)],
            'precipitation': [weather_data.get('precipitation', 2.0)],
            'is_rainy': [weather_data.get('is_rainy', False)],
            'is_snowy': [weather_data.get('is_snowy', False)],
            'is_nice_weather': [weather_data.get('is_nice_weather', False)],
        })
        
        # Vytvoření features
        df_pred = create_all_features(df_pred)
        
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
                "temperature_mean": float(weather_data.get('temperature_mean', 10.0)),
                "precipitation": float(weather_data.get('precipitation', 2.0)),
                "weather_description": weather_data.get('weather_description', 'Neznámé'),
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
    """
    if not models:
        raise HTTPException(status_code=503, detail="Modely nejsou načteny")
    
    try:
        start_date = pd.to_datetime(request.start_date)
        end_date = pd.to_datetime(request.end_date)
        
        if start_date > end_date:
            raise HTTPException(status_code=400, detail="start_date musí být před end_date")
        
        # Vytvoření date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Vytvoření DataFrame pro všechna data
        df_pred = pd.DataFrame({
            'date': date_range,
            'is_holiday': False,  # Můžeme později rozšířit o automatickou detekci
            'opening_hours': '9-17'
        })
        
        # Vytvoření features
        df_pred = create_all_features(df_pred)
        
        # Ensemble predikce
        predictions_array = make_ensemble_prediction(df_pred)
        
        # Formátování výstupu
        predictions = []
        for i, pred_date in enumerate(date_range):
            predictions.append({
                "date": pred_date.strftime('%Y-%m-%d'),
                "predicted_visitors": int(np.round(predictions_array[i]))
            })
        
        total = int(np.sum(predictions_array))
        
        return {
            "predictions": predictions,
            "total_predicted": total,
            "average_daily": float(np.mean(predictions_array)),
            "period_days": len(date_range)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chyba při predikci: {str(e)}")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
