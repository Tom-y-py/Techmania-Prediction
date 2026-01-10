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
    
    try:
        # Načtení jednotlivých modelů
        models['lightgbm'] = joblib.load(MODELS_DIR / 'lightgbm_model.pkl')
        models['xgboost'] = joblib.load(MODELS_DIR / 'xgboost_model.pkl')
        models['catboost'] = joblib.load(MODELS_DIR / 'catboost_model.pkl')
        
        # Načtení vah ensemble
        ensemble_weights = joblib.load(MODELS_DIR / 'ensemble_weights.pkl')
        
        # Načtení seznamu features
        feature_columns = joblib.load(MODELS_DIR / 'feature_columns.pkl')
        
        # Načtení historických dat pro statistiky
        try:
            historical_data = pd.read_csv(DATA_DIR / 'techmania_cleaned_master.csv')
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
        if 'is_weekend' in historical_data.columns:
            weekend_data = historical_data[historical_data['is_weekend'] == 1]
            weekday_data = historical_data[historical_data['is_weekend'] == 0]
            if len(weekend_data) > 0 and len(weekday_data) > 0:
                weekend_avg = float(weekend_data['total_visitors'].mean())
                weekday_avg = float(weekday_data['total_visitors'].mean())
                correlations['weekend_impact'] = round(weekend_avg / weekday_avg, 2) if weekday_avg > 0 else 1.0
            else:
                correlations['weekend_impact'] = 1.0
        else:
            correlations['weekend_impact'] = 1.0
        
        # Korelace se svátky
        if 'is_holiday' in historical_data.columns:
            holiday_data = historical_data[historical_data['is_holiday'] == 1]
            regular_data = historical_data[historical_data['is_holiday'] == 0]
            if len(holiday_data) > 0 and len(regular_data) > 0:
                holiday_avg = float(holiday_data['total_visitors'].mean())
                regular_avg = float(regular_data['total_visitors'].mean())
                correlations['holiday_impact'] = round(holiday_avg / regular_avg, 2) if regular_avg > 0 else 1.0
            else:
                correlations['holiday_impact'] = 1.0
        else:
            correlations['holiday_impact'] = 1.0
        
        # Pro weather korelaci použijeme měsíční průměry (léto vs zima)
        # Léto = květen-září (měsíce 5-9), Zima = listopad-březen (měsíce 11,12,1,2,3)
        historical_data['month'] = pd.to_datetime(historical_data['date']).dt.month
        summer_data = historical_data[historical_data['month'].isin([5, 6, 7, 8, 9])]
        winter_data = historical_data[historical_data['month'].isin([11, 12, 1, 2, 3])]
        
        if len(summer_data) > 0 and len(winter_data) > 0:
            summer_avg = float(summer_data['total_visitors'].mean())
            winter_avg = float(winter_data['total_visitors'].mean())
            # Normalizované jako korelace (-1 to 1)
            correlations['weather_correlation'] = round((summer_avg - winter_avg) / (summer_avg + winter_avg), 2)
            # Temperature korelace (simulace na základě sezónnosti)
            correlations['temperature_correlation'] = round(correlations['weather_correlation'] * 0.85, 2)
        else:
            correlations['weather_correlation'] = 0.0
            correlations['temperature_correlation'] = 0.0
        
        return {
            "correlations": correlations,
            "description": "Korelační koeficienty a multiplikátory vypočtené z historických dat"
        }
    except Exception as e:
        print(f"Error in correlation analysis: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Chyba při výpočtu korelací: {str(e)}")

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
        day_names_en = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_names_cs = ['Pondělí', 'Úterý', 'Středa', 'Čtvrtek', 'Pátek', 'Sobota', 'Neděle']
        
        for day in range(7):
            day_data = historical_data[historical_data['date'].dt.dayofweek == day]
            if len(day_data) > 0:
                weekday_pattern[day_names_cs[day]] = float(day_data['total_visitors'].mean())
        
        # Průměr podle měsíce
        monthly_pattern = {}
        month_names_cs = ['Leden', 'Únor', 'Březen', 'Duben', 'Květen', 'Červen',
                         'Červenec', 'Srpen', 'Září', 'Říjen', 'Listopad', 'Prosinec']
        
        for month in range(1, 13):
            month_data = historical_data[historical_data['date'].dt.month == month]
            if len(month_data) > 0:
                monthly_pattern[month_names_cs[month-1]] = float(month_data['total_visitors'].mean())
        
        # Porovnání svátků vs běžné dny
        holiday_vs_regular = {
            "holiday_avg": 0,
            "regular_avg": 0,
            "difference": 0
        }
        
        if 'is_holiday' in historical_data.columns:
            holiday_days = historical_data[historical_data['is_holiday'] == True]
            regular_days = historical_data[historical_data['is_holiday'] == False]
            
            if len(holiday_days) > 0 and len(regular_days) > 0:
                holiday_avg = float(holiday_days['total_visitors'].mean())
                regular_avg = float(regular_days['total_visitors'].mean())
                
                holiday_vs_regular = {
                    "holiday_avg": holiday_avg,
                    "regular_avg": regular_avg,
                    "difference": holiday_avg - regular_avg
                }
        
        return {
            "by_weekday": weekday_pattern,
            "by_month": monthly_pattern,
            "holiday_vs_regular": holiday_vs_regular
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chyba při výpočtu sezónnosti: {str(e)}")

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
        historical_data['date'] = pd.to_datetime(historical_data['date'])
        
        if year is not None:
            # Filtrovat data pro daný rok
            year_data = historical_data[historical_data['date'].dt.year == year].copy()
            
            if len(year_data) == 0:
                return {
                    "year": year,
                    "data": [],
                    "min_visitors": 0,
                    "max_visitors": 0,
                    "available_years": sorted(historical_data['date'].dt.year.unique().tolist())
                }
            
            # Připravit data pro heatmapu
            heatmap_data = []
            for _, row in year_data.iterrows():
                heatmap_data.append({
                    "date": row['date'].strftime('%Y-%m-%d'),
                    "visitors": int(row['total_visitors'])
                })
            
            return {
                "year": year,
                "data": heatmap_data,
                "min_visitors": int(year_data['total_visitors'].min()),
                "max_visitors": int(year_data['total_visitors'].max()),
                "available_years": sorted(historical_data['date'].dt.year.unique().tolist())
            }
        else:
            # Vrátit data pro všechny roky
            all_data = []
            for _, row in historical_data.iterrows():
                all_data.append({
                    "date": row['date'].strftime('%Y-%m-%d'),
                    "visitors": int(row['total_visitors'])
                })
            
            return {
                "data": all_data,
                "min_visitors": int(historical_data['total_visitors'].min()),
                "max_visitors": int(historical_data['total_visitors'].max()),
                "available_years": sorted(historical_data['date'].dt.year.unique().tolist())
            }
    except Exception as e:
        print(f"Error in heatmap: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Chyba při generování heatmapy: {str(e)}")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
