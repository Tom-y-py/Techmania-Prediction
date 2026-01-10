"""
Database models and initialization for Techmania predictions
"""
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Date, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./techmania.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class HistoricalData(Base):
    """Historical visitor data from CSV"""
    __tablename__ = "historical_data"
    
    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, nullable=False, unique=True, index=True)
    
    # Visitor data
    day_of_week = Column(String)  # pátek, sobota, neděle, pondělí, etc.
    school_visitors = Column(Float)
    public_visitors = Column(Float)
    total_visitors = Column(Float, nullable=False)
    extra = Column(String)
    opening_hours = Column(String)
    
    # Calendar features - Basic
    is_weekend = Column(Integer)
    is_holiday = Column(Integer)
    nazvy_svatek = Column(String)
    
    # Calendar features - School breaks
    is_spring_break = Column(Integer)
    is_autumn_break = Column(Integer)
    is_winter_break = Column(Integer)
    is_easter_break = Column(Integer)
    is_halfyear_break = Column(Integer)
    is_summer_holiday = Column(Integer)
    is_any_school_break = Column(Integer)
    school_break_type = Column(String)
    days_to_next_break = Column(Integer)
    days_from_last_break = Column(Integer)
    is_week_before_break = Column(Integer)
    is_week_after_break = Column(Integer)
    
    # Calendar features - Advanced
    season_exact = Column(String)
    week_position = Column(String)
    is_month_end = Column(Integer)
    school_week_number = Column(Integer)
    is_bridge_day = Column(Integer)
    long_weekend_length = Column(Integer)
    
    # Weather - Temperature
    temperature_max = Column(Float)
    temperature_min = Column(Float)
    temperature_mean = Column(Float)
    apparent_temp_max = Column(Float)
    apparent_temp_min = Column(Float)
    apparent_temp_mean = Column(Float)
    
    # Weather - Precipitation
    precipitation = Column(Float)
    precipitation_probability = Column(Float)
    rain = Column(Float)
    snowfall = Column(Float)
    precipitation_hours = Column(Float)
    weather_code = Column(Integer)
    
    # Weather - Wind
    wind_speed_max = Column(Float)
    wind_gusts_max = Column(Float)
    wind_direction = Column(Integer)
    
    # Weather - Sun & Cloud
    sunshine_duration = Column(Float)
    daylight_duration = Column(Float)
    sunshine_ratio = Column(Float)
    cloud_cover_percent = Column(Float)
    
    # Weather - Computed features
    is_rainy = Column(Integer)
    is_snowy = Column(Integer)
    is_windy = Column(Integer)
    is_nice_weather = Column(Integer)
    feels_like_delta = Column(Float)
    weather_forecast_confidence = Column(Float)
    temperature_trend_3d = Column(Float)
    is_weather_improving = Column(Integer)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)


class Prediction(Base):
    """Versioned predictions made by the model"""
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    prediction_date = Column(Date, nullable=False, index=True)
    predicted_visitors = Column(Integer, nullable=False)
    
    # Input weather features used for prediction (from forecast)
    temperature_mean = Column(Float)
    precipitation = Column(Float)
    wind_speed_max = Column(Float)
    is_rainy = Column(Integer)
    is_snowy = Column(Integer)
    is_nice_weather = Column(Integer)
    
    # Calendar features
    day_of_week = Column(String)
    is_weekend = Column(Integer)
    is_holiday = Column(Integer)
    
    # Model info
    model_name = Column(String, default="ensemble")
    confidence_lower = Column(Float, nullable=True)
    confidence_upper = Column(Float, nullable=True)
    
    # Versioning
    version = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    created_by = Column(String, default="api")
    
    # Metadata
    notes = Column(String, nullable=True)


def get_db():
    """Dependency for getting database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)
    print("Database initialized successfully")


def get_next_version(db, prediction_date: Date) -> int:
    """Get the next version number for a prediction date"""
    from sqlalchemy import func
    max_version = db.query(func.max(Prediction.version))\
        .filter(Prediction.prediction_date == prediction_date)\
        .scalar()
    return (max_version or 0) + 1
