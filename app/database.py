"""
Database models and initialization for Techmania predictions
Refactored version with three separate tables:
- historical_data: Historical data from techmania_with_weather_and_holidays.csv (training data)
- template_data: Template data for 2026 with complete_flag for real data detection
- predictions: Versioned predictions with multiple versions per date
"""
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Date, Boolean, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, date as date_type
import os

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./techmania.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class HistoricalData(Base):
    """
    Historical visitor data from techmania_with_weather_and_holidays.csv
    Used ONLY for model training. Read-only after initial import.
    """
    __tablename__ = "historical_data"
    
    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, nullable=False, unique=True, index=True)
    
    # Visitor data
    day_of_week = Column(String)  # pátek, sobota, neděle, pondělí, etc.
    school_visitors = Column(Float)
    public_visitors = Column(Float)
    total_visitors = Column(Float, nullable=False)
    extra = Column(String)
    
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
    wind_speed = Column(Float)
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
    
    # Additional features
    google_trend = Column(Float)
    Mateřská_škola = Column(Float)
    Střední_škola = Column(Float)
    Základní_škola = Column(Float)
    is_event = Column(Integer)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)


class TemplateData(Base):
    """
    Template data for 2026 predictions (from techmania_2026_template.csv)
    Contains default values for predictions. When real data is added via Excel,
    the complete_flag is set to True and the record can be used for charts/stats.
    """
    __tablename__ = "template_data"
    
    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, nullable=False, unique=True, index=True)
    
    # Data completeness flag
    is_complete = Column(Boolean, default=False, nullable=False, index=True)
    # Tracks when real data was added
    completed_at = Column(DateTime, nullable=True)
    
    # Visitor data (null until real data is added)
    day_of_week = Column(String)
    school_visitors = Column(Float, nullable=True)
    public_visitors = Column(Float, nullable=True)
    total_visitors = Column(Float, nullable=True)  # Nullable until completed
    extra = Column(String, nullable=True)
    opening_hours = Column(String, nullable=True)
    
    # Calendar features - Basic
    is_weekend = Column(Integer)
    is_holiday = Column(Integer)
    nazvy_svatek = Column(String, nullable=True)
    
    # Calendar features - School breaks
    is_spring_break = Column(Integer)
    is_autumn_break = Column(Integer)
    is_winter_break = Column(Integer)
    is_easter_break = Column(Integer)
    is_halfyear_break = Column(Integer)
    is_summer_holiday = Column(Integer)
    is_any_school_break = Column(Integer)
    school_break_type = Column(String, nullable=True)
    days_to_next_break = Column(Integer, nullable=True)
    days_from_last_break = Column(Integer, nullable=True)
    is_week_before_break = Column(Integer)
    is_week_after_break = Column(Integer)
    
    # Calendar features - Advanced
    season_exact = Column(String)
    week_position = Column(String)
    is_month_end = Column(Integer)
    school_week_number = Column(Integer, nullable=True)
    is_bridge_day = Column(Integer)
    long_weekend_length = Column(Integer, nullable=True)
    
    # Weather - Temperature (from forecast/template)
    temperature_max = Column(Float, nullable=True)
    temperature_min = Column(Float, nullable=True)
    temperature_mean = Column(Float, nullable=True)
    apparent_temp_max = Column(Float, nullable=True)
    apparent_temp_min = Column(Float, nullable=True)
    apparent_temp_mean = Column(Float, nullable=True)
    
    # Weather - Precipitation
    precipitation = Column(Float, nullable=True)
    precipitation_probability = Column(Float, nullable=True)
    rain = Column(Float, nullable=True)
    snowfall = Column(Float, nullable=True)
    precipitation_hours = Column(Float, nullable=True)
    weather_code = Column(Integer, nullable=True)
    
    # Weather - Wind
    wind_speed = Column(Float, nullable=True)
    wind_gusts_max = Column(Float, nullable=True)
    wind_direction = Column(Integer, nullable=True)
    
    # Weather - Sun & Cloud
    sunshine_duration = Column(Float, nullable=True)
    daylight_duration = Column(Float, nullable=True)
    sunshine_ratio = Column(Float, nullable=True)
    cloud_cover_percent = Column(Float, nullable=True)
    
    # Weather - Computed features
    is_rainy = Column(Integer, nullable=True)
    is_snowy = Column(Integer, nullable=True)
    is_windy = Column(Integer, nullable=True)
    is_nice_weather = Column(Integer, nullable=True)
    feels_like_delta = Column(Float, nullable=True)
    weather_forecast_confidence = Column(Float, nullable=True)
    temperature_trend_3d = Column(Float, nullable=True)
    is_weather_improving = Column(Integer, nullable=True)
    
    # Additional features
    google_trend = Column(Float, nullable=True)
    Mateřská_škola = Column(Float, nullable=True)
    Střední_škola = Column(Float, nullable=True)
    Základní_škola = Column(Float, nullable=True)
    is_event = Column(Integer, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Prediction(Base):
    """
    Versioned predictions made by the model.
    Multiple predictions can exist for the same date (different versions).
    Only stores predictions for future dates.
    """
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
    
    # Composite index for efficient querying of latest predictions
    __table_args__ = (
        Index('ix_prediction_date_version', 'prediction_date', 'version'),
        Index('ix_prediction_date_created', 'prediction_date', 'created_at'),
    )


class Event(Base):
    """
    Events in Plzen and surroundings that might impact Techmania visitor numbers.
    Scraped from various sources (GoOut, Plzen.eu, etc.)
    """
    __tablename__ = "events"
    
    id = Column(Integer, primary_key=True, index=True)
    event_date = Column(Date, nullable=False, index=True)
    
    # Event details
    title = Column(String, nullable=False)
    description = Column(String, nullable=True)
    venue = Column(String, nullable=True)
    category = Column(String, nullable=True)  # koncert, sport, festival, veletrh, atd.
    expected_attendance = Column(String, nullable=True)  # male/stredni/velke/masivni
    
    # Source info
    source = Column(String, nullable=False)  # goout, plzen.eu, custom
    source_url = Column(String, nullable=True)
    
    # Impact assessment
    impact_level = Column(Integer, default=1)  # 1-5: vliv na navstevnost Techmanie
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Index for efficient date queries
    __table_args__ = (
        Index('ix_event_date_active', 'event_date', 'is_active'),
    )


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
    print("✓ Database initialized successfully")


def get_next_version(db, prediction_date: date_type) -> int:
    """Get the next version number for a prediction date"""
    from sqlalchemy import func
    max_version = db.query(func.max(Prediction.version))\
        .filter(Prediction.prediction_date == prediction_date)\
        .scalar()
    return (max_version or 0) + 1


def validate_future_date(prediction_date: date_type) -> bool:
    """
    Validate that the prediction date is today or in the future.
    Prevents predictions for past dates.
    """
    today = date_type.today()
    return prediction_date >= today


def get_latest_prediction(db, prediction_date: date_type):
    """
    Get the latest (highest version) prediction for a given date.
    """
    return db.query(Prediction)\
        .filter(Prediction.prediction_date == prediction_date)\
        .order_by(Prediction.version.desc())\
        .first()


def get_all_predictions_for_date(db, prediction_date: date_type):
    """
    Get all prediction versions for a given date, ordered by version desc.
    """
    return db.query(Prediction)\
        .filter(Prediction.prediction_date == prediction_date)\
        .order_by(Prediction.version.desc())\
        .all()


def mark_template_complete(db, record_date: date_type, visitor_data: dict) -> bool:
    """
    Mark a template record as complete when real data is added.
    Updates the is_complete flag and visitor data.
    
    Args:
        db: Database session
        record_date: Date of the record
        visitor_data: Dict with real visitor data (total_visitors, school_visitors, etc.)
    
    Returns:
        bool: True if successfully marked complete, False otherwise
    """
    try:
        template_record = db.query(TemplateData)\
            .filter(TemplateData.date == record_date)\
            .first()
        
        if not template_record:
            return False
        
        # Update with real data
        template_record.is_complete = True
        template_record.completed_at = datetime.utcnow()
        
        if 'total_visitors' in visitor_data:
            template_record.total_visitors = visitor_data['total_visitors']
        if 'school_visitors' in visitor_data:
            template_record.school_visitors = visitor_data['school_visitors']
        if 'public_visitors' in visitor_data:
            template_record.public_visitors = visitor_data['public_visitors']
        if 'extra' in visitor_data:
            template_record.extra = visitor_data['extra']
        if 'opening_hours' in visitor_data:
            template_record.opening_hours = visitor_data['opening_hours']
        
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        print(f"Error marking template complete: {e}")
        return False


def get_complete_template_records(db, start_date: date_type = None, end_date: date_type = None):
    """
    Get all template records that have been marked as complete (real data added).
    These can be used for charts and statistics.
    
    Args:
        db: Database session
        start_date: Optional start date filter
        end_date: Optional end date filter
    
    Returns:
        Query result with complete template records
    """
    query = db.query(TemplateData).filter(TemplateData.is_complete == True)
    
    if start_date:
        query = query.filter(TemplateData.date >= start_date)
    if end_date:
        query = query.filter(TemplateData.date <= end_date)
    
    return query.order_by(TemplateData.date).all()


def get_events_for_date(db, event_date: date_type):
    """
    Get all active events for a specific date.
    
    Args:
        db: Database session
        event_date: Date to query events for
    
    Returns:
        List of Event objects
    """
    return db.query(Event)\
        .filter(Event.event_date == event_date)\
        .filter(Event.is_active == True)\
        .all()


def get_events_for_range(db, start_date: date_type, end_date: date_type):
    """
    Get all active events for a date range.
    
    Args:
        db: Database session
        start_date: Start date
        end_date: End date
    
    Returns:
        List of Event objects
    """
    return db.query(Event)\
        .filter(Event.event_date >= start_date)\
        .filter(Event.event_date <= end_date)\
        .filter(Event.is_active == True)\
        .order_by(Event.event_date)\
        .all()


def update_template_event_flag(db, event_date: date_type) -> bool:
    """
    Update template_data is_event flag based on events table.
    Sets is_event=1 if there are any events for this date.
    
    Args:
        db: Database session
        event_date: Date to update
    
    Returns:
        bool: True if successfully updated
    """
    try:
        # Check if there are any events for this date
        events_count = db.query(Event)\
            .filter(Event.event_date == event_date)\
            .filter(Event.is_active == True)\
            .count()
        
        # Update template_data
        template_record = db.query(TemplateData)\
            .filter(TemplateData.date == event_date)\
            .first()
        
        if template_record:
            template_record.is_event = 1 if events_count > 0 else 0
            db.commit()
            return True
        
        return False
    except Exception as e:
        db.rollback()
        print(f"Error updating template event flag: {e}")
        return False