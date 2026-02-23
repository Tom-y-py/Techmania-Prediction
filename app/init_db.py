"""
Script to initialize database and load data from CSV files.
Loads historical data and template data into separate tables.
"""
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from database import init_db, SessionLocal, HistoricalData, TemplateData


def load_historical_data(csv_path: str = "../data/processed/techmania_with_weather_and_holidays.csv", auto_skip_if_exists: bool = False):
    """
    Load historical data from CSV into historical_data table.
    This data is used ONLY for model training.
    
    Args:
        csv_path: Path to CSV file
        auto_skip_if_exists: If True, skip loading if data already exists (non-interactive mode)
    """
    
    # Read CSV
    print(f"📖 Reading historical data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Convert date column
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"CSV contains {len(df.columns)} columns and {len(df)} rows")
    
    # Create database session
    db = SessionLocal()
    
    try:
        # Check if data already exists
        existing_count = db.query(HistoricalData).count()
        if existing_count > 0:
            print(f"⚠️  Database already contains {existing_count} historical records.")
            if auto_skip_if_exists:
                print("⏭️  Skipping historical data load (auto mode).")
                return
            response = input("Do you want to clear and reload? (yes/no): ")
            if response.lower() == 'yes':
                db.query(HistoricalData).delete()
                db.commit()
                print("✓ Existing historical data cleared.")
            else:
                print("⏭️  Skipping historical data load.")
                return
        
        # Insert data
        print(f"💾 Loading {len(df)} historical records...")
        records_added = 0
        
        for _, row in df.iterrows():
            try:
                record = HistoricalData(
                    date=row['date'].date(),
                    # Visitor data
                    day_of_week=str(row['day_of_week']) if pd.notna(row.get('day_of_week')) else None,
                    school_visitors=float(row['school_visitors']) if pd.notna(row.get('school_visitors')) else None,
                    public_visitors=float(row['public_visitors']) if pd.notna(row.get('public_visitors')) else None,
                    total_visitors=float(row['total_visitors']),
                    extra=str(row['extra']) if pd.notna(row.get('extra')) else None,
                    
                    # Calendar features - Basic
                    is_weekend=int(row['is_weekend']) if pd.notna(row.get('is_weekend')) else None,
                    is_holiday=int(row['is_holiday']) if pd.notna(row.get('is_holiday')) else None,
                    nazvy_svatek=str(row['nazvy_svatek']) if pd.notna(row.get('nazvy_svatek')) else None,
                    
                    # Calendar features - School breaks
                    is_spring_break=int(row['is_spring_break']) if pd.notna(row.get('is_spring_break')) else None,
                    is_autumn_break=int(row['is_autumn_break']) if pd.notna(row.get('is_autumn_break')) else None,
                    is_winter_break=int(row['is_winter_break']) if pd.notna(row.get('is_winter_break')) else None,
                    is_easter_break=int(row['is_easter_break']) if pd.notna(row.get('is_easter_break')) else None,
                    is_halfyear_break=int(row['is_halfyear_break']) if pd.notna(row.get('is_halfyear_break')) else None,
                    is_summer_holiday=int(row['is_summer_holiday']) if pd.notna(row.get('is_summer_holiday')) else None,
                    is_any_school_break=int(row['is_any_school_break']) if pd.notna(row.get('is_any_school_break')) else None,
                    school_break_type=str(row['school_break_type']) if pd.notna(row.get('school_break_type')) else None,
                    days_to_next_break=int(row['days_to_next_break']) if pd.notna(row.get('days_to_next_break')) else None,
                    days_from_last_break=int(row['days_from_last_break']) if pd.notna(row.get('days_from_last_break')) else None,
                    is_week_before_break=int(row['is_week_before_break']) if pd.notna(row.get('is_week_before_break')) else None,
                    is_week_after_break=int(row['is_week_after_break']) if pd.notna(row.get('is_week_after_break')) else None,
                    
                    # Calendar features - Advanced
                    season_exact=str(row['season_exact']) if pd.notna(row.get('season_exact')) else None,
                    week_position=str(row['week_position']) if pd.notna(row.get('week_position')) else None,
                    is_month_end=int(row['is_month_end']) if pd.notna(row.get('is_month_end')) else None,
                    school_week_number=int(row['school_week_number']) if pd.notna(row.get('school_week_number')) else None,
                    is_bridge_day=int(row['is_bridge_day']) if pd.notna(row.get('is_bridge_day')) else None,
                    long_weekend_length=int(row['long_weekend_length']) if pd.notna(row.get('long_weekend_length')) else None,
                    
                    # Weather - Temperature
                    temperature_max=float(row['temperature_max']) if pd.notna(row.get('temperature_max')) else None,
                    temperature_min=float(row['temperature_min']) if pd.notna(row.get('temperature_min')) else None,
                    temperature_mean=float(row['temperature_mean']) if pd.notna(row.get('temperature_mean')) else None,
                    
                    # Weather - Precipitation
                    precipitation=float(row['precipitation']) if pd.notna(row.get('precipitation')) else None,
                    precipitation_probability=float(row['precipitation_probability']) if pd.notna(row.get('precipitation_probability')) else None,
                    rain=float(row['rain']) if pd.notna(row.get('rain')) else None,
                    snowfall=float(row['snowfall']) if pd.notna(row.get('snowfall')) else None,
                    precipitation_hours=float(row['precipitation_hours']) if pd.notna(row.get('precipitation_hours')) else None,
                    weather_code=int(row['weather_code']) if pd.notna(row.get('weather_code')) else None,
                    
                    # Weather - Wind
                    wind_speed=float(row['wind_speed']) if pd.notna(row.get('wind_speed')) else None,
                    wind_gusts_max=float(row['wind_gusts_max']) if pd.notna(row.get('wind_gusts_max')) else None,
                    
                    # Weather - Computed features
                    is_rainy=int(row['is_rainy']) if pd.notna(row.get('is_rainy')) else None,
                    is_snowy=int(row['is_snowy']) if pd.notna(row.get('is_snowy')) else None,
                    is_windy=int(row['is_windy']) if pd.notna(row.get('is_windy')) else None,
                    is_nice_weather=int(row['is_nice_weather']) if pd.notna(row.get('is_nice_weather')) else None,
                )
                db.add(record)
                records_added += 1
                
                # Commit in batches
                if records_added % 100 == 0:
                    db.commit()
                    print(f"  Loaded {records_added} records...")
                    
            except Exception as e:
                print(f"Error loading row {_}: {e}")
                continue
        
        # Final commit
        db.commit()
        print(f"\n✓ Successfully loaded {records_added} historical records!")
        
        # Show statistics
        total_records = db.query(HistoricalData).count()
        date_range = db.query(
            HistoricalData.date
        ).order_by(HistoricalData.date).first(), db.query(
            HistoricalData.date
        ).order_by(HistoricalData.date.desc()).first()
        
        print(f"\nHistorical Data Statistics:")
        print(f"  Total records: {total_records}")
        if date_range[0] and date_range[1]:
            print(f"  Date range: {date_range[0][0]} to {date_range[1][0]}")
        
    except Exception as e:
        db.rollback()
        print(f"❌ Error: {e}")
        raise
    finally:
        db.close()


def load_template_data(csv_path: str = "../data/raw/techmania_2026_template.csv", auto_skip_if_exists: bool = False):
    """
    Load template data for 2026 into template_data table.
    This data serves as base for predictions and will be updated with real data.
    
    Args:
        csv_path: Path to CSV file
        auto_skip_if_exists: If True, skip loading if data already exists (non-interactive mode)
    """
    
    # Read CSV
    print(f"\n📖 Reading template data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Convert date column
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"Template CSV contains {len(df.columns)} columns and {len(df)} rows")
    
    # Create database session
    db = SessionLocal()
    
    try:
        # Check if data already exists
        existing_count = db.query(TemplateData).count()
        if existing_count > 0:
            print(f"⚠️  Database already contains {existing_count} template records.")
            if auto_skip_if_exists:
                print("⏭️  Skipping template data load (auto mode).")
                return
            response = input("Do you want to clear and reload? (yes/no): ")
            if response.lower() == 'yes':
                db.query(TemplateData).delete()
                db.commit()
                print("✓ Existing template data cleared.")
            else:
                print("⏭️  Skipping template data load.")
                return
        
        # Insert data
        print(f"💾 Loading {len(df)} template records...")
        records_added = 0
        
        for _, row in df.iterrows():
            try:
                record = TemplateData(
                    date=row['date'].date(),
                    is_complete=False,  # Initially all records are incomplete
                    
                    # Visitor data (initially null/from template)
                    day_of_week=str(row['day_of_week']) if pd.notna(row.get('day_of_week')) else None,
                    school_visitors=None,  # Will be filled when real data arrives
                    public_visitors=None,
                    total_visitors=None,
                    extra=str(row['extra']) if pd.notna(row.get('extra')) else None,
                    opening_hours=str(row['opening_hours']) if pd.notna(row.get('opening_hours')) else None,
                    
                    # Calendar features - Basic
                    is_weekend=int(row['is_weekend']) if pd.notna(row.get('is_weekend')) else None,
                    is_holiday=int(row['is_holiday']) if pd.notna(row.get('is_holiday')) else None,
                    nazvy_svatek=str(row['nazvy_svatek']) if pd.notna(row.get('nazvy_svatek')) else None,
                    
                    # Calendar features - School breaks
                    is_spring_break=int(row['is_spring_break']) if pd.notna(row.get('is_spring_break')) else None,
                    is_autumn_break=int(row['is_autumn_break']) if pd.notna(row.get('is_autumn_break')) else None,
                    is_winter_break=int(row['is_winter_break']) if pd.notna(row.get('is_winter_break')) else None,
                    is_easter_break=int(row['is_easter_break']) if pd.notna(row.get('is_easter_break')) else None,
                    is_halfyear_break=int(row['is_halfyear_break']) if pd.notna(row.get('is_halfyear_break')) else None,
                    is_summer_holiday=int(row['is_summer_holiday']) if pd.notna(row.get('is_summer_holiday')) else None,
                    is_any_school_break=int(row['is_any_school_break']) if pd.notna(row.get('is_any_school_break')) else None,
                    school_break_type=str(row['school_break_type']) if pd.notna(row.get('school_break_type')) else None,
                    days_to_next_break=int(row['days_to_next_break']) if pd.notna(row.get('days_to_next_break')) else None,
                    days_from_last_break=int(row['days_from_last_break']) if pd.notna(row.get('days_from_last_break')) else None,
                    is_week_before_break=int(row['is_week_before_break']) if pd.notna(row.get('is_week_before_break')) else None,
                    is_week_after_break=int(row['is_week_after_break']) if pd.notna(row.get('is_week_after_break')) else None,
                    
                    # Calendar features - Advanced
                    season_exact=str(row['season_exact']) if pd.notna(row.get('season_exact')) else None,
                    week_position=str(row['week_position']) if pd.notna(row.get('week_position')) else None,
                    is_month_end=int(row['is_month_end']) if pd.notna(row.get('is_month_end')) else None,
                    school_week_number=int(row['school_week_number']) if pd.notna(row.get('school_week_number')) else None,
                    is_bridge_day=int(row['is_bridge_day']) if pd.notna(row.get('is_bridge_day')) else None,
                    long_weekend_length=int(row['long_weekend_length']) if pd.notna(row.get('long_weekend_length')) else None,
                    
                    # Weather features (from template/forecast)
                    temperature_max=float(row['temperature_max']) if pd.notna(row.get('temperature_max')) else None,
                    temperature_min=float(row['temperature_min']) if pd.notna(row.get('temperature_min')) else None,
                    temperature_mean=float(row['temperature_mean']) if pd.notna(row.get('temperature_mean')) else None,
                    
                    precipitation=float(row['precipitation']) if pd.notna(row.get('precipitation')) else None,
                    precipitation_probability=float(row['precipitation_probability']) if pd.notna(row.get('precipitation_probability')) else None,
                    rain=float(row['rain']) if pd.notna(row.get('rain')) else None,
                    snowfall=float(row['snowfall']) if pd.notna(row.get('snowfall')) else None,
                    precipitation_hours=float(row['precipitation_hours']) if pd.notna(row.get('precipitation_hours')) else None,
                    weather_code=int(row['weather_code']) if pd.notna(row.get('weather_code')) else None,
                    
                    wind_speed=float(row['wind_speed']) if pd.notna(row.get('wind_speed')) else None,
                    wind_gusts_max=float(row['wind_gusts_max']) if pd.notna(row.get('wind_gusts_max')) else None,
                    
                    is_rainy=int(row['is_rainy']) if pd.notna(row.get('is_rainy')) else None,
                    is_snowy=int(row['is_snowy']) if pd.notna(row.get('is_snowy')) else None,
                    is_windy=int(row['is_windy']) if pd.notna(row.get('is_windy')) else None,
                    is_nice_weather=int(row['is_nice_weather']) if pd.notna(row.get('is_nice_weather')) else None,
                    
                    # Additional features
                    google_trend=float(row['google_trend']) if pd.notna(row.get('google_trend')) else None,
                    Mateřská_škola=float(row['Mateřská_škola']) if pd.notna(row.get('Mateřská_škola')) else None,
                    Střední_škola=float(row['Střední_škola']) if pd.notna(row.get('Střední_škola')) else None,
                    Základní_škola=float(row['Základní_škola']) if pd.notna(row.get('Základní_škola')) else None,
                    is_event=int(row['is_event']) if pd.notna(row.get('is_event')) else None,
                )
                db.add(record)
                records_added += 1
                
                # Commit in batches
                if records_added % 50 == 0:
                    db.commit()
                    print(f"  Loaded {records_added} template records...")
                    
            except Exception as e:
                print(f"Error loading template row {_}: {e}")
                continue
        
        # Final commit
        db.commit()
        print(f"\n✓ Successfully loaded {records_added} template records!")
        
        # Show statistics
        total_records = db.query(TemplateData).count()
        complete_records = db.query(TemplateData).filter(TemplateData.is_complete == True).count()
        date_range = db.query(
            TemplateData.date
        ).order_by(TemplateData.date).first(), db.query(
            TemplateData.date
        ).order_by(TemplateData.date.desc()).first()
        
        print(f"\nTemplate Data Statistics:")
        print(f"  Total records: {total_records}")
        print(f"  Complete records: {complete_records}")
        print(f"  Incomplete records: {total_records - complete_records}")
        if date_range[0] and date_range[1]:
            print(f"  Date range: {date_range[0][0]} to {date_range[1][0]}")
        
    except Exception as e:
        db.rollback()
        print(f"❌ Error: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize database and load CSV data")
    parser.add_argument(
        "--historical",
        default="../data/processed/techmania_with_weather_and_holidays.csv",
        help="Path to historical CSV file"
    )
    parser.add_argument(
        "--template",
        default="../data/raw/techmania_2026_template.csv",
        help="Path to template CSV file"
    )
    parser.add_argument(
        "--skip-historical",
        action="store_true",
        help="Skip loading historical data"
    )
    parser.add_argument(
        "--skip-template",
        action="store_true",
        help="Skip loading template data"
    )
    
    args = parser.parse_args()
    
    # Initialize database tables
    print("🔧 Initializing database...")
    init_db()
    
    # Load historical data
    if not args.skip_historical:
        load_historical_data(args.historical)
    
    # Load template data
    if not args.skip_template:
        load_template_data(args.template)
    
    print("\n✅ Database initialization complete!")
