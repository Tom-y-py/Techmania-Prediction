"""
Migration script to add missing columns to both template_data and historical_data tables.
This fixes SQLite OperationalError for missing columns like wind_speed, google_trend, etc.
"""
import sqlite3
import os
from pathlib import Path

# Path to database
DB_PATH = Path(__file__).parent / "techmania.db"

def check_column_exists(cursor, table_name, column_name):
    """Check if a column exists in a table."""
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [info[1] for info in cursor.fetchall()]
    return column_name in columns

def migrate():
    """Add missing columns to both template_data and historical_data tables."""
    
    if not DB_PATH.exists():
        print(f"❌ Database not found at {DB_PATH}")
        print("Please ensure the database exists before running migration.")
        return False
    
    print(f"📊 Connecting to database: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # All columns that should exist in both tables
        columns_to_add = {
            # Wind columns
            'wind_speed': 'FLOAT',
            'wind_gusts_max': 'FLOAT',
            'wind_direction': 'INTEGER',
            # Additional features
            'google_trend': 'FLOAT',
            'Mateřská_škola': 'FLOAT',
            'Střední_škola': 'FLOAT',
            'Základní_škola': 'FLOAT',
            'is_event': 'INTEGER'
        }
        
        # Process both tables
        tables = ['template_data', 'historical_data']
        all_success = True
        
        for table_name in tables:
            print(f"\n📋 Checking table: {table_name}")
            
            missing_columns = []
            for column_name, column_type in columns_to_add.items():
                if not check_column_exists(cursor, table_name, column_name):
                    missing_columns.append((column_name, column_type))
            
            if not missing_columns:
                print(f"✅ All required columns already exist in {table_name} table.")
                continue
            
            print(f"📝 Found {len(missing_columns)} missing columns in {table_name}:")
            for col, _ in missing_columns:
                print(f"   - {col}")
            
            # Add missing columns
            for column_name, column_type in missing_columns:
                sql = f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"
                print(f"🔧 Adding column: {column_name} ({column_type})")
                try:
                    cursor.execute(sql)
                except sqlite3.Error as e:
                    print(f"   ⚠️  Error adding {column_name}: {e}")
                    all_success = False
            
            # Verify the changes
            print(f"🔍 Verifying columns in {table_name}...")
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            # Show all added columns
            for col_name, _ in missing_columns:
                found = any(col[1] == col_name for col in columns)
                status = "✓" if found else "✗"
                print(f"   {status} {col_name}")
        
        conn.commit()
        print("\n✅ Migration completed successfully for all tables!")
        
        return all_success
        
    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
        conn.rollback()
        return False
    
    finally:
        conn.close()
        print("\n🔒 Database connection closed.")

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Techmania Database Migration: Add Missing Columns")
    print("=" * 60)
    print("This script will add missing columns to:")
    print("  - template_data table")
    print("  - historical_data table")
    print("\nColumns to be added (if missing):")
    print("  • wind_speed, wind_gusts_max, wind_direction")
    print("  • google_trend")
    print("  • Mateřská_škola, Střední_škola, Základní_škola")
    print("  • is_event")
    print()
    
    success = migrate()
    
    print()
    if success:
        print("🎉 Migration finished successfully!")
        print("You can now restart the Flask application.")
    else:
        print("⚠️  Migration failed. Please check the errors above.")
    
    print("=" * 60)
