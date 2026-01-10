"""
Vytvoří template CSV pro rok 2026 s předvyplněnými holiday features.
Ostatní sloupce (weather, návštěvnost) zůstanou prázdné pro manuální doplnění.
"""

import pandas as pd
import numpy as np
from datetime import date, timedelta
import sys
from pathlib import Path

# Přidat app do path
sys.path.append(str(Path(__file__).parent.parent / 'app'))

from services.holiday_service import holiday_service


def create_2026_template():
    """Vytvoří CSV template pro rok 2026."""
    
    print("=" * 80)
    print("VYTVÁŘENÍ TEMPLATE CSV PRO ROK 2026")
    print("=" * 80)
    
    # Vytvořit všechny dny v roce 2026
    start_date = date(2026, 1, 1)
    end_date = date(2026, 12, 31)
    
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    print(f"\n📅 Generuji {len(date_range)} dní pro rok 2026...")
    
    # Připravit data
    data = []
    
    for dt in date_range:
        current_date = dt.date()
        
        # Získat holiday info z rozšířeného servisu
        holiday_info = holiday_service.get_holiday_info(current_date)
        
        # Základní časové info
        row = {
            'date': current_date,
            
            # Návštěvnost - PRÁZDNÉ (budoucnost)
            'total_visitors': np.nan,
            'school_visitors': np.nan,
            'public_visitors': np.nan,
            
            # Svátky - VYPLNĚNO z holiday_service
            'extra': holiday_info['holiday_name'] if holiday_info['is_holiday'] else None,
            
            # Provozní info - PRÁZDNÉ (bude doplněno manuálně)
            'opening_hours': None,
            
            # WEATHER FEATURES - PRÁZDNÉ (nelze předvídat tak daleko)
            'temperature_max': np.nan,
            'temperature_min': np.nan,
            'temperature_mean': np.nan,
            'apparent_temp_max': np.nan,
            'apparent_temp_min': np.nan,
            'apparent_temp_mean': np.nan,
            'precipitation': np.nan,
            'rain': np.nan,
            'snowfall': np.nan,
            'precipitation_hours': np.nan,
            'precipitation_probability': np.nan,
            'weather_code': np.nan,
            'wind_speed_max': np.nan,
            'wind_gusts_max': np.nan,
            'wind_direction': np.nan,
            'sunshine_duration': np.nan,
            'daylight_duration': np.nan,
            'cloud_cover_percent': np.nan,
            'is_rainy': np.nan,
            'is_snowy': np.nan,
            'is_windy': np.nan,
            'is_nice_weather': np.nan,
            'sunshine_ratio': np.nan,
            'feels_like_delta': np.nan,
            'weather_forecast_confidence': np.nan,
            'temperature_trend_3d': np.nan,
            'is_weather_improving': np.nan,
            
            # HOLIDAY FEATURES - VYPLNĚNO
            'is_holiday': int(holiday_info['is_holiday']),
            'is_spring_break': int(holiday_info['is_spring_break']),
            'is_autumn_break': int(holiday_info['is_autumn_break']),
            'is_winter_break': int(holiday_info['is_winter_break']),
            'is_easter_break': int(holiday_info['is_easter_break']),
            'is_halfyear_break': int(holiday_info['is_halfyear_break']),
            'is_summer_holiday': int(holiday_info['is_summer_holiday']),
            'is_any_school_break': int(holiday_info['is_any_school_break']),
            'school_break_type': holiday_info['school_break_type'],
            'days_to_next_break': holiday_info['days_to_next_break'],
            'days_from_last_break': holiday_info['days_from_last_break'],
            'is_week_before_break': int(holiday_info['is_week_before_break']),
            'is_week_after_break': int(holiday_info['is_week_after_break']),
            'season_exact': holiday_info['season_exact'],
            'week_position': holiday_info['week_position'],
            'is_month_end': int(holiday_info['is_month_end']),
            'school_week_number': holiday_info['school_week_number'],
            'is_bridge_day': int(holiday_info['is_bridge_day']),
            'long_weekend_length': holiday_info['long_weekend_length'],
        }
        
        data.append(row)
    
    # Vytvořit DataFrame
    df = pd.DataFrame(data)
    
    # Uložit
    output_path = Path(__file__).parent.parent / 'data' / 'raw' / 'techmania_2026_template.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Template CSV vytvořen: {output_path}")
    print(f"   📊 Celkem řádků: {len(df)}")
    
    # Statistiky
    print("\n" + "=" * 80)
    print("STATISTIKY PRÁZDNIN A SVÁTKŮ V ROCE 2026")
    print("=" * 80)
    
    print(f"\n🎉 Státní svátky: {df['is_holiday'].sum()} dní")
    holidays = df[df['is_holiday'] == 1][['date', 'extra']]
    for _, row in holidays.iterrows():
        print(f"   - {row['date']}: {row['extra']}")
    
    print(f"\n🏫 Školní prázdniny (celkem): {df['is_any_school_break'].sum()} dní")
    print(f"   - Podzimní prázdniny: {df['is_autumn_break'].sum()} dní")
    print(f"   - Vánoční prázdniny: {df['is_winter_break'].sum()} dní")
    print(f"   - Pololetní prázdniny: {df['is_halfyear_break'].sum()} dní")
    print(f"   - Jarní prázdniny: {df['is_spring_break'].sum()} dní")
    print(f"   - Velikonoční prázdniny: {df['is_easter_break'].sum()} dní")
    print(f"   - Letní prázdniny: {df['is_summer_holiday'].sum()} dní")
    
    print(f"\n🌉 Bridge days (mosty): {df['is_bridge_day'].sum()} dní")
    bridges = df[df['is_bridge_day'] == 1][['date']]
    for _, row in bridges.iterrows():
        print(f"   - {row['date']}")
    
    print(f"\n📅 Prodloužené víkendy (3+ dny): {len(df[df['long_weekend_length'] >= 3])} dní")
    
    print("\n" + "=" * 80)
    print("NÁVOD K POUŽITÍ")
    print("=" * 80)
    print("""
1. ✅ Holiday features jsou již vyplněné
2. ⏳ Weather features jsou prázdné (NaN) - doplní se automaticky při predikci
3. ⏳ Návštěvnost (total_visitors) je prázdná - doplní se po skončení dne
4. ⏳ opening_hours je prázdné - doplň manuálně podle provozu

Pro přidání skutečné návštěvnosti po skončení dne:
- Otevři CSV v Excelu/LibreOffice
- Najdi datum
- Doplň sloupce: total_visitors, school_visitors, public_visitors, opening_hours
- Ulož
- Weather data se automaticky doplní při dalším trénování modelu
    """)
    
    print("\n✅ Hotovo!")
    
    return df


if __name__ == '__main__':
    df = create_2026_template()
