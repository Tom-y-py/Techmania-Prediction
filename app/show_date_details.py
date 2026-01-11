from database import SessionLocal, HistoricalData
import datetime

db = SessionLocal()

target_date = datetime.date(2025, 12, 30)
rec = db.query(HistoricalData).filter(HistoricalData.date == target_date).first()

print("=" * 80)
print(f"VŠECHNY ÚDAJE PRO {target_date} Z DATABÁZE")
print("=" * 80)

if rec:
    # Základní info
    print("\n📊 ZÁKLADNÍ ÚDAJE:")
    print(f"  ID: {rec.id}")
    print(f"  Datum: {rec.date}")
    print(f"  Den v týdnu: {rec.day_of_week}")
    
    # Návštěvnost
    print("\n👥 NÁVŠTĚVNOST:")
    print(f"  Celkem: {rec.total_visitors}")
    print(f"  Školní: {rec.school_visitors}")
    print(f"  Veřejnost: {rec.public_visitors}")
    print(f"  Extra: {rec.extra}")
    
    # Kalendář - základní
    print("\n📅 KALENDÁŘ - ZÁKLADNÍ:")
    print(f"  Víkend: {rec.is_weekend}")
    print(f"  Svátek: {rec.is_holiday}")
    print(f"  Název svátku: {rec.nazvy_svatek}")
    
    # Školní prázdniny
    print("\n🏫 ŠKOLNÍ PRÁZDNINY:")
    print(f"  Jarní: {rec.is_spring_break}")
    print(f"  Podzimní: {rec.is_autumn_break}")
    print(f"  Zimní: {rec.is_winter_break}")
    print(f"  Velikonoční: {rec.is_easter_break}")
    print(f"  Pololetní: {rec.is_halfyear_break}")
    print(f"  Letní: {rec.is_summer_holiday}")
    print(f"  Jakékoliv prázdniny: {rec.is_any_school_break}")
    print(f"  Typ prázdnin: {rec.school_break_type}")
    print(f"  Dní do příštích prázdnin: {rec.days_to_next_break}")
    print(f"  Dní od posledních prázdnin: {rec.days_from_last_break}")
    print(f"  Týden před prázdninami: {rec.is_week_before_break}")
    print(f"  Týden po prázdninách: {rec.is_week_after_break}")
    
    # Kalendář - pokročilé
    print("\n📆 KALENDÁŘ - POKROČILÉ:")
    print(f"  Sezóna: {rec.season_exact}")
    print(f"  Pozice v týdnu: {rec.week_position}")
    print(f"  Konec měsíce: {rec.is_month_end}")
    print(f"  Číslo školního týdne: {rec.school_week_number}")
    print(f"  Můstek: {rec.is_bridge_day}")
    print(f"  Délka prodlouženého víkendu: {rec.long_weekend_length}")
    
    # Počasí - teplota
    print("\n🌡️ POČASÍ - TEPLOTA:")
    print(f"  Max: {rec.temperature_max}°C")
    print(f"  Min: {rec.temperature_min}°C")
    print(f"  Průměr: {rec.temperature_mean}°C")
    print(f"  Pocitová max: {rec.apparent_temp_max}°C")
    print(f"  Pocitová min: {rec.apparent_temp_min}°C")
    print(f"  Pocitová průměr: {rec.apparent_temp_mean}°C")
    print(f"  Rozdíl pocitové: {rec.feels_like_delta}°C")
    print(f"  3-denní trend: {rec.temperature_trend_3d}°C")
    
    # Počasí - srážky
    print("\n🌧️ POČASÍ - SRÁŽKY:")
    print(f"  Srážky celkem: {rec.precipitation}mm")
    print(f"  Pravděpodobnost srážek: {rec.precipitation_probability}%")
    print(f"  Déšť: {rec.rain}mm")
    print(f"  Sníh: {rec.snowfall}mm")
    print(f"  Hodin se srážkami: {rec.precipitation_hours}h")
    print(f"  Kód počasí: {rec.weather_code}")
    
    # Počasí - vítr
    print("\n💨 POČASÍ - VÍTR:")
    print(f"  Rychlost: {rec.wind_speed} m/s")
    print(f"  Max poryvy: {rec.wind_gusts_max} m/s")
    print(f"  Směr: {rec.wind_direction}°")
    
    # Počasí - slunce a oblačnost
    print("\n☀️ POČASÍ - SLUNCE & OBLAČNOST:")
    print(f"  Slunce trvání: {rec.sunshine_duration}s")
    print(f"  Denní světlo trvání: {rec.daylight_duration}s")
    print(f"  Poměr slunce: {rec.sunshine_ratio}")
    print(f"  Oblačnost: {rec.cloud_cover_percent}%")
    
    # Počasí - vypočítané
    print("\n🔧 POČASÍ - VYPOČÍTANÉ PŘÍZNAKY:")
    print(f"  Deštivo: {rec.is_rainy}")
    print(f"  Sníh: {rec.is_snowy}")
    print(f"  Větrno: {rec.is_windy}")
    print(f"  Pěkné počasí: {rec.is_nice_weather}")
    print(f"  Důvěra předpovědi: {rec.weather_forecast_confidence}")
    print(f"  Počasí se zlepšuje: {rec.is_weather_improving}")
    
    # Ostatní
    print("\n🔍 OSTATNÍ:")
    print(f"  Google Trends: {rec.google_trend}")
    print(f"  Mateřská škola: {rec.Mateřská_škola}")
    print(f"  Střední škola: {rec.Střední_škola}")
    print(f"  Základní škola: {rec.Základní_škola}")
    print(f"  Event: {rec.is_event}")
    print(f"  Vytvořeno: {rec.created_at}")
    
else:
    print("\n❌ Datum 2025-12-30 nebylo nalezeno v databázi!")

db.close()
print("\n" + "=" * 80)
