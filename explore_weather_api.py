"""
Standalone skript pro prozkoumání Open-Meteo Weather API.
Udělá API call a vypíše VŠECHNY dostupné parametry.

Spuštění: python explore_weather_api.py
"""

import requests
import json
from datetime import date, timedelta
from pprint import pprint

# ──────────────────────────────────────────────────────────────────────────────
# Plzeň souřadnice
LAT = 49.7384
LON = 13.3736
TODAY = date.today()
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive"

# ══════════════════════════════════════════════════════════════════════════════
# 1) FORECAST API – všechny dostupné daily parametry
# ══════════════════════════════════════════════════════════════════════════════
FORECAST_DAILY_ALL = (
    # Teplota
    "temperature_2m_max,"
    "temperature_2m_min,"
    "temperature_2m_mean,"
    "apparent_temperature_max,"
    "apparent_temperature_min,"
    "apparent_temperature_mean,"
    # Srážky
    "precipitation_sum,"
    "rain_sum,"
    "showers_sum,"
    "snowfall_sum,"
    "precipitation_hours,"
    "precipitation_probability_max,"
    "precipitation_probability_min,"
    "precipitation_probability_mean,"
    # Počasí
    "weather_code,"
    # Vítr
    "wind_speed_10m_max,"
    "wind_gusts_10m_max,"
    "wind_direction_10m_dominant,"
    # Slunce / světlo
    "sunrise,"
    "sunset,"
    "sunshine_duration,"
    "daylight_duration,"
    # UV a záření
    "uv_index_max,"
    "uv_index_clear_sky_max,"
    "shortwave_radiation_sum,"
    # Oblačnost / atmosféra
    "cloud_cover_mean,"
    "et0_fao_evapotranspiration"
)

# ══════════════════════════════════════════════════════════════════════════════
# 2) ARCHIVE API – všechny dostupné daily parametry
# ══════════════════════════════════════════════════════════════════════════════
ARCHIVE_DAILY_ALL = (
    "temperature_2m_max,"
    "temperature_2m_min,"
    "temperature_2m_mean,"
    "apparent_temperature_max,"
    "apparent_temperature_min,"
    "apparent_temperature_mean,"
    "precipitation_sum,"
    "rain_sum,"
    "snowfall_sum,"
    "precipitation_hours,"
    "weathercode,"
    "windspeed_10m_max,"
    "windgusts_10m_max,"
    "winddirection_10m_dominant,"
    "sunrise,"
    "sunset,"
    "sunshine_duration,"
    "daylight_duration,"
    "uv_index_max,"
    "uv_index_clear_sky_max,"
    "shortwave_radiation_sum,"
    "cloudcover_mean,"
    "et0_fao_evapotranspiration"
)


def separator(title: str, char: str = "═", width: int = 70):
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def print_daily_row(label: str, key: str, daily: dict, idx: int):
    """Tiskne jeden řádek z daily dat."""
    val = daily.get(key, ["N/A"])[idx] if key in daily else "❌ NEDOSTUPNÉ"
    print(f"  {label:<45} {val}")


def call_forecast_api() -> dict:
    """Zavolá Forecast API se všemi daily parametry a vrátí surová data."""
    params = {
        "latitude":     LAT,
        "longitude":    LON,
        "daily":        FORECAST_DAILY_ALL,
        "timezone":     "Europe/Prague",
        "forecast_days": 7,
    }
    print(f"🌐 GET {FORECAST_URL}")
    print(f"   params: {params}")
    resp = requests.get(FORECAST_URL, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def call_archive_api() -> dict:
    """Zavolá Archive API se všemi daily parametry a vrátí surová data."""
    end_date = TODAY - timedelta(days=6)    # archive má zpoždění 5 dní
    start_date = end_date - timedelta(days=2)
    params = {
        "latitude":   LAT,
        "longitude":  LON,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date":   end_date.strftime("%Y-%m-%d"),
        "daily":      ARCHIVE_DAILY_ALL,
        "timezone":   "Europe/Prague",
    }
    print(f"\n🌐 GET {ARCHIVE_URL}")
    print(f"   params: {params}")
    resp = requests.get(ARCHIVE_URL, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def display_meta(data: dict, api_name: str):
    """Zobrazí metadata z odpovědi."""
    separator(f"META – {api_name}")
    meta_keys = ["latitude", "longitude", "elevation", "timezone", "timezone_abbreviation",
                 "utc_offset_seconds", "generationtime_ms"]
    for k in meta_keys:
        print(f"  {k:<35} {data.get(k, 'N/A')}")


def display_daily_table(data: dict, api_name: str):
    """Zobrazí tabulku všech daily hodnot pro každý den."""
    separator(f"DAILY HODNOTY – {api_name}")
    daily = data.get("daily", {})
    if not daily:
        print("  ❌ Žádná daily data v odpovědi!")
        return

    dates = daily.get("time", [])
    keys  = [k for k in daily.keys() if k != "time"]

    print(f"\n  Počet dostupných parametrů: {len(keys)}")
    print(f"  Počet dní v odpovědi:        {len(dates)}\n")

    # Záhlaví tabulky
    header = f"  {'Parametr':<45}" + "".join(f"{d:<14}" for d in dates)
    print(header)
    print("  " + "─" * (45 + 14 * len(dates)))

    for key in sorted(keys):
        values = daily[key]
        row = f"  {key:<45}" + "".join(f"{str(v):<14}" for v in values)
        print(row)


def display_availability_summary(forecast_data: dict, archive_data: dict):
    """Porovná, které parametry jsou v obou API."""
    separator("SROVNÁNÍ: Forecast vs. Archive API dostupnost")

    fc_keys = set(forecast_data.get("daily", {}).keys()) - {"time"}
    ar_keys = set(archive_data.get("daily",  {}).keys()) - {"time"}

    only_forecast = sorted(fc_keys - ar_keys)
    only_archive  = sorted(ar_keys - fc_keys)
    both          = sorted(fc_keys & ar_keys)

    print(f"\n  ✅ V OBOU API ({len(both)} parametrů):")
    for k in both:
        print(f"     {k}")

    print(f"\n  🔵 JEN VE FORECAST API ({len(only_forecast)} parametrů):")
    for k in only_forecast:
        print(f"     {k}")

    print(f"\n  🟠 JEN V ARCHIVE API ({len(only_archive)} parametrů):")
    for k in only_archive:
        print(f"     {k}")


def display_what_we_use():
    """Vypíše, co naše WeatherService aktuálně používá a co ignoruje."""
    separator("CO NAŠE WeatherService POUŽÍVÁ")

    from_api = [
        ("temperature_max",          "temperature_2m_max",              "✅ Forecast + Archive"),
        ("temperature_min",          "temperature_2m_min",              "✅ Forecast + Archive"),
        ("temperature_mean",         "temperature_2m_mean",             "✅ Forecast + Archive"),
        ("apparent_temp_max",        "apparent_temperature_max",        "✅ Forecast + Archive"),
        ("apparent_temp_min",        "apparent_temperature_min",        "✅ Forecast + Archive"),
        ("apparent_temp_mean",       "apparent_temperature_mean",       "✅ Forecast + Archive"),
        ("precipitation",            "precipitation_sum",               "✅ Forecast + Archive"),
        ("rain",                     "rain_sum",                        "✅ Forecast + Archive"),
        ("snowfall",                 "snowfall_sum",                    "✅ Forecast + Archive"),
        ("precipitation_hours",      "precipitation_hours",             "✅ Forecast + Archive"),
        ("weather_code",             "weathercode",                     "✅ Forecast + Archive"),
        ("wind_speed_max",           "windspeed_10m_max",               "✅ Forecast + Archive"),
        ("wind_gusts_max",           "windgusts_10m_max",               "✅ Forecast + Archive"),
        ("wind_direction",           "winddirection_10m_dominant",      "✅ Forecast + Archive"),
        ("sunshine_duration",        "sunshine_duration",               "✅ Forecast + Archive"),
        ("daylight_duration",        "daylight_duration",               "✅ Forecast + Archive"),
        ("uv_index",                 "uv_index_max",                    "✅ Forecast + Archive"),
        ("cloud_cover_percent",      "cloudcover_mean",                 "✅ Forecast + Archive"),
        ("precipitation_probability","precipitation_probability_max",   "⚠️  JEN Forecast, None u Archive"),
    ]

    computed = [
        ("weather_description",      "lookup tabulka z weather_code"),
        ("is_rainy",                 "precipitation > 1.0"),
        ("is_snowy",                 "snowfall > 1.0"),
        ("is_windy",                 "wind_speed_max > 30"),
        ("is_nice_weather",          "temp>15 AND precip<1 AND code in [0,1,2]"),
        ("feels_like_delta",         "apparent_temp_mean - temperature_mean"),
        ("sunshine_ratio",           "sunshine_duration / daylight_duration"),
        ("weather_forecast_confidence","1.0 pro hist., lineární pokles pro forecast"),
        ("temperature_trend_3d",     "(temp[i] - temp[i-2]) / 2  (žádá 3 dny z API)"),
        ("is_weather_improving",     "hlasování: temp↑ + srážky↓ + oblačnost↓ (≥2/3)"),
    ]

    not_used = [
        ("showers_sum",              "přeháňky zvlášť od rain_sum"),
        ("precipitation_probability_min/mean", "jen max se bere"),
        ("sunrise / sunset",         "čas východu/západu slunce"),
        ("uv_index_clear_sky_max",   "UV index bez oblačnosti"),
        ("shortwave_radiation_sum",  "celkové sluneční záření [MJ/m²]"),
        ("et0_fao_evapotranspiration","evapotranspirace"),
        ("weather_interpretation_codes","textová interpretace rovnou z API"),
    ]

    print("\n  📥 PŘÍMO Z API:")
    print(f"  {'Naše pole':<35} {'API parametr':<40} {'Dostupnost'}")
    print("  " + "─" * 90)
    for our, api, avail in from_api:
        print(f"  {our:<35} {api:<40} {avail}")

    print("\n  🧮 DOPOČÍTÁVANÉ:")
    print(f"  {'Naše pole':<35} {'Jak se počítá'}")
    print("  " + "─" * 75)
    for field, how in computed:
        print(f"  {field:<35} {how}")

    print("\n  ❌ DOSTUPNÉ V API, ALE NEVYUŽITÉ:")
    print(f"  {'API parametr':<45} {'Popis'}")
    print("  " + "─" * 75)
    for field, desc in not_used:
        print(f"  {field:<45} {desc}")


def save_raw_json(forecast_data: dict, archive_data: dict):
    """Uloží surová JSON data pro další prozkoumání."""
    with open("weather_api_forecast_raw.json", "w", encoding="utf-8") as f:
        json.dump(forecast_data, f, indent=2, ensure_ascii=False)
    with open("weather_api_archive_raw.json", "w", encoding="utf-8") as f:
        json.dump(archive_data, f, indent=2, ensure_ascii=False)
    print("\n  💾 Surová JSON data uložena do:")
    print("     weather_api_forecast_raw.json")
    print("     weather_api_archive_raw.json")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("  Open-Meteo API Explorer – Techmania Weather")
    print(f"  Plzeň | lat={LAT}, lon={LON} | Dnes: {TODAY}")
    print("=" * 70)

    # 1) Forecast API
    separator("VOLÁM FORECAST API", char="─")
    try:
        forecast_data = call_forecast_api()
        display_meta(forecast_data, "Forecast API")
        display_daily_table(forecast_data, "Forecast API (příštích 7 dní)")
        forecast_ok = True
    except Exception as e:
        print(f"\n  ❌ Forecast API selhalo: {e}")
        forecast_data = {}
        forecast_ok = False

    # 2) Archive API
    separator("VOLÁM ARCHIVE API", char="─")
    try:
        archive_data = call_archive_api()
        display_meta(archive_data, "Archive API")
        display_daily_table(archive_data, "Archive API (posledních 3 dní)")
        archive_ok = True
    except Exception as e:
        print(f"\n  ❌ Archive API selhalo: {e}")
        archive_data = {}
        archive_ok = False

    # 3) Srovnání dostupnosti
    if forecast_ok and archive_ok:
        display_availability_summary(forecast_data, archive_data)

    # 4) Co naše WeatherService aktuálně dělá
    display_what_we_use()

    # 5) Uložit surová JSON
    if forecast_ok or archive_ok:
        separator("UKLÁDÁM SUROVÁ JSON DATA", char="─")
        save_raw_json(
            forecast_data if forecast_ok else {},
            archive_data  if archive_ok  else {}
        )

    print("\n" + "=" * 70)
    print("  ✅ Hotovo!")
    print("=" * 70)
