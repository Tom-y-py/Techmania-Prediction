"""
Služba pro získání informací o počasí.
Kombinuje historická data a API předpověď pro budoucnost.
"""

from datetime import date, datetime, timedelta
from typing import Dict, Optional, List
import pandas as pd
import requests
from pathlib import Path


class WeatherService:
    """
    Služba pro získání informací o počasí.
    - Historická data z CSV
    - Aktuální počasí a předpověď z API (Open-Meteo - ZDARMA, bez API klíče)
    """
    
    # Plzeň souřadnice
    PLZEN_LAT = 49.7384
    PLZEN_LON = 13.3736
    
    # Open-Meteo API (free, bez registrace)
    # Archive API pro historická data (1940-present, delay 5 dní)
    ARCHIVE_API_BASE = "https://archive-api.open-meteo.com/v1"
    # Forecast API pro aktuální počasí a předpověď
    FORECAST_API_BASE = "https://api.open-meteo.com/v1"
    
    def __init__(self, historical_data_path: Optional[str] = None):
        """
        Inicializace služby.
        
        Args:
            historical_data_path: Cesta k CSV s historickými daty počasí
        """
        self.historical_data = None
        
        if historical_data_path:
            try:
                self.historical_data = pd.read_csv(historical_data_path)
                self.historical_data['date'] = pd.to_datetime(self.historical_data['date'])
                print(f"✅ Načteno {len(self.historical_data)} historických záznamů počasí")
            except Exception as e:
                print(f"⚠️ Nepodařilo se načíst historická data: {e}")
    
    def get_weather_from_api(self, target_date: date) -> Optional[Dict]:
        """
        Získá počasí z Open-Meteo API.
        Podporuje historická data (od 1940) i předpověď (16 dní dopředu).
        
        Args:
            target_date: Datum pro získání počasí
            
        Returns:
            Slovník s informacemi o počasí nebo None při chybě
        """
        try:
            today = date.today()
            # Datum s 5-denním zpožděním (hranice mezi archive a forecast)
            archive_cutoff = today - timedelta(days=5)
            
            # Pro výpočet trendů potřebujeme 3 dny (target_date -2, -1, target_date)
            start_date_for_trend = target_date - timedelta(days=2)
            
            # Rozhodnout, zda použít archive nebo forecast API
            if target_date <= archive_cutoff:
                # Historická data (archive API) - ZDARMA od 1940!
                url = f"{self.ARCHIVE_API_BASE}/archive"
                params = {
                    'latitude': self.PLZEN_LAT,
                    'longitude': self.PLZEN_LON,
                    'start_date': start_date_for_trend.strftime('%Y-%m-%d'),
                    'end_date': target_date.strftime('%Y-%m-%d'),
                    'daily': 'temperature_2m_max,temperature_2m_min,temperature_2m_mean,'
                            'apparent_temperature_max,apparent_temperature_min,apparent_temperature_mean,'
                            'precipitation_sum,rain_sum,snowfall_sum,precipitation_hours,'
                            'weathercode,windspeed_10m_max,windgusts_10m_max,winddirection_10m_dominant,'
                            'sunshine_duration,daylight_duration,'
                            'uv_index_max,cloudcover_mean',
                    'timezone': 'Europe/Prague'
                }
            else:
                # Předpověď (forecast API) - max 16 dní dopředu
                days_ahead = (target_date - today).days
                if days_ahead > 16:
                    self._raise_no_data_error(
                        target_date,
                        f"Forecast is only available up to 16 days ahead (requested {days_ahead} days)"
                    )
                
                url = f"{self.FORECAST_API_BASE}/forecast"
                params = {
                    'latitude': self.PLZEN_LAT,
                    'longitude': self.PLZEN_LON,
                    'daily': 'temperature_2m_max,temperature_2m_min,temperature_2m_mean,'
                            'apparent_temperature_max,apparent_temperature_min,apparent_temperature_mean,'
                            'precipitation_sum,rain_sum,snowfall_sum,precipitation_hours,'
                            'weathercode,windspeed_10m_max,windgusts_10m_max,winddirection_10m_dominant,'
                            'sunshine_duration,daylight_duration,'
                            'uv_index_max,cloudcover_mean,'
                            'precipitation_probability_max',
                    'timezone': 'Europe/Prague',
                    'forecast_days': days_ahead + 3  # +3 pro trend
                }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Najít správný den v odpovědi
            if 'daily' not in data:
                self._raise_no_data_error(target_date, "API response missing 'daily' data")
            
            daily = data['daily']
            
            # Pro archive i forecast - najdeme index našeho data
            dates = [datetime.strptime(d, '%Y-%m-%d').date() for d in daily['time']]
            if target_date not in dates:
                self._raise_no_data_error(target_date, f"Date not found in API response (available: {dates[0]} to {dates[-1]})")
            
            idx = dates.index(target_date)
            
            # Sestavit výsledek - základní parametry
            weather = {
                'temperature_max': daily['temperature_2m_max'][idx],
                'temperature_min': daily['temperature_2m_min'][idx],
                'temperature_mean': daily['temperature_2m_mean'][idx],
                'apparent_temp_max': daily.get('apparent_temperature_max', [None]*len(dates))[idx],
                'apparent_temp_min': daily.get('apparent_temperature_min', [None]*len(dates))[idx],
                'apparent_temp_mean': daily.get('apparent_temperature_mean', [None]*len(dates))[idx],
                'precipitation': daily['precipitation_sum'][idx],
                'rain': daily['rain_sum'][idx],
                'snowfall': daily['snowfall_sum'][idx],
                'precipitation_hours': daily['precipitation_hours'][idx],
                'weather_code': daily['weathercode'][idx],
                'wind_speed_max': daily['windspeed_10m_max'][idx],
                'wind_gusts_max': daily['windgusts_10m_max'][idx],
                'wind_direction': daily.get('winddirection_10m_dominant', [None]*len(dates))[idx],
                'sunshine_duration': daily.get('sunshine_duration', [None]*len(dates))[idx],
                'daylight_duration': daily.get('daylight_duration', [None]*len(dates))[idx],
                'uv_index': daily.get('uv_index_max', [None]*len(dates))[idx],
                'cloud_cover_percent': daily.get('cloudcover_mean', [None]*len(dates))[idx],
                'precipitation_probability': daily.get('precipitation_probability_max', [None]*len(dates))[idx],
            }
            
            # Přidat interpretaci
            weather['weather_description'] = self._interpret_weather_code(weather['weather_code'])
            weather['is_rainy'] = weather['precipitation'] > 1.0
            weather['is_snowy'] = weather['snowfall'] > 1.0
            weather['is_windy'] = weather['wind_speed_max'] > 30
            weather['is_nice_weather'] = (
                weather['temperature_mean'] > 15 and 
                weather['precipitation'] < 1.0 and
                weather['weather_code'] in [0, 1, 2]  # Clear, mainly clear, partly cloudy
            )
            
            # Vypočítat odvozené features
            # feels_like_delta: Rozdíl mezi pocitovou a skutečnou teplotou
            if weather['apparent_temp_mean'] is not None and weather['temperature_mean'] is not None:
                weather['feels_like_delta'] = weather['apparent_temp_mean'] - weather['temperature_mean']
            else:
                weather['feels_like_delta'] = 0.0
            
            # sunshine_ratio: Poměr slunečního svitu k délce dne
            if weather['sunshine_duration'] is not None and weather['daylight_duration'] is not None and weather['daylight_duration'] > 0:
                weather['sunshine_ratio'] = weather['sunshine_duration'] / weather['daylight_duration']
            else:
                weather['sunshine_ratio'] = 0.0
            
            # weather_forecast_confidence: Spolehlivost předpovědi (0-1)
            # Čím vzdálenější datum, tím nižší spolehlivost
            # Pro historická data je to 1.0, pro předpověď klesá s časem
            today = date.today()
            if target_date <= today - timedelta(days=5):
                weather['weather_forecast_confidence'] = 1.0  # Historická data
            else:
                days_ahead = (target_date - today).days
                # Lineární pokles od 1.0 (dnes) do 0.5 (14 dní dopředu)
                weather['weather_forecast_confidence'] = max(0.5, 1.0 - (days_ahead * 0.035))
            
            # temperature_trend_3d: Trend teploty za 3 dny (target_date -2, -1, target_date)
            # Pozitivní = oteplování, negativní = ochlazování
            if len(dates) >= 3:
                # Najít indexy posledních 3 dnů
                target_idx = dates.index(target_date)
                if target_idx >= 2:
                    temp_day_minus_2 = daily['temperature_2m_mean'][target_idx - 2]
                    temp_day_minus_1 = daily['temperature_2m_mean'][target_idx - 1]
                    temp_today = daily['temperature_2m_mean'][target_idx]
                    
                    # Průměrný denní růst/pokles teploty
                    weather['temperature_trend_3d'] = (temp_today - temp_day_minus_2) / 2.0
                    
                    # is_weather_improving: Počasí se zlepšuje
                    # Kritéria: teplota roste NEBO srážky klesají NEBO oblačnost klesá
                    temp_improving = temp_today > temp_day_minus_1
                    
                    precip_day_minus_1 = daily['precipitation_sum'][target_idx - 1]
                    precip_today = daily['precipitation_sum'][target_idx]
                    rain_improving = precip_today < precip_day_minus_1
                    
                    # Pokud máme data o oblačnosti
                    if 'cloudcover_mean' in daily:
                        cloud_day_minus_1 = daily['cloudcover_mean'][target_idx - 1]
                        cloud_today = daily['cloudcover_mean'][target_idx]
                        cloud_improving = cloud_today < cloud_day_minus_1 if cloud_today is not None else False
                    else:
                        cloud_improving = False
                    
                    # Počasí se zlepšuje, pokud alespoň 2 ze 3 kritérií platí
                    improvements = sum([temp_improving, rain_improving, cloud_improving])
                    weather['is_weather_improving'] = 1 if improvements >= 2 else 0
                else:
                    weather['temperature_trend_3d'] = 0.0
                    weather['is_weather_improving'] = 0
            else:
                weather['temperature_trend_3d'] = 0.0
                weather['is_weather_improving'] = 0
            
            return weather
            
        except requests.RequestException as e:
            error_msg = f"API request failed: {str(e)}"
            print(f"❌ {error_msg}")
            self._raise_no_data_error(target_date, error_msg)
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            print(f"❌ {error_msg}")
            self._raise_no_data_error(target_date, error_msg)
    
    def _interpret_weather_code(self, code: int) -> str:
        """
        Interpretuje WMO weather code do čitelného textu.
        
        Args:
            code: WMO weather code
            
        Returns:
            Popis počasí
        """
        weather_codes = {
            0: "Jasno",
            1: "Převážně jasno",
            2: "Polojasno",
            3: "Zataženo",
            45: "Mlha",
            48: "Náledí z mlhy",
            51: "Mrholení: lehké",
            53: "Mrholení: mírné",
            55: "Mrholení: husté",
            61: "Déšť: slabý",
            63: "Déšť: mírný",
            65: "Déšť: silný",
            71: "Sněžení: slabé",
            73: "Sněžení: mírné",
            75: "Sněžení: silné",
            77: "Sněhové vločky",
            80: "Přeháňky: slabé",
            81: "Přeháňky: mírné",
            82: "Přeháňky: silné",
            85: "Sněhové přeháňky: slabé",
            86: "Sněhové přeháňky: silné",
            95: "Bouřka",
            96: "Bouřka s kroupami: slabá",
            99: "Bouřka s kroupami: silná",
        }
        return weather_codes.get(code, "Neznámé")
    
    def _raise_no_data_error(self, target_date: date, reason: str) -> None:
        """
        Vyhodí chybu, pokud nejsou dostupná žádná data o počasí.
        
        Args:
            target_date: Datum, pro které data chybí
            reason: Důvod, proč data nejsou k dispozici
            
        Raises:
            ValueError: Vždy - data nejsou k dispozici
        """
        raise ValueError(
            f"Weather data not available for {target_date.strftime('%Y-%m-%d')}. "
            f"Reason: {reason}. Cannot make prediction without real weather data."
        )
    
    def get_weather(self, target_date: date) -> Dict:
        """
        Hlavní metoda - získá počasí pro dané datum.
        Nejdřív zkusí historická data, pak API.
        
        Args:
            target_date: Datum
            
        Returns:
            Slovník s informacemi o počasí
        """
        # Zkusit historická data
        if self.historical_data is not None:
            row = self.historical_data[self.historical_data['date'] == pd.to_datetime(target_date)]
            if not row.empty:
                return row.iloc[0].to_dict()
        
        # Pokud nejsou historická data, zkusit API
        if self.historical_data is None:
            print(f"⚠️ Historical data not loaded, trying API for {target_date}")
        
        # API
        return self.get_weather_from_api(target_date)
    
    def get_weather_for_range(self, start_date: date, end_date: date) -> pd.DataFrame:
        """
        Získá počasí pro celé období.
        
        Args:
            start_date: Začátek období
            end_date: Konec období
            
        Returns:
            DataFrame s počasím pro každý den
        """
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        weather_data = []
        for dt in date_range:
            weather = self.get_weather(dt.date())
            weather['date'] = dt.date()
            weather_data.append(weather)
        
        return pd.DataFrame(weather_data)


# Globální instance bez historických dat (používáme Open-Meteo API)
# Pro rychlost můžeme API volat přímo - Archive API je zdarma od roku 1940
weather_service = WeatherService(historical_data_path=None)


if __name__ == '__main__':
    # Test služby
    print("=" * 60)
    print("Testing Weather Service")
    print("=" * 60)
    
    service = WeatherService()
    
    # Test historického data
    print("\n🕒 Test historického data (2025-01-01):")
    weather = service.get_weather(date(2025, 1, 1))
    for key, value in weather.items():
        print(f"  {key}: {value}")
    
    # Test dnešního data
    print(f"\n☀️ Test dnešního data ({date.today()}):")
    weather = service.get_weather(date.today())
    for key, value in weather.items():
        print(f"  {key}: {value}")
    
    # Test budoucího data
    future_date = date.today() + timedelta(days=7)
    print(f"\n🔮 Test předpovědi ({future_date}):")
    weather = service.get_weather(future_date)
    for key, value in weather.items():
        print(f"  {key}: {value}")
    
    # Test období
    print(f"\n📊 Test období (7 dní od dnes):")
    weather_df = service.get_weather_for_range(date.today(), future_date)
    print(weather_df[['date', 'temperature_mean', 'precipitation', 'weather_description']].to_string())
