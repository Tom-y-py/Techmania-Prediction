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
            
            # Rozhodnout, zda použít archive nebo forecast API
            if target_date <= archive_cutoff:
                # Historická data (archive API) - ZDARMA od 1940!
                url = f"{self.ARCHIVE_API_BASE}/archive"
                params = {
                    'latitude': self.PLZEN_LAT,
                    'longitude': self.PLZEN_LON,
                    'start_date': target_date.strftime('%Y-%m-%d'),
                    'end_date': target_date.strftime('%Y-%m-%d'),
                    'daily': 'temperature_2m_max,temperature_2m_min,temperature_2m_mean,'
                            'precipitation_sum,rain_sum,snowfall_sum,precipitation_hours,'
                            'weathercode,windspeed_10m_max,windgusts_10m_max',
                    'timezone': 'Europe/Prague'
                }
            else:
                # Předpověď (forecast API) - max 16 dní dopředu
                days_ahead = (target_date - today).days
                if days_ahead > 16:
                    print(f"⚠️ Předpověď je dostupná max 16 dní dopředu (požadováno {days_ahead} dní)")
                    return self._get_default_weather()
                
                url = f"{self.FORECAST_API_BASE}/forecast"
                params = {
                    'latitude': self.PLZEN_LAT,
                    'longitude': self.PLZEN_LON,
                    'daily': 'temperature_2m_max,temperature_2m_min,temperature_2m_mean,'
                            'precipitation_sum,rain_sum,snowfall_sum,precipitation_hours,'
                            'weathercode,windspeed_10m_max,windgusts_10m_max',
                    'timezone': 'Europe/Prague',
                    'forecast_days': days_ahead + 1
                }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Najít správný den v odpovědi
            if 'daily' not in data:
                return self._get_default_weather()
            
            daily = data['daily']
            
            # Pro archive i forecast - najdeme index našeho data
            dates = [datetime.strptime(d, '%Y-%m-%d').date() for d in daily['time']]
            if target_date not in dates:
                return self._get_default_weather()
            
            idx = dates.index(target_date)
            
            # Sestavit výsledek
            weather = {
                'temperature_max': daily['temperature_2m_max'][idx],
                'temperature_min': daily['temperature_2m_min'][idx],
                'temperature_mean': daily['temperature_2m_mean'][idx],
                'precipitation': daily['precipitation_sum'][idx],
                'rain': daily['rain_sum'][idx],
                'snowfall': daily['snowfall_sum'][idx],
                'precipitation_hours': daily['precipitation_hours'][idx],
                'weather_code': daily['weathercode'][idx],
                'wind_speed_max': daily['windspeed_10m_max'][idx],
                'wind_gusts_max': daily['windgusts_10m_max'][idx],
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
            
            return weather
            
        except requests.RequestException as e:
            print(f"⚠️ Chyba při získávání počasí z API: {e}")
            return self._get_default_weather()
        except Exception as e:
            print(f"⚠️ Neočekávaná chyba: {e}")
            return self._get_default_weather()
    
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
    
    def _get_default_weather(self) -> Dict:
        """
        Vrací průměrné hodnoty počasí jako fallback.
        
        Returns:
            Slovník s průměrnými hodnotami
        """
        return {
            'temperature_max': 15.0,
            'temperature_min': 5.0,
            'temperature_mean': 10.0,
            'precipitation': 2.0,
            'rain': 2.0,
            'snowfall': 0.0,
            'precipitation_hours': 4.0,
            'weather_code': 2,  # Polojasno
            'wind_speed_max': 15.0,
            'wind_gusts_max': 25.0,
            'weather_description': "Průměrné počasí (odhad)",
            'is_rainy': False,
            'is_snowy': False,
            'is_windy': False,
            'is_nice_weather': False,
            'is_default': True
        }
    
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
        
        # Jinak použít API
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


# Globální instance
weather_service = WeatherService()


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
