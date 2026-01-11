"""
Services pro Techmania Prediction API.
"""

from .holiday_service import HolidayService, holiday_service
from .weather_service import WeatherService, weather_service
from .event_scraper_service import EventScraperService, event_scraper_service

__all__ = [
    'HolidayService',
    'holiday_service',
    'WeatherService',
    'weather_service',
    'EventScraperService',
    'event_scraper_service',
]
