"""
Services pro Techmania Prediction API.
"""

from .holiday_service import HolidayService, holiday_service
from .weather_service import WeatherService, weather_service

__all__ = [
    'HolidayService',
    'holiday_service',
    'WeatherService',
    'weather_service',
]
