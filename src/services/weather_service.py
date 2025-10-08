"""
Weather API service for fetching real-time weather data.
"""
import requests
import json
from typing import Dict, Any, Optional
from datetime import datetime
from .config import get_settings

class WeatherService:
    """Service for interacting with OpenWeatherMap API."""
    
    def __init__(self):
        self.settings = get_settings()
        self.base_url = "https://api.openweathermap.org/data/2.5"
        self.api_key = self.settings.openweather_api_key
    
    def get_current_weather(self, city: str, country_code: Optional[str] = None) -> Dict[str, Any]:
        """
        Get current weather data for a city.
        
        Args:
            city: Name of the city
            country_code: Optional country code (e.g., 'US', 'UK')
            
        Returns:
            Dictionary containing weather data
        """
        try:
            # Validate API key before making the request
            if not self.api_key:
                return {
                    "error": "Missing OPENWEATHER_API_KEY. Set it in your .env file.",
                    "location": city,
                    "timestamp": datetime.now().isoformat()
                }
            # Construct location string
            location = f"{city},{country_code}" if country_code else city
            
            # Construct complete URL as specified: https://api.openweathermap.org/data/2.5/weather?q={CITY_NAME}&appid={API_KEY}&units=metric
            url = f"{self.base_url}/weather?q={location}&appid={self.api_key}&units=metric"
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Format the response
            weather_data = {
                "location": {
                    "city": data["name"],
                    "country": data["sys"]["country"],
                    "coordinates": {
                        "lat": data["coord"]["lat"],
                        "lon": data["coord"]["lon"]
                    }
                },
                "current_weather": {
                    "temperature": data["main"]["temp"],
                    "feels_like": data["main"]["feels_like"],
                    "humidity": data["main"]["humidity"],
                    "pressure": data["main"]["pressure"],
                    "description": data["weather"][0]["description"],
                    "main": data["weather"][0]["main"],
                    "wind_speed": data["wind"]["speed"],
                    "wind_direction": data["wind"].get("deg", "N/A"),
                    "visibility": data.get("visibility", "N/A"),
                    "cloudiness": data["clouds"]["all"]
                },
                "timestamp": datetime.now().isoformat(),
                "source": "OpenWeatherMap"
            }
            
            return weather_data
            
        except requests.exceptions.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status == 401:
                return {
                    "error": "Unauthorized (401). Your OPENWEATHER_API_KEY is invalid or not activated. Verify the key, account activation, and API access.",
                    "location": city,
                    "timestamp": datetime.now().isoformat()
                }
            return {
                "error": f"Failed to fetch weather data: {str(e)}",
                "location": city,
                "timestamp": datetime.now().isoformat()
            }
        except requests.exceptions.RequestException as e:
            return {
                "error": f"Failed to fetch weather data: {str(e)}",
                "location": city,
                "timestamp": datetime.now().isoformat()
            }
        except KeyError as e:
            return {
                "error": f"Unexpected API response format: {str(e)}",
                "location": city,
                "timestamp": datetime.now().isoformat()
            }
    
    def get_weather_forecast(self, city: str, country_code: Optional[str] = None, days: int = 5) -> Dict[str, Any]:
        """
        Get weather forecast for a city.
        
        Args:
            city: Name of the city
            country_code: Optional country code
            days: Number of days for forecast (max 5)
            
        Returns:
            Dictionary containing forecast data
        """
        try:
            # Validate API key before making the request
            if not self.api_key:
                return {
                    "error": "Missing OPENWEATHER_API_KEY. Set it in your .env file.",
                    "location": city,
                    "timestamp": datetime.now().isoformat()
                }
            location = f"{city},{country_code}" if country_code else city
            
            # Construct complete URL for forecast
            url = f"{self.base_url}/forecast?q={location}&appid={self.api_key}&units=metric&cnt={days * 8}"
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Process forecast data
            forecast_data = {
                "location": {
                    "city": data["city"]["name"],
                    "country": data["city"]["country"],
                    "coordinates": {
                        "lat": data["city"]["coord"]["lat"],
                        "lon": data["city"]["coord"]["lon"]
                    }
                },
                "forecast": [],
                "timestamp": datetime.now().isoformat(),
                "source": "OpenWeatherMap"
            }
            
            for item in data["list"]:
                forecast_item = {
                    "datetime": item["dt_txt"],
                    "temperature": {
                        "min": item["main"]["temp_min"],
                        "max": item["main"]["temp_max"],
                        "current": item["main"]["temp"]
                    },
                    "humidity": item["main"]["humidity"],
                    "pressure": item["main"]["pressure"],
                    "description": item["weather"][0]["description"],
                    "wind_speed": item["wind"]["speed"],
                    "cloudiness": item["clouds"]["all"]
                }
                forecast_data["forecast"].append(forecast_item)
            
            return forecast_data
            
        except requests.exceptions.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status == 401:
                return {
                    "error": "Unauthorized (401). Your OPENWEATHER_API_KEY is invalid or not activated. Verify the key, account activation, and API access.",
                    "location": city,
                    "timestamp": datetime.now().isoformat()
                }
            return {
                "error": f"Failed to fetch forecast data: {str(e)}",
                "location": city,
                "timestamp": datetime.now().isoformat()
            }
        except requests.exceptions.RequestException as e:
            return {
                "error": f"Failed to fetch forecast data: {str(e)}",
                "location": city,
                "timestamp": datetime.now().isoformat()
            }
        except KeyError as e:
            return {
                "error": f"Unexpected API response format: {str(e)}",
                "location": city,
                "timestamp": datetime.now().isoformat()
            }
    
    def format_weather_response(self, weather_data: Dict[str, Any]) -> str:
        """
        Format weather data into a human-readable string.
        
        Args:
            weather_data: Weather data dictionary
            
        Returns:
            Formatted weather information string
        """
        if "error" in weather_data:
            return f"❌ Error: {weather_data['error']}"
        
        location = weather_data["location"]
        current = weather_data["current_weather"]
        
        response = f"""

🌤️ **Weather Report for {location['city']}, {location['country']}**

🌡️ **Temperature**: {current['temperature']}°C (feels like {current['feels_like']}°C)
☁️ **Condition**: {current['description'].title()}
💧 **Humidity**: {current['humidity']}%
🌬️ **Wind**: {current['wind_speed']} m/s
👁️ **Visibility**: {current['visibility']}m
☁️ **Cloudiness**: {current['cloudiness']}%

📍 **Coordinates**: {location['coordinates']['lat']:.2f}°N, {location['coordinates']['lon']:.2f}°E
🕐 **Last Updated**: {weather_data['timestamp']}
        """.strip()
        
        return response
