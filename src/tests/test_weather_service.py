"""
Unit tests for Weather Service.
"""
import pytest
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.services.weather_service import WeatherService

class TestWeatherService:
    """Test cases for WeatherService."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.weather_service = WeatherService()
    
    @patch('requests.get')
    def test_get_current_weather_success(self, mock_get):
        """Test successful weather data retrieval."""
        # Mock successful API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "name": "London",
            "sys": {"country": "GB"},
            "coord": {"lat": 51.5074, "lon": -0.1278},
            "main": {
                "temp": 15.5,
                "feels_like": 14.2,
                "humidity": 65,
                "pressure": 1013
            },
            "weather": [{"description": "clear sky", "main": "Clear"}],
            "wind": {"speed": 3.2, "deg": 180},
            "clouds": {"all": 10},
            "visibility": 10000
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Test the method
        result = self.weather_service.get_current_weather("London")
        
        # Assertions
        assert "error" not in result
        assert result["location"]["city"] == "London"
        assert result["location"]["country"] == "GB"
        assert result["current_weather"]["temperature"] == 15.5
        assert result["current_weather"]["description"] == "clear sky"
        assert result["source"] == "OpenWeatherMap"
    
    @patch('requests.get')
    def test_get_current_weather_api_error(self, mock_get):
        """Test weather API error handling."""
        # Mock API error
        mock_get.side_effect = Exception("API Error")
        
        result = self.weather_service.get_current_weather("London")
        
        # Assertions
        assert "error" in result
        assert "Failed to fetch weather data" in result["error"]
        assert result["location"] == "London"
    
    def test_format_weather_response(self):
        """Test weather response formatting."""
        weather_data = {
            "location": {
                "city": "London",
                "country": "GB",
                "coordinates": {"lat": 51.5074, "lon": -0.1278}
            },
            "current_weather": {
                "temperature": 15.5,
                "feels_like": 14.2,
                "humidity": 65,
                "pressure": 1013,
                "description": "clear sky",
                "wind_speed": 3.2,
                "visibility": 10000,
                "cloudiness": 10
            },
            "timestamp": "2024-01-01T12:00:00"
        }
        
        result = self.weather_service.format_weather_response(weather_data)
        
        # Assertions
        assert "London, GB" in result
        assert "15.5°C" in result
        assert "clear sky" in result
        assert "65%" in result
        assert "3.2 m/s" in result
    
    def test_format_weather_response_error(self):
        """Test weather response formatting with error."""
        weather_data = {"error": "API Error"}
        
        result = self.weather_service.format_weather_response(weather_data)
        
        # Assertions
        assert "❌ Error: API Error" in result
