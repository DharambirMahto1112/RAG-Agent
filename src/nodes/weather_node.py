"""
Weather node for handling weather-related queries.
"""
from typing import Dict, Any
import re
from ..services.weather_service import WeatherService

class WeatherNode:
    """Node for processing weather-related queries."""
    
    def __init__(self):
        self.weather_service = WeatherService()
    
    def extract_location(self, query: str) -> tuple[str, str]:
        """
        Extract city and country from query.
        
        Args:
            query: User query string
            
        Returns:
            Tuple of (city, country_code)
        """
        query_lower = query.lower()
        
        # Common patterns for location extraction - improved regex patterns
        patterns = [
            # "weather in London", "weather at Paris", "weather for New York"
            r'weather\s+(?:in|at|for)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)',
            # "London weather", "New York weather"
            r'([a-zA-Z]+(?:\s+[a-zA-Z]+)*)\s+weather',
            # "temperature in London", "temperature at Paris"
            r'temperature\s+(?:in|at|for)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)',
            # "forecast for London", "forecast in Paris"
            r'forecast\s+(?:for|in|at)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)',
            # "what is the weather in London" or "what is the weather now in London"
            r'what\s+is\s+the\s+weather\s+(?:now\s+)?(?:in|at|for)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)',
            # "tell me the weather in London" or "tell me the weather now in London"
            r'tell\s+me\s+(?:the\s+)?weather\s+(?:now\s+)?(?:in|at|for)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)',
            # "how is the weather in London" or "how is the weather now in London"
            r'how\s+is\s+the\s+weather\s+(?:now\s+)?(?:in|at|for)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)',
            # "what's the weather in London" or "what's the weather now in London"
            r'what\'?s\s+the\s+weather\s+(?:now\s+)?(?:in|at|for)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)',
            # "current weather in London"
            r'current\s+weather\s+(?:in|at|for)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                location = match.group(1).strip()
                # Clean up the location - remove common stop words that might be captured
                location = re.sub(r'\b(what|is|the|in|at|for|weather|temperature|forecast)\b', '', location).strip()
                
                # If location is empty after cleaning, skip this match
                if not location:
                    continue
                    
                # Try to split city and country
                parts = location.split(',')
                if len(parts) == 2:
                    return parts[0].strip(), parts[1].strip()
                else:
                    return location, None
        
        # If no pattern matches, try to extract any capitalized words (but exclude common words)
        words = query.split()
        excluded_words = {'What', 'Is', 'The', 'Weather', 'Temperature', 'Forecast', 'In', 'At', 'For', 'Tell', 'Me', 'How'}
        capitalized_words = [word for word in words if word[0].isupper() and len(word) > 2 and word not in excluded_words]
        
        if capitalized_words:
            return capitalized_words[0], None
        
        return "London", None  # Default fallback
    
    def determine_query_type(self, query: str) -> str:
        """
        Determine if the query is asking for current weather or forecast.
        
        Args:
            query: User query string
            
        Returns:
            Query type: 'current' or 'forecast'
        """
        query_lower = query.lower()
        
        forecast_keywords = ['forecast', 'tomorrow', 'next week', 'upcoming', 'future']
        
        if any(keyword in query_lower for keyword in forecast_keywords):
            return 'forecast'
        else:
            return 'current'
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process weather-related query.
        
        Args:
            state: Current state dictionary
            
        Returns:
            Updated state with weather response
        """
        query = state.get("query", "")
        
        try:
            # Extract location
            city, country_code = self.extract_location(query)
            
            # Debug logging
            print(f"DEBUG - Weather Node: Query='{query}', Extracted city='{city}', country='{country_code}'")
            
            # Determine query type
            query_type = self.determine_query_type(query)
            
            # Get weather data
            if query_type == 'forecast':
                weather_data = self.weather_service.get_weather_forecast(city, country_code)
            else:
                weather_data = self.weather_service.get_current_weather(city, country_code)
            
            # Format response
            if "error" in weather_data:
                response = f"❌ {weather_data['error']}"
            else:
                response = self.weather_service.format_weather_response(weather_data)
            
            # Update state
            state["weather_data"] = weather_data
            state["response"] = response
            state["response_type"] = "weather"
            state["location_used"] = f"{city}, {country_code}" if country_code else city
            
            return state
            
        except Exception as e:
            error_response = f"❌ Error processing weather request: {str(e)}"
            state["response"] = error_response
            state["response_type"] = "error"
            state["error"] = str(e)
            
            return state
