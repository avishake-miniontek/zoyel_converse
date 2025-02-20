#!/usr/bin/env python3
"""Tool for getting weather information using Open-Meteo API."""

import json
import logging
import asyncio
import aiohttp
from ..base_tool import BaseTool

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Add a console handler if none exists
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

class WeatherTool(BaseTool):
    # Class-level configuration validation
    required_config_keys = ['default_location', 'units', 'forecast_days']
    valid_temp_units = ["celsius", "fahrenheit"]
    valid_wind_units = ["kmh", "ms", "mph", "kn"]
    valid_precip_units = ["mm", "inch"]

    def _validate_config(self, config: dict) -> None:
        """Additional validation for weather tool configuration."""
        super()._validate_config(config)
        
        settings = config['settings']
        # Validate default location
        if 'city' not in settings['default_location']:
            raise ValueError("default_location must contain a city")
        
        # Validate units
        if ('temperature' not in settings['units'] or
            'wind_speed' not in settings['units'] or
            'precipitation' not in settings['units']):
            raise ValueError("Missing required unit configurations")
        
        if settings['units']['temperature'] not in self.valid_temp_units:
            raise ValueError(f"Invalid temperature unit. Must be one of: {self.valid_temp_units}")
        if settings['units']['wind_speed'] not in self.valid_wind_units:
            raise ValueError(f"Invalid wind speed unit. Must be one of: {self.valid_wind_units}")
        if settings['units']['precipitation'] not in self.valid_precip_units:
            raise ValueError(f"Invalid precipitation unit. Must be one of: {self.valid_precip_units}")
        
        # Validate forecast days
        if not isinstance(settings['forecast_days'], int) or settings['forecast_days'] < 1 or settings['forecast_days'] > 16:
            raise ValueError("forecast_days must be an integer between 1 and 16")

    def __init__(self):
        super().__init__()  # Call super().__init__() first to load config
        self.name = "weather"
        self.description = "Get current weather and forecast information"
        self.system_role = "Weather Forecaster"
        self.llm_response = False  # Use direct text response
        self.prompt_instructions = """Empty arguments = default location
For ambiguous cities, add state/country:
- "weather in Paris" -> Paris, France
- "weather in Paris, TX" -> Paris, Texas"""
        
        # Set up schema using loaded config
        default_loc = self.config['settings']['default_location']
        default_loc_str = f"{default_loc['city']}"
        if 'state' in default_loc:
            default_loc_str += f", {default_loc['state']}"
        if 'country' in default_loc:
            default_loc_str += f", {default_loc['country']}"
        
        self.schema = {
            "type": "object",
            "properties": {
                "location": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                        "state": {"type": "string", "description": "State code (US only)"},
                        "country": {"type": "string", "description": "Two-letter country code (e.g., US for United States, GB for United Kingdom)"}
                    },
                    "required": ["city"],
                    "description": f"Location to get weather for. Default location is {default_loc_str}"
                }
            }
        }
        
    async def _get_coordinates(self, city: str, state: str = None, country: str = None) -> tuple:
        """Get coordinates for a location using Open-Meteo Geocoding API."""
        timeout = aiohttp.ClientTimeout(total=10)  # 10 second timeout
        try:
            # Create a new session for each request to ensure clean state
            async with aiohttp.ClientSession(timeout=timeout) as session:
                url = f"https://geocoding-api.open-meteo.com/v1/search"
                params = {
                    "name": city,
                    "count": 10,
                    "language": "en",
                    "format": "json"
                }
                logger.debug(f"Geocoding API request: GET {url} params={params}")
                async with session.get(url, params=params) as response:
                    text = await response.text()
                    logger.debug(f"Geocoding API response: {response.status} {text}")
                    
                    if response.status != 200:
                        error_msg = f"Unable to search for location due to a service error (HTTP {response.status}). Please try again later."
                        logger.error(f"Geocoding API error: {response.status} - {text}")
                        raise ValueError(error_msg)
            
                    try:
                        data = json.loads(text)
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON response from Geocoding API: {text}")
                        raise ValueError("Received invalid data from the location service. Please try again later.")

                    if not data.get("results"):
                        suggestion = ""
                        if len(city) > 3:
                            suggestion = " Double-check the spelling or try adding the country code (e.g., 'Paris, FR' for Paris, France)."
                        error_msg = f"I couldn't find any location matching '{city}'.{suggestion}"
                        logger.error(f"Geocoding API returned no results for city: {city}")
                        raise ValueError(error_msg)
        except aiohttp.ClientError as e:
            logger.error(f"Network error during geocoding request: {str(e)}")
            raise ValueError("I'm having trouble connecting to the location service. Please check your internet connection and try again.")
        except asyncio.TimeoutError:
            logger.error("Geocoding request timed out")
            raise ValueError("The location service is taking too long to respond. Please try again later.")

        filtered_results = data["results"]
        
        if country:
            country = country.upper()
            filtered_results = [r for r in filtered_results if r.get("country_code") == country]
            if not filtered_results:
                logger.error(f"No results found in country {country}")
                raise ValueError(f"No locations found in {country}")
        
        if state:
            state_upper = state.upper()
            state_results = []
            for r in filtered_results:
                admin1 = r.get("admin1", "").upper()
                state_mappings = {
                    "AL": "ALABAMA", "AK": "ALASKA", "AZ": "ARIZONA", "AR": "ARKANSAS",
                    "CA": "CALIFORNIA", "CO": "COLORADO", "CT": "CONNECTICUT", "DE": "DELAWARE",
                    "FL": "FLORIDA", "GA": "GEORGIA", "HI": "HAWAII", "ID": "IDAHO",
                    "IL": "ILLINOIS", "IN": "INDIANA", "IA": "IOWA", "KS": "KANSAS",
                    "KY": "KENTUCKY", "LA": "LOUISIANA", "ME": "MAINE", "MD": "MARYLAND",
                    "MA": "MASSACHUSETTS", "MI": "MICHIGAN", "MN": "MINNESOTA", "MS": "MISSISSIPPI",
                    "MO": "MISSOURI", "MT": "MONTANA", "NE": "NEBRASKA", "NV": "NEVADA",
                    "NH": "NEW HAMPSHIRE", "NJ": "NEW JERSEY", "NM": "NEW MEXICO", "NY": "NEW YORK",
                    "NC": "NORTH CAROLINA", "ND": "NORTH DAKOTA", "OH": "OHIO", "OK": "OKLAHOMA",
                    "OR": "OREGON", "PA": "PENNSYLVANIA", "RI": "RHODE ISLAND", "SC": "SOUTH CAROLINA",
                    "SD": "SOUTH DAKOTA", "TN": "TENNESSEE", "TX": "TEXAS", "UT": "UTAH",
                    "VT": "VERMONT", "VA": "VIRGINIA", "WA": "WASHINGTON", "WV": "WEST VIRGINIA",
                    "WI": "WISCONSIN", "WY": "WYOMING"
                }
                if state_upper == admin1 or (len(state_upper) == 2 and state_mappings.get(state_upper) == admin1):
                    state_results.append(r)
            if state_results:
                filtered_results = state_results
            else:
                logger.error(f"No results found in state {state_upper}")
                raise ValueError(f"No locations found in state {state_upper}")
        
        if not filtered_results:
            raise ValueError(f"No matching locations found for {city}" + 
                          (f" in {state}" if state else "") +
                          (f", {country}" if country else ""))
        
        result = filtered_results[0]
        logger.info(f"Found coordinates for {city}" +
                  (f", {result.get('admin1', '')}" if result.get('admin1') else "") +
                  (f", {result.get('country_code', '')}" if result.get('country_code') else "") +
                  f": {result['latitude']}, {result['longitude']}")
        return result["latitude"], result["longitude"]
    
    def _get_weather_code_description(self, code: int) -> str:
        """Convert WMO weather code to description."""
        codes = {
            0: "clear skys",
            1: "mainly clear skys",
            2: "partly cloudy skys",
            3: "overcast skys",
            45: "foggy",
            48: "foggy with frost",
            51: "light drizzle",
            53: "moderate drizzle",
            55: "heavy drizzle",
            56: "light freezing drizzle",
            57: "heavy freezing drizzle",
            61: "light rain",
            63: "moderate rain",
            65: "heavy rain",
            66: "light freezing rain",
            67: "heavy freezing rain",
            71: "light snow",
            73: "moderate snow",
            75: "heavy snow",
            77: "snow grains",
            80: "light rain showers",
            81: "moderate rain showers",
            82: "heavy rain showers",
            85: "light snow showers",
            86: "heavy snow showers",
            95: "thunderstorms",
            96: "thunderstorms with light hail",
            99: "thunderstorms with heavy hail"
        }
        return codes.get(code, "unknown conditions")
    
    async def execute(self, args: dict) -> dict:
        """Get weather information for the specified location."""
        try:
            # Get location from args or config
            if "location" in args:
                location = args["location"]
            else:
                location = self.config["settings"]["default_location"]
            
            # Always use units from config
            units = self.config["settings"]["units"]
            
            try:
                # Get coordinates
                lat, lon = await self._get_coordinates(
                    location["city"],
                    location.get("state"),
                    location.get("country")
                )
            except ValueError as e:
                # Pass through user-friendly location errors
                raise ValueError(str(e))
            except aiohttp.ClientError as e:
                logger.error(f"Network error during geocoding: {str(e)}")
                raise ValueError("I'm having trouble connecting to the location service. Please check your internet connection and try again.")
            except asyncio.TimeoutError:
                logger.error("Geocoding request timed out")
                raise ValueError("The location service is taking too long to respond. Please try again later.")
            
            # Get weather data using a new session
            timeout = aiohttp.ClientTimeout(total=10)  # 10 second timeout
            async with aiohttp.ClientSession(timeout=timeout) as session:
                url = "https://api.open-meteo.com/v1/forecast"
                params = {
                    "latitude": lat,
                    "longitude": lon,
                    "current": [
                        "temperature_2m",
                        "weather_code",
                        "wind_speed_10m"
                    ],
                    "daily": [
                        "temperature_2m_max",
                        "temperature_2m_min",
                        "precipitation_probability_max"
                    ],
                    "temperature_unit": units["temperature"],
                    "wind_speed_unit": units["wind_speed"],
                    "precipitation_unit": units["precipitation"],
                    "timezone": "auto",
                    "forecast_days": 1
                }
                
                logger.debug(f"Weather API request: GET {url} params={params}")
                async with session.get(url, params=params) as response:
                    try:
                        text = await response.text()
                        logger.debug(f"Weather API response: {response.status} {text}")
                        
                        if response.status != 200:
                            logger.error(f"Weather API error: {response.status} - {text}")
                            raise ValueError("I'm having trouble getting the weather data right now. Please try again later.")
                    
                        data = json.loads(text)
                        logger.info(f"Successfully retrieved weather data for {location['city']}")
                    except asyncio.TimeoutError:
                        logger.error("Weather API response timeout")
                        raise ValueError("The weather service took too long to respond. Please try again.")
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON response from Weather API: {text}")
                        raise ValueError("Received invalid weather data. Please try again later.")
                    except Exception as e:
                        logger.error(f"Unexpected error processing weather response: {str(e)}")
                        raise ValueError("An unexpected error occurred while getting weather data. Please try again later.")
                    
                    # Format current weather with units
                    current = data["current"]
                    daily = data["daily"]
                    
                    # Format the response text directly
                    conditions = self._get_weather_code_description(current['weather_code'])
                    wind_speed_unit = "miles per hour" if units["wind_speed"] == "mph" else "kilometers per hour"
                    
                    # Return formatted text directly
                    return {
                        "result": (
                            f"The current temperature is {round(current['temperature_2m'])} degrees {units['temperature']} "
                            f"with {conditions}. The wind speed is {round(current['wind_speed_10m'])} {wind_speed_unit}. "
                            f"Today's high will be {round(daily['temperature_2m_max'][0])} degrees {units['temperature']} "
                            f"and the low will be {round(daily['temperature_2m_min'][0])} degrees {units['temperature']}. "
                            f"There is a {round(daily['precipitation_probability_max'][0])} percent chance of precipitation."
                        )
                    }
                    
        except ValueError as e:
            # Pass through our custom error messages
            raise
        except aiohttp.ClientError as e:
            logger.error(f"Network error during weather request: {str(e)}")
            raise ValueError("I'm having trouble connecting to the weather service. Please check your internet connection and try again.")
        except asyncio.TimeoutError:
            logger.error("Weather request timed out")
            raise ValueError("The weather service is taking too long to respond. Please try again later.")
        except Exception as e:
            logger.error(f"Unexpected error getting weather data: {str(e)}")
            raise ValueError("Something unexpected went wrong while getting the weather data. Please try again later.")
