import requests

def get_weather(location: str) -> str:
    """Get current weather for a location name via Open‑Meteo APIs."""
    # 1. Geocode
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    geo_params = {"name": location, "count": 1, "language": "en", "format": "json"}
    geo_resp = requests.get(geo_url, params=geo_params)
    if geo_resp.status_code != 200:
        return f"Error: Geocoding API failure (status {geo_resp.status_code})."
    geo_data = geo_resp.json()
    if not geo_data.get("results"):
        return f"Location '{location}' not found."

    first = geo_data["results"][0]
    lat = first["latitude"]
    lon = first["longitude"]
    place_name = first.get("name")
    country = first.get("country")

    # 2. Weather API
    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_params = {
        "latitude": lat,
        "longitude": lon,
        "current_weather": "true",
        "timezone": "auto"
    }
    weather_resp = requests.get(weather_url, params=weather_params)
    if weather_resp.status_code != 200:
        return f"Error: Weather API failure (status {weather_resp.status_code})."
    wdata = weather_resp.json()
    cw = wdata.get("current_weather")
    if not cw:
        return "Current weather data unavailable."

    temp = cw.get("temperature")
    windspeed = cw.get("windspeed")
    winddir = cw.get("winddirection")
    weathercode = cw.get("weathercode")

    # Optional: Map weathercode to human description
    weather_desc = f"Weather code {weathercode}"  # For brevity; you can map codes if desired.

    location_str = f"{place_name}, {country}" if country else place_name
    return (f"Current weather in {location_str}: {temp}°C, {weather_desc}, "
            f"wind {windspeed} km/h at {winddir}°.")


def get_weather_schema():
    return {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather conditions for a specific location. Uses Open-Meteo geocoding to find latitude and longitude, then fetches current weather conditions (e.g., temperature, conditions). Returns formatted weather info.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The name of the city or location to get weather for. Can include country or state for specificity (e.g., 'New Delhi, India')."
                    }
                },
                "required": ["location"]
            },
            "examples": [
                {
                    "input": "What's the weather like in Paris?",
                    "call": {"name": "get_weather", "arguments": {"location": "Paris"}}
                },
                {
                    "input": "How's the weather in Tokyo today?",
                    "call": {"name": "get_weather", "arguments": {"location": "Tokyo"}}
                },
                {
                    "input": "Do I need an umbrella in Seattle?",
                    "call": {"name": "get_weather", "arguments": {"location": "Seattle, US"}}
                }
            ]
        }
    }
