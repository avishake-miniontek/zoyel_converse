from .get_time import get_time, get_time_schema
from .calculate import calculate, calculate_schema
from .search_web import search_web, search_web_schema
from .get_weather import get_weather, get_weather_schema
from .save_patient_data import save_patient_data, save_patient_data_schema
from .fetch_patient_data import fetch_patient_data, fetch_patient_data_schema

# Dictionary mapping function names to actual functions
TOOL_FUNCTIONS = {
    "get_time": get_time,
    "calculate": calculate,
    "search_web": search_web,
    "get_weather": get_weather,
    "save_patient_data": save_patient_data,
    "fetch_patient_data": fetch_patient_data,
}

# List of tool schemas for OpenAI function calling
TOOL_SCHEMAS = [
    get_time_schema(),
    calculate_schema(),
    search_web_schema(),
    get_weather_schema(),
    save_patient_data_schema(),
    fetch_patient_data_schema(),
]