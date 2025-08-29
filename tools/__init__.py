from typing import Any, Dict
from .get_time import get_time, get_time_schema
from .calculate import calculate, calculate_schema
from .search_web import search_web, search_web_schema
from .get_weather import get_weather, get_weather_schema

# Mapping from tool/function name to callable
TOOL_FUNCTIONS: Dict[str, Any] = {
    "get_time": get_time,
    "calculate": calculate,
    "search_web": search_web,
    "get_weather": get_weather,
}

# OpenAI Chat Completions tool schemas (official format)
TOOL_SCHEMAS = [
    {"type": "function", "function": get_time_schema()},
    {"type": "function", "function": calculate_schema()},
    {"type": "function", "function": search_web_schema()},
    {"type": "function", "function": get_weather_schema()},
]

__all__ = ["TOOL_FUNCTIONS", "TOOL_SCHEMAS"]
