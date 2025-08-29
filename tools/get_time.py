from datetime import datetime


def get_time() -> str:
    """Return the current server time as an ISO 8601 string."""
    return datetime.now().isoformat()


def get_time_schema():
    return {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current server time. Always use this when the user asks for the current time, date, or what time it is. Returns the time in ISO 8601 format.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            },
            "examples": [
                {
                    "input": "What time is it?",
                    "call": {"name": "get_time", "arguments": {}}
                },
                {
                    "input": "Can you tell me the current date and time?",
                    "call": {"name": "get_time", "arguments": {}}
                }
            ]
        }
    }
