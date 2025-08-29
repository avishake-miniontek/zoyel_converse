from typing import List


def search_web(query: str) -> str:
    """Mock web search that returns a simple text summary.
    Replace with a real search implementation if desired.
    """
    return f"Search results for '{query}': This is a mock response."


def search_web_schema():
    return {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for the given query and return a brief summary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string"
                    }
                },
                "required": ["query"]
            }
        }
    }
