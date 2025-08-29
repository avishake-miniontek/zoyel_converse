import math
from typing import Any


def calculate(expression: str) -> str:
    """Safely evaluate a simple arithmetic expression.
    Allowed: digits, operators + - * / % ** ( ), decimal points, and whitespace.
    """
    allowed_chars = set("0123456789+-*/%.() \t\n\r")
    if not expression or any(c not in allowed_chars for c in expression):
        return "Invalid expression"
    try:
        # very restricted eval
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return str(result)
    except Exception:
        return "Invalid expression"


def calculate_schema():
    return {
        "name": "calculate",
        "description": "Use this tool to perform mathematical calculations. It can handle basic arithmetic operations including addition (+), subtraction (-), multiplication (*), division (/), modulo (%), and exponentiation (**). It also supports parentheses for grouping and decimal numbers. Always use this when a mathematical calculation is needed.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate. Example: '(2+3)*4' or '10/2 + 5**2'"
                }
            },
            "required": ["expression"]
        },
        "examples": [
            {
                "input": "What is 2 plus 2?",
                "call": {"name": "calculate", "arguments": {"expression": "2+2"}}
            },
            {
                "input": "Calculate 10 divided by 2 plus 5 squared",
                "call": {"name": "calculate", "arguments": {"expression": "10/2 + 5**2"}}
            },
            {
                "input": "What's 15% of 80?",
                "call": {"name": "calculate", "arguments": {"expression": "80*0.15"}}
            }
        ]
    }
