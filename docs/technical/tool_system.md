# Tool System Documentation

## Overview

The tool system allows the AI assistant to execute external actions through a structured interface. Tools are Python classes that implement specific functionality and can be called by the AI during conversations.

## Architecture

The tool system consists of several key components:

- `BaseTool`: Base class that all tools must inherit from
- `ToolRegistry`: Manages tool discovery, registration, and execution
- `LLMClient`: Handles tool integration with the AI conversation flow

## Creating a New Tool

To create a new tool:

1. Create a new directory under `client/src/tools/` for your tool:
```
client/src/tools/your_tool/
├── __init__.py
├── your_tool.py
└── config/
    └── config.json
```

2. Create your tool class inheriting from `BaseTool`:

```python
from ..base_tool import BaseTool

class YourTool(BaseTool):
    def __init__(self):
        super().__init__()  # Always call super().__init__() first to load config
        self.name = "your_tool_name"  # Unique identifier for the tool
        self.description = "Description of what your tool does"
        self.args = ["arg1", "arg2"]  # List of argument names in order
        self.llm_response = True  # Set to False for direct text responses
        self.result_prompt = "Optional prompt for formatting LLM responses"
        self.needs_translation = False  # Set to True to enable automatic translation

    async def execute(self, args: List[str]) -> str:
        """Execute the tool with ordered arguments."""
        try:
            # Your implementation here
            # args will be a list of strings matching the order in self.args
            result = "your result"
            return result
        except Exception as e:
            return f"Error executing tool: {str(e)}"
```

3. Create a config.json file in your tool's config directory:

```json
{
  "settings": {
    "setting1": "value1",
    "setting2": "value2"
  }
}
```

## Tool Configuration

Tools use a simple configuration system:

1. Each tool has its own config file in its config directory
2. Config files must contain a "settings" object with tool-specific settings
3. Settings are loaded automatically by the BaseTool class
4. Example (from weather tool):
```json
{
  "settings": {
    "units": {
      "temperature": "fahrenheit",
      "wind_speed": "mph",
      "precipitation": "inch"
    }
  }
}
```

## Tool Registration and Discovery

Tools are automatically discovered and registered by the `ToolRegistry` class:

1. The registry recursively scans the tools directory for Python files
2. Files are imported and classes inheriting from BaseTool are identified
3. Tools are instantiated and registered if they have a valid name
4. The registry provides a tool prompt showing available tools and usage

The tool registry also supports multilingual tool prompts:
- Tool descriptions can be provided in multiple languages
- The system automatically selects the appropriate language based on the client's configuration
- The `get_tool_prompt` method accepts a language code parameter to retrieve language-specific tool documentation

## Tool Execution Flow

1. The LLM client receives user input and identifies tool calls in the format:
```
<tool>tool_name('arg1', 'arg2')</tool>
```

2. The tool registry validates and executes the tool:
   - Checks if the tool exists
   - Passes ordered arguments to the tool's execute method
   - Returns the result

3. Results are processed based on the tool's llm_response setting:
   - If True: Result is sent to LLM for natural language formatting using result_prompt
   - If False: Result is returned directly to the user

4. If needed, tool responses are translated to the client's configured language:
   - Tools can set `needs_translation = True` to enable automatic translation
   - Translation is performed for non-English languages when needed
   - Original measurements, numbers, and proper nouns are preserved

## Example: Weather Tool

The weather tool demonstrates a complete implementation:

1. Tool Structure:
```
client/src/tools/weather/
├── __init__.py
├── weather_tool.py
└── config/
    └── config.json
```

2. Configuration:
```json
{
  "settings": {
    "units": {
      "temperature": "fahrenheit",
      "wind_speed": "mph",
      "precipitation": "inch"
    }
  }
}
```

3. Implementation:
```python
class WeatherTool(BaseTool):
    def __init__(self):
        super().__init__()
        self.name = "weather"
        self.description = "Get weather information for a location"
        self.args = ["city", "state", "country"]  # country is optional
        self.llm_response = False  # Weather tool returns formatted text directly
        self.needs_translation = True  # Enable automatic translation for non-English languages

    async def execute(self, args: List[str]) -> str:
        try:
            # Convert args list to dict based on defined order
            arg_dict = {}
            for i, value in enumerate(args):
                if value and i < len(self.args):  # Only include non-empty args
                    arg_dict[self.args[i]] = value

            # Get coordinates and weather data...
            result = "Weather information formatted as text"
            return result
        except ValueError as e:
            return str(e)
        except Exception as e:
            return "An error occurred. Please try again later."
```

## Best Practices

1. Configuration:
   - Keep settings organized and well-named
   - Include clear documentation in config files
   - Use appropriate data types for settings

2. Arguments:
   - List required arguments first in self.args
   - Use clear, descriptive argument names
   - Document argument requirements in description
   - Handle optional arguments appropriately

3. Implementation:
   - Implement comprehensive error handling
   - Use async/await for I/O operations
   - Log important operations and errors
   - Return clear, actionable error messages
   - Consider whether to use LLM response formatting

4. Testing:
   - Test with various argument combinations
   - Verify error handling paths
   - Test with missing optional arguments
   - Ensure proper API error handling

## Adding New Tools Checklist

1. [ ] Create tool directory structure
2. [ ] Implement tool class inheriting from BaseTool
3. [ ] Define ordered arguments list
4. [ ] Create config.json with necessary settings
5. [ ] Implement execute method with proper error handling
6. [ ] Test tool with various inputs
7. [ ] Document tool usage and arguments
