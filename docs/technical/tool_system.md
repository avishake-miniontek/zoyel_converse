# Tool System Documentation

## Overview

The tool system allows the AI assistant to execute external actions through a structured interface. Tools are Python classes that implement specific functionality and can be called by the AI during conversations.

## Architecture

The tool system consists of several key components:

- `BaseTool`: Abstract base class that all tools must inherit from
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
    # Required class-level configuration validation
    required_config_keys = ['required_setting1', 'required_setting2']
    optional_config_keys = ['optional_setting1']

    def __init__(self):
        super().__init__()  # Always call super().__init__() first to load config
        self.name = "your_tool_name"  # Unique identifier for the tool
        self.description = "Description of what your tool does"
        self.system_role = "Role for LLM when processing results"
        self.llm_response = True  # Set to False for direct text responses
        self.prompt_instructions = "Instructions for the AI on how to use this tool"
        
        # Define the JSON schema for tool arguments
        self.schema = {
            "type": "object",
            "properties": {
                "param1": {
                    "type": "string",
                    "description": "Description of parameter 1"
                },
                # Add more parameters as needed
            },
            "required": ["param1"]  # List required parameters
        }

    async def execute(self, args: dict) -> dict:
        """Implement the tool's functionality."""
        try:
            # Your implementation here
            result = {"result": "your result"}
            return result
        except Exception as e:
            raise ValueError(f"Error executing tool: {str(e)}")
```

3. Create a config.json file in your tool's config directory:

```json
{
  "settings": {
    "required_setting1": "value1",
    "required_setting2": "value2",
    "optional_setting1": "value3"
  }
}
```

## Tool Configuration

Tools use a two-level configuration system:

1. Global Registry (`client/src/tools/config/registry.json`):
   - Tracks enabled/disabled status of all tools
   - Example:
   ```json
   {
     "enabled_tools": {
       "your_tool_name": true
     }
   }
   ```

2. Tool-specific Config (`your_tool/config/config.json`):
   - Contains tool-specific settings
   - Must include a "settings" object
   - Settings are validated against required_config_keys
   - Example (from weather tool):
   ```json
   {
     "settings": {
       "default_location": {
         "city": "Chicago",
         "state": "IL",
         "country": "US"
       },
       "units": {
         "temperature": "fahrenheit",
         "wind_speed": "mph",
         "precipitation": "inch"
       },
       "forecast_days": 7
     }
   }
   ```

## Tool Registration and Discovery

Tools are automatically discovered and registered by the `ToolRegistry` class:

1. The registry scans the tools directory for subdirectories containing tool implementations
2. Each tool module is imported and classes ending with 'Tool' are identified
3. Tools are instantiated and registered if enabled in the registry.json
4. Tools can be enabled/disabled at runtime using the registry's enable_tool/disable_tool methods

## Tool Execution Flow

1. The LLM client receives user input and identifies tool calls in the format:
```
<tool_call>
{
    "name": "tool_name",
    "arguments": {
        "param1": "value1"
    }
}
</tool_call>
```

2. The tool registry validates and executes the tool:
   - Checks if the tool is enabled
   - Validates arguments against the tool's schema
   - Calls the tool's execute method
   - Returns the result

3. Results are processed based on the tool's llm_response setting:
   - If True: Result is sent to LLM for natural language formatting
   - If False: Result is returned directly to the user

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
   - Registry entry enables the tool:
   ```json
   {
     "enabled_tools": {
       "weather": true
     }
   }
   ```
   - Tool config defines settings:
   ```json
   {
     "settings": {
       "default_location": {
         "city": "Chicago",
         "state": "IL",
         "country": "US"
       },
       "units": {
         "temperature": "fahrenheit",
         "wind_speed": "mph",
         "precipitation": "inch"
       },
       "forecast_days": 7
     }
   }
   ```

3. Implementation Highlights:
   - Validates required configuration keys
   - Provides clear schema for location arguments
   - Implements direct text response (llm_response = False)
   - Includes comprehensive error handling
   - Uses external API (Open-Meteo) with proper error handling

## Best Practices

1. Configuration:
   - Always define required_config_keys for validation
   - Use clear, descriptive settings names
   - Include default values where appropriate
   - Document valid options for settings

2. Schema:
   - Provide clear descriptions for all parameters
   - Use appropriate JSON schema types
   - Include examples in prompt_instructions
   - Document any parameter constraints

3. Implementation:
   - Implement comprehensive error handling
   - Use async/await for I/O operations
   - Log important operations and errors
   - Return clear, actionable error messages
   - Consider whether to use LLM response formatting

4. Testing:
   - Test with various input combinations
   - Verify error handling paths
   - Test configuration validation
   - Ensure proper API error handling

## Adding New Tools Checklist

1. [ ] Create tool directory structure
2. [ ] Implement tool class inheriting from BaseTool
3. [ ] Define configuration schema and required keys
4. [ ] Create config.json with necessary settings
5. [ ] Implement execute method with proper error handling
6. [ ] Add tool to registry.json (enabled: true)
7. [ ] Test tool with various inputs
8. [ ] Document tool usage and parameters
