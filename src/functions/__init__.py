"""
Function registry system for dynamic function discovery and management.
Handles loading, validation, and registration of functions for LLM use.
"""

from typing import Dict, Callable, Any, Optional
import importlib
import pkgutil
import json
import inspect
from pathlib import Path
import logging
from dataclasses import dataclass
from typing import get_type_hints

logger = logging.getLogger(__name__)

@dataclass
class FunctionMetadata:
    """Metadata for a registered function"""
    description: str
    parameters: Dict[str, inspect.Parameter]
    return_type: Any
    category: str
    module: str

class FunctionRegistry:
    def __init__(self):
        self.registered_functions: Dict[str, Callable] = {}
        self.function_metadata: Dict[str, FunctionMetadata] = {}
        self.config = self._load_function_config()
        
    def _load_function_config(self) -> dict:
        """Load and validate the function configuration file."""
        try:
            # Look for config in root directory (two levels up from src/functions)
            config_path = Path(__file__).parent.parent.parent / "function_calls.json"
            if not config_path.exists():
                logger.warning("function_calls.json not found, using default configuration")
                return {
                    "enabled_categories": [],
                    "excluded_functions": [],
                    "function_defaults": {"timeout": 30}
                }
                
            with open(config_path) as f:
                config = json.load(f)
                
            # Validate required fields
            required_fields = ["enabled_categories", "excluded_functions"]
            for field in required_fields:
                if field not in config:
                    logger.warning(f"Missing required field '{field}' in function_calls.json")
                    config[field] = []
                    
            return config
            
        except Exception as e:
            logger.error(f"Error loading function configuration: {e}")
            return {
                "enabled_categories": [],
                "excluded_functions": [],
                "function_defaults": {"timeout": 30}
            }

    def register_function(self, func: Callable, metadata: Optional[dict] = None) -> bool:
        """
        Register a function with optional metadata.
        Returns True if registration was successful, False otherwise.
        """
        try:
            # Get function category from module path
            module_parts = func.__module__.split('.')
            if len(module_parts) < 4:  # src.functions.category.module
                logger.warning(f"Invalid module path for function {func.__name__}")
                return False
                
            category = module_parts[-2]  # Get category from module path
            
            # Check if category is enabled
            if category not in self.config.get("enabled_categories", []):
                logger.debug(f"Category '{category}' not enabled, skipping {func.__name__}")
                return False
                
            # Generate full function name
            func_name = f"{category}.{func.__name__}"
            
            # Check if function is explicitly excluded
            if func_name in self.config.get("excluded_functions", []):
                logger.debug(f"Function '{func_name}' is excluded, skipping registration")
                return False
                
            # Get function signature and return type
            sig = inspect.signature(func)
            type_hints = get_type_hints(func)
            return_type = type_hints.get('return', Any)
            
            # Create metadata
            self.function_metadata[func_name] = FunctionMetadata(
                description=func.__doc__ or "",
                parameters=sig.parameters,
                return_type=return_type,
                category=category,
                module=func.__module__
            )
            
            # Store the function
            self.registered_functions[func_name] = func
            logger.info(f"Successfully registered function '{func_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error registering function {func.__name__}: {e}")
            return False

    def discover_functions(self):
        """
        Automatically discover and register functions from all modules in the functions directory.
        """
        try:
            functions_dir = Path(__file__).parent
            
            # Iterate through all categories (subdirectories)
            for category_path in functions_dir.iterdir():
                if not category_path.is_dir() or category_path.name.startswith('_'):
                    continue
                    
                category = category_path.name
                if category not in self.config.get("enabled_categories", []):
                    logger.debug(f"Skipping disabled category: {category}")
                    continue
                
                # Import the category's functions module
                try:
                    module = importlib.import_module(f".{category}.functions", package="src.functions")
                    
                    # Look for and call register() function in the module
                    if hasattr(module, "register"):
                        module.register(self)
                    else:
                        logger.warning(f"No register() function found in {category}.functions")
                        
                except ImportError as e:
                    logger.error(f"Error importing functions from category {category}: {e}")
                
        except Exception as e:
            logger.error(f"Error during function discovery: {e}")

    def get_openai_schema(self) -> list[dict]:
        """
        Convert registered functions to OpenAI function call schema format.
        """
        schema = []
        for func_name, func in self.registered_functions.items():
            metadata = self.function_metadata[func_name]
            
            # Convert parameters to OpenAI schema
            parameters = {}
            required = []
            
            for name, param in metadata.parameters.items():
                param_type = get_type_hints(func).get(name, Any)
                
                # Skip self parameter for methods
                if name == 'self':
                    continue
                    
                param_schema = {"type": "string"}  # Default to string
                
                # Basic type mapping
                if param_type in (int, float):
                    param_schema["type"] = "number"
                elif param_type == bool:
                    param_schema["type"] = "boolean"
                elif param_type == list:
                    param_schema["type"] = "array"
                    param_schema["items"] = {"type": "string"}
                
                parameters[name] = param_schema
                
                # Check if parameter is required
                if param.default == inspect.Parameter.empty:
                    required.append(name)
            
            schema.append({
                "name": func_name,
                "description": metadata.description,
                "parameters": {
                    "type": "object",
                    "properties": parameters,
                    "required": required
                }
            })
            
        return schema

# Create global registry instance
registry = FunctionRegistry()

def initialize():
    """Initialize the function registry and discover functions."""
    registry.discover_functions()
    return registry