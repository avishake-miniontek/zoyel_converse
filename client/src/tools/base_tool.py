#!/usr/bin/env python3
"""Base class for all tools."""

import json
import os
import sys
from typing import Dict, Any, Optional

class BaseTool:
    # Class-level metadata
    tool_version = "1.0.0"
    required_config_keys: list = []
    optional_config_keys: list = []

    def __init__(self):
        self.name = ""
        self.description = ""
        self.schema = {}
        self.result_prompt = ""
        self.system_role = ""
        self.config: Dict[str, Any] = {}
        self.llm_response = True  # Default to using LLM processing
        self.prompt_instructions = ""  # Instructions for the LLM on how to use this tool
        self._load_config()
    
    def _get_config_path(self) -> str:
        """Get the path to this tool's config file."""
        # Get the path of the actual tool implementation file
        tool_module = sys.modules[self.__class__.__module__]
        tool_dir = os.path.dirname(os.path.abspath(tool_module.__file__))
        return os.path.join(tool_dir, 'config', 'config.json')

    def _validate_config(self, config: dict) -> None:
        """Validate the configuration against required and optional keys."""
        if 'settings' not in config:
            raise ValueError("Config must contain a 'settings' object")
        
        settings = config['settings']
        missing_keys = [key for key in self.required_config_keys if key not in settings]
        if missing_keys:
            raise ValueError(f"Missing required config keys: {', '.join(missing_keys)}")

    def _load_config(self) -> None:
        """Load tool configuration from JSON file."""
        config_path = self._get_config_path()
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            self._validate_config(config)
            self.config = config
        except FileNotFoundError:
            raise ValueError(f"Config file not found at {config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {str(e)}")

    def _save_config(self, config: dict) -> None:
        """Save tool configuration to JSON file."""
        config_path = self._get_config_path()
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

    async def execute(self, args: dict) -> dict:
        """Execute the tool with the given arguments."""
        raise NotImplementedError("Tool must implement execute method")
    
    def format_result_prompt(self, result: dict) -> dict:
        """Format the result prompt with the tool's result"""
        return {
            "role": "system",
            "content": f"You are a {self.system_role}.\n\n{self.result_prompt}".format(
                result=json.dumps(result, indent=2)
            )
        }
