#!/usr/bin/env python3
"""Base class for all tools."""

import json
import os
import sys
from typing import Dict, Any, List

class BaseTool:
    """Base class for tools with simplified argument handling."""
    
    def __init__(self):
        self.name = ""
        self.description = ""
        self.args = []  # List of argument names in order
        self.config = {}
        self.llm_response = True  # Default to using LLM processing
        self.needs_translation = True  # Whether tool output needs translation for non-English languages
        self.result_prompt = ""  # Prompt for formatting LLM responses
        self._load_config()
    
    def _get_config_path(self) -> str:
        """Get the path to this tool's config file."""
        tool_module = sys.modules[self.__class__.__module__]
        tool_dir = os.path.dirname(os.path.abspath(tool_module.__file__))
        return os.path.join(tool_dir, 'config', 'config.json')

    def _load_config(self) -> None:
        """Load tool configuration from JSON file."""
        config_path = self._get_config_path()
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            raise ValueError(f"Config file not found at {config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {str(e)}")

    async def execute(self, args: List[str]) -> str:
        """Execute the tool with the given ordered arguments."""
        raise NotImplementedError("Tool must implement execute method")
        
    def format_result_prompt(self, result: dict) -> dict:
        """Format the result prompt with the tool's result"""
        return {
            "role": "system",
            "content": f"You are a helpful assistant. Format this tool result in a natural way: {json.dumps(result, indent=2)}"
        }
