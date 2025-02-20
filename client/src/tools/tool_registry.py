#!/usr/bin/env python3
"""Registry for managing available tools."""

import importlib
import json
import os
import pkgutil
from typing import Dict, Type
from .base_tool import BaseTool

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._registry_path = os.path.join(os.path.dirname(__file__), 'config', 'registry.json')
        self._registry = self._load_registry()
        self._discover_and_register_tools()
    
    def _load_registry(self) -> dict:
        """Load tool registry from JSON file."""
        try:
            with open(self._registry_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Create default registry if it doesn't exist
            default_registry = {"enabled_tools": {}}
            self._save_registry(default_registry)
            return default_registry

    def _save_registry(self, registry: dict):
        """Save tool registry to JSON file."""
        os.makedirs(os.path.dirname(self._registry_path), exist_ok=True)
        with open(self._registry_path, 'w') as f:
            json.dump(registry, f, indent=2)

    def _discover_and_register_tools(self):
        """Discover and register tools from the tools directory."""
        tools_dir = os.path.dirname(__file__)
        print(f"\n[Tool Discovery] Starting tool discovery in: {tools_dir}")
        
        def scan_directory(directory):
            """Recursively scan directory for tool modules."""
            print(f"[Tool Discovery] Scanning directory: {directory}")
            for item in os.listdir(directory):
                if item == "__pycache__" or item.startswith('_'):
                    continue
                    
                item_path = os.path.join(directory, item)
                if os.path.isdir(item_path):
                    if os.path.exists(os.path.join(item_path, '__init__.py')):
                        try:
                            # Get relative path from tools directory
                            rel_path = os.path.relpath(item_path, tools_dir)
                            # Convert path to module notation
                            module_path = rel_path.replace(os.path.sep, '.')
                            print(f"[Tool Discovery] Attempting to import module: {module_path}")
                            # Import the module
                            module = importlib.import_module(f".{module_path}", package=__package__)
                            print(f"[Tool Discovery] Successfully imported module: {module_path}")
                            
                            # Look for Tool classes
                            for attr_name in dir(module):
                                if attr_name.endswith('Tool') and attr_name != 'BaseTool':
                                    print(f"[Tool Discovery] Found tool class: {attr_name}")
                                    tool_class = getattr(module, attr_name)
                                    if isinstance(tool_class, type) and issubclass(tool_class, BaseTool):
                                        try:
                                            print(f"[Tool Discovery] Initializing {attr_name}")
                                            tool = tool_class()
                                            print(f"[Tool Discovery] Tool name: {tool.name}")
                                            print(f"[Tool Discovery] Tool enabled status: {self.is_tool_enabled(tool.name)}")
                                            if tool.name and self.is_tool_enabled(tool.name):
                                                self.register_tool(tool)
                                                print(f"[Tool Discovery] Successfully registered tool: {tool.name}")
                                        except Exception as e:
                                            print(f"[Tool Discovery] Error initializing {attr_name}: {str(e)}")
                        except ImportError as e:
                            print(f"[Tool Discovery] Error importing {module_path}: {str(e)}")
                    # Recursively scan subdirectories
                    scan_directory(item_path)
        
        scan_directory(tools_dir)
        print("[Tool Discovery] Completed tool discovery")
        print(f"[Tool Discovery] Registered tools: {list(self._tools.keys())}")

    def is_tool_enabled(self, name: str) -> bool:
        """Check if a tool is enabled in the registry."""
        return self._registry.get("enabled_tools", {}).get(name, True)
    
    def register_tool(self, tool: BaseTool) -> None:
        """Register a new tool."""
        if not tool.name:
            raise ValueError("Tool must have a name")
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name} is already registered")
        
        # Update registry if tool not present
        if tool.name not in self._registry.get("enabled_tools", {}):
            if "enabled_tools" not in self._registry:
                self._registry["enabled_tools"] = {}
            self._registry["enabled_tools"][tool.name] = True
            self._save_registry(self._registry)
        
        self._tools[tool.name] = tool
    
    def get_tool(self, name: str) -> BaseTool:
        """Get a tool by name."""
        if not self.is_tool_enabled(name):
            raise ValueError(f"Tool {name} is disabled")
        if name not in self._tools:
            raise ValueError(f"Tool {name} not found")
        return self._tools[name]
    
    def list_tools(self) -> Dict[str, dict]:
        """List all registered and enabled tools with their schemas and instructions."""
        return {
            name: {
                "description": tool.description,
                "schema": tool.schema,
                "prompt_instructions": tool.prompt_instructions,
                "enabled": self.is_tool_enabled(name)
            }
            for name, tool in self._tools.items()
            if self.is_tool_enabled(name)
        }

    def enable_tool(self, name: str) -> None:
        """Enable a tool in the registry."""
        if "enabled_tools" not in self._registry:
            self._registry["enabled_tools"] = {}
        self._registry["enabled_tools"][name] = True
        self._save_registry(self._registry)
        # Re-discover and register tools
        self._discover_and_register_tools()

    def disable_tool(self, name: str) -> None:
        """Disable a tool in the registry."""
        if "enabled_tools" not in self._registry:
            self._registry["enabled_tools"] = {}
        self._registry["enabled_tools"][name] = False
        self._save_registry(self._registry)
        # Remove the tool from registry if it exists
        if name in self._tools:
            del self._tools[name]
    
    async def execute_tool(self, name: str, args: dict) -> dict:
        """Execute a tool by name with the given arguments."""
        tool = self.get_tool(name)
        return await tool.execute(args)
