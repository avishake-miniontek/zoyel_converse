#!/usr/bin/env python3
"""Registry for managing available tools."""

import importlib
import inspect
import logging
import os
import sys
from typing import Dict, List, Tuple, Optional

from .base_tool import BaseTool

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add a console handler if none exists
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

class ToolRegistry:
    """Registry for managing available tools."""
    
    def __init__(self):
        """Initialize tool registry."""
        self.tools: Dict[str, BaseTool] = {}
        self._discover_tools()
    
    def _discover_tools(self) -> None:
        """Discover and register available tools."""
        logger.info("[Tool Discovery] Starting tool discovery in: %s", os.path.dirname(__file__))
        
        # Walk through the tools directory
        for root, dirs, files in os.walk(os.path.dirname(__file__)):
            if "__pycache__" in root:
                continue
                
            for file in files:
                if file.endswith(".py") and not file.startswith("__"):
                    module_name = file[:-3]
                    rel_path = os.path.relpath(root, os.path.dirname(__file__))
                    if rel_path == ".":
                        continue
                    
                    full_module = ".".join(rel_path.split(os.sep) + [module_name])
                    
                    try:
                        logger.info("[Tool Discovery] Attempting to import module: %s", module_name)
                        module = importlib.import_module(f".{full_module}", package=__package__)
                        
                        for name, obj in inspect.getmembers(module):
                            if (inspect.isclass(obj) and 
                                issubclass(obj, BaseTool) and 
                                obj != BaseTool):
                                try:
                                    tool = obj()
                                    if tool.name:
                                        self.tools[tool.name] = tool
                                        logger.info("[Tool Discovery] Registered tool: %s", tool.name)
                                except Exception as e:
                                    logger.error(
                                        "[Tool Discovery] Error initializing tool %s: %s",
                                        name,
                                        str(e)
                                    )
                    except Exception as e:
                        logger.error(
                            "[Tool Discovery] Error importing module %s: %s",
                            module_name,
                            str(e)
                        )
        
        logger.info("[Tool Discovery] Completed tool discovery")
        logger.info("[Tool Discovery] Registered tools: %s", list(self.tools.keys()))
    
    def get_tool_prompt(self) -> str:
        """Get prompt showing available tools."""
        lines = ["Available tools:\n"]
        for tool in self.tools.values():
            args = ", ".join(f"{arg}?" if i > 0 else arg 
                           for i, arg in enumerate(tool.args))
            lines.append(f"{tool.name}({args}) - {tool.description}")
        lines.append("\nTo use a tool:")
        lines.append("1. Write a brief intro like \"Let me check that for you\"")
        lines.append("2. Call the tool with <tool></tool> tags")
        lines.append("3. STOP - do not write anything after the tool call")
        lines.append("4. The response will come in a new message")
        lines.append("\nExample:")
        lines.append("User: What's the weather in Chicago?")
        lines.append("Assistant: Let me check that for you.")
        lines.append("<tool>weather('Chicago', 'IL')</tool>")
        return "\n".join(lines)

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self.tools.get(name)

    async def execute_tool(self, name: str, args: List[str]) -> str:
        """Execute a tool with ordered arguments."""
        if name not in self.tools:
            raise ValueError(f"Tool '{name}' not found")
        return await self.tools[name].execute(args)
