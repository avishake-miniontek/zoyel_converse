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
    
    def get_tool_prompt(self, language: str = "en") -> str:
        """
        Get prompt showing available tools in the specified language.
        Tool names and syntax remain in English for consistency.
        """
        # Tool documentation translations
        translations = {
            "en": {
                "available_tools": "Available tools:",
                "to_use_tool": "To use a tool:",
                "intro_step": "1. Write a brief intro like \"Let me check that for you\"",
                "call_step": "2. Call the tool with <tool></tool> tags",
                "stop_step": "3. STOP - do not write anything after the tool call",
                "response_step": "4. The response will come in a new message",
                "example": "Example:",
                "user_example": "User: What's the weather in Chicago?",
                "assistant_example": "Assistant: Let me check that for you.",
                "optional_arg": "?"  # Suffix for optional arguments
            },
            "es": {
                "available_tools": "Herramientas disponibles:",
                "to_use_tool": "Para usar una herramienta:",
                "intro_step": "1. Escribe una breve introducción como \"Déjame verificar eso para ti\"",
                "call_step": "2. Llama a la herramienta con etiquetas <tool></tool>",
                "stop_step": "3. DETENTE - no escribas nada después de la llamada a la herramienta",
                "response_step": "4. La respuesta vendrá en un nuevo mensaje",
                "example": "Ejemplo:",
                "user_example": "Usuario: ¿Cómo está el clima en Madrid?",
                "assistant_example": "Asistente: Déjame verificar eso para ti.",
                "optional_arg": "?"
            },
            "fr": {
                "available_tools": "Outils disponibles:",
                "to_use_tool": "Pour utiliser un outil:",
                "intro_step": "1. Écris une brève introduction comme \"Je vais vérifier cela pour toi\"",
                "call_step": "2. Appelle l'outil avec les balises <tool></tool>",
                "stop_step": "3. ARRÊTE - n'écris rien après l'appel de l'outil",
                "response_step": "4. La réponse viendra dans un nouveau message",
                "example": "Exemple:",
                "user_example": "Utilisateur: Quel temps fait-il à Paris?",
                "assistant_example": "Assistant: Je vais vérifier cela pour toi.",
                "optional_arg": "?"
            }
        }
        
        # Default to English if language not supported
        lang_dict = translations.get(language, translations["en"])
        
        lines = [f"{lang_dict['available_tools']}\n"]
        
        # Tool definitions remain in English for consistency
        for tool in self.tools.values():
            args = ", ".join(f"{arg}{lang_dict['optional_arg']}" if i > 0 else arg 
                           for i, arg in enumerate(tool.args))
            lines.append(f"{tool.name}({args}) - {tool.description}")
        
        # Usage instructions in the target language
        lines.append(f"\n{lang_dict['to_use_tool']}")
        lines.append(lang_dict['intro_step'])
        lines.append(lang_dict['call_step'])
        lines.append(lang_dict['stop_step'])
        lines.append(lang_dict['response_step'])
        
        # Example in the target language
        lines.append(f"\n{lang_dict['example']}")
        lines.append(lang_dict['user_example'])
        lines.append(lang_dict['assistant_example'])
        
        # Tool call syntax remains in English
        if language == "es":
            lines.append("<tool>weather('Madrid', '', 'Spain')</tool>")
        elif language == "fr":
            lines.append("<tool>weather('Paris', '', 'France')</tool>")
        else:
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
