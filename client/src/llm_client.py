#!/usr/bin/env python3
"""
LLM Client for handling interactions with vLLM server using OpenAI-compatible API.
Maintains conversation context with automatic timeout and manual reset capabilities.
"""

import json
import os
import sys
import asyncio
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
import time
from openai import AsyncOpenAI
from typing import List, Dict, Tuple, Optional, Any
from src.token_counter import estimate_messages_tokens
import re
import logging
from src.tools.tool_registry import ToolRegistry

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add console handler if none exists
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def find_sentence_boundary(text: str) -> int:
    """
    Returns the index of a valid sentence boundary in the given text,
    supporting multiple languages (Latin, Chinese, Japanese, Hindi),
    while ignoring punctuation that is part of a decimal number.
    """
    # Include Latin, Chinese, Japanese, and Hindi sentence-ending punctuation
    pattern = re.compile(r'(?<!\d)([.!?。！？।॥…」])(?=\s|$|」)')
    matches = list(pattern.finditer(text))
    if matches:
        return matches[-1].start()
    # Fallback: manually scan from the end
    for i in range(len(text) - 1, -1, -1):
        if text[i] in ".!?。！？।॥…」":
            # If at the end of the string, check if it might be part of a decimal
            if i == len(text) - 1 and i > 0 and text[i - 1].isdigit():
                continue  # might be incomplete decimal
            # Skip if it's a decimal like 3.14
            if text[i] == '.' and i > 0 and i < len(text) - 1 and text[i - 1].isdigit() and text[i + 1].isdigit():
                continue
            return i
    return -1


def parse_tool_call(text: str) -> Optional[Tuple[str, List[str]]]:
    """Extract and parse tool call from text."""
    logger.debug(f"Attempting to parse tool call from text: {text}")
    
    # Find content between <tool> tags
    match = re.search(r'<tool>(.*?)</tool>', text)
    if not match:
        logger.debug("No <tool> tags found in text")
        return None
    
    logger.debug(f"Found tool tags, content: {match.group(1)}")
    
    # Parse function call: name(args)
    call = match.group(1).strip()
    name_match = re.match(r'(\w+)\s*\((.*)\)', call)
    if not name_match:
        logger.debug("Failed to parse function call syntax")
        return None
    
    tool_name = name_match.group(1)
    args_str = name_match.group(2)
    
    # Split args by comma and strip quotes
    args = [arg.strip().strip("'\"") for arg in args_str.split(',') if arg.strip()]
    logger.debug(f"Parsed tool name: {tool_name}, args: {args}")
    
    return tool_name, args


class LLMClient:
    """Client for interacting with vLLM server using OpenAI-compatible API."""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize LLM client with configuration."""
        self.config = self._load_config(config_path)
        # Use API key from config, defaulting to "not-needed" if not present
        self.config['llm']['model_name'] = os.getenv('MODEL_NAME', self.config['llm']['model_name'])
        self.config['llm']['api_key'] = os.getenv('API_SECRET_KEY', self.config['llm']['api_key'])
        self.config['llm']['api_base'] = os.getenv('API_BASE', self.config['llm']['api_base'])

        api_key = self.config["llm"].get("api_key", "not-needed")
        self.client = AsyncOpenAI(
            base_url=self.config["llm"]["api_base"],
            api_key=api_key
        )
        # Initialize conversation settings from config
        self.conversation_history: List[Dict[str, str]] = []
        self.last_message_time: float = 0
        self.context_timeout: float = self.config["llm"]["conversation"]["context_timeout"]
        
        # Initialize tool registry
        self.tool_registry = ToolRegistry()
        
        # Load prompt based on configuration
        self.prompt_data = self._load_prompt()
        logger.info(f"Loaded prompt for language: {self.prompt_data.get('language_name', 'unknown')}")

    @property
    def conversation_history(self) -> List[Dict[str, str]]:
        return self._conversation_history

    @conversation_history.setter
    def conversation_history(self, history: List[Dict[str, str]]):
        self._conversation_history = history
        self._sanitize_conversation_history()
        logger.debug(f"Conversation history updated and sanitized with {len(self._conversation_history)} messages")
        
    def _load_prompt(self) -> Dict[str, Any]:
        """
        Load the appropriate prompt based on configuration.
        First tries to load a custom prompt if specified, then falls back to the language-specific prompt,
        and finally defaults to English if neither is available.
        """
        prompt_config = self.config["llm"].get("prompt", {})
        language = prompt_config.get("language", "en")
        custom_path = prompt_config.get("custom_path")
        prompts_dir = prompt_config.get("directory", "prompts")
        
        # Try to load custom prompt if specified
        if custom_path:
            try:
                custom_prompt_path = custom_path
                if not os.path.isabs(custom_prompt_path):
                    # If relative path, resolve from current directory
                    custom_prompt_path = os.path.join(os.getcwd(), custom_prompt_path)
                
                if os.path.exists(custom_prompt_path):
                    logger.info(f"Loading custom prompt from: {custom_prompt_path}")
                    with open(custom_prompt_path, 'r', encoding='utf-8') as f:
                        prompt_data = json.load(f)
                    return self._validate_prompt(prompt_data, "custom")
                else:
                    logger.warning(f"Custom prompt file not found: {custom_prompt_path}")
            except Exception as e:
                logger.error(f"Error loading custom prompt: {e}")
        
        # Try to load language-specific prompt
        try:
            # First check if the prompts directory exists relative to the current directory
            base_prompts_dir = os.path.join(os.getcwd(), prompts_dir)
            if not os.path.exists(base_prompts_dir):
                # If not, try relative to the script directory
                script_dir = os.path.dirname(os.path.abspath(__file__))
                base_dir = os.path.dirname(script_dir)  # Go up one level from src/
                base_prompts_dir = os.path.join(base_dir, prompts_dir)
            
            # Try to load the language-specific prompt
            lang_prompt_path = os.path.join(base_prompts_dir, "default", f"{language}.json")
            if os.path.exists(lang_prompt_path):
                logger.info(f"Loading language prompt from: {lang_prompt_path}")
                with open(lang_prompt_path, 'r', encoding='utf-8') as f:
                    prompt_data = json.load(f)
                return self._validate_prompt(prompt_data, language)
            else:
                logger.warning(f"Language prompt file not found: {lang_prompt_path}")
        except Exception as e:
            logger.error(f"Error loading language prompt: {e}")
        
        # Fall back to English if available
        if language != "en":
            try:
                en_prompt_path = os.path.join(base_prompts_dir, "default", "en.json")
                if os.path.exists(en_prompt_path):
                    logger.info(f"Falling back to English prompt: {en_prompt_path}")
                    with open(en_prompt_path, 'r', encoding='utf-8') as f:
                        prompt_data = json.load(f)
                    return self._validate_prompt(prompt_data, "en")
            except Exception as e:
                logger.error(f"Error loading fallback English prompt: {e}")
        
        # If all else fails, return a default prompt structure
        logger.warning("Using hardcoded default prompt as fallback")
        return {
            "system_prompt": "You are {assistant_name}, a helpful AI assistant who communicates through voice. Keep your responses conversational and avoid using formatting that doesn't work in speech.",
            "language": "en",
            "language_name": "English (Default Fallback)"
        }
    
    def _validate_prompt(self, prompt_data: Dict[str, Any], language: str) -> Dict[str, Any]:
        """Validate the prompt data structure and ensure it has all required fields."""
        required_fields = ["system_prompt"]
        for field in required_fields:
            if field not in prompt_data:
                logger.warning(f"Prompt is missing required field: {field}")
                prompt_data[field] = "You are {assistant_name}, a helpful AI assistant."
        
        # Ensure language fields are set
        if "language" not in prompt_data:
            prompt_data["language"] = language
        if "language_name" not in prompt_data:
            language_names = {
                "en": "English",
                "es": "Spanish",
                "fr": "French",
                "de": "German",
                "it": "Italian",
                "pt": "Portuguese",
                "custom": "Custom"
            }
            prompt_data["language_name"] = language_names.get(language, "Unknown")
            
        return prompt_data
    
    def get_system_prompt(self) -> str:
        """
        Get the system prompt with variables replaced and dynamic tool documentation.
        """
        system_prompt = self.prompt_data.get("system_prompt", "")
        
        # Get current language
        current_language = self.prompt_data.get("language", "en")
        
        # Replace variables in the prompt
        system_prompt = system_prompt.replace("{assistant_name}", self.config['assistant']['name'])
        
        # Check if we need to replace tool documentation
        # Look for hardcoded tool documentation in the prompt
        tool_doc_markers = [
            "Available tools:", 
            "Herramientas disponibles:", 
            "Outils disponibles:"
        ]
        
        # If any tool documentation marker is found, replace the entire tool section
        for marker in tool_doc_markers:
            if marker in system_prompt:
                # Generate dynamic tool documentation
                tool_docs = self.tool_registry.get_tool_prompt(current_language)
                
                # Find the start of the tool documentation
                start_idx = system_prompt.find(marker)
                if start_idx == -1:
                    continue
                
                # Find the end of the tool documentation (look for the next empty line after an example)
                example_markers = ["<tool>", "</tool>"]
                for example_marker in example_markers:
                    example_idx = system_prompt.find(example_marker, start_idx)
                    if example_idx != -1:
                        end_idx = system_prompt.find("\n\n", example_idx)
                        if end_idx == -1:  # If no empty line after example, use the end of the string
                            end_idx = len(system_prompt)
                        
                        # Replace the tool documentation section
                        system_prompt = system_prompt[:start_idx] + tool_docs + system_prompt[end_idx:]
                        break
                
                # Only need to replace once
                break
        
        return system_prompt
    
    async def translate_tool_response(self, text: str, target_language: str) -> str:
        """Translate tool response to target language."""
        if target_language == "en":
            return text
            
        logger.info(f"Translating tool response to {target_language}")
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise translator. Translate the following text to "
                    f"{target_language}, preserving all measurements, numbers, and "
                    "proper nouns exactly as they appear."
                )
            },
            {
                "role": "user",
                "content": text
            }
        ]
        
        try:
            response = await self.client.chat.completions.create(
                model=self.config["llm"]["model_name"],
                messages=messages,
                temperature=0.3  # Lower temperature for more consistent translations
            )
            
            translated_text = response.choices[0].message.content
            logger.debug(f"Translated text: {translated_text}")
            return translated_text
        except Exception as e:
            logger.error(f"Translation error: {e}")
            # Return original text if translation fails
            return text
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from JSON file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _check_context_timeout(self) -> bool:
        """Check if the conversation context has timed out."""
        if not self.last_message_time:
            return False
        return (time.time() - self.last_message_time) > self.context_timeout

    def reset_context(self):
        """Manually reset the conversation context."""
        self.conversation_history = []
        self.last_message_time = 0
        print("\n[Context Reset] Conversation history cleared.")

    def _sanitize_conversation_history(self):
        """
        Ensures conversation history alternates roles correctly.
        If consecutive messages from the same role are found,
        merge their content to maintain conversation flow.
        """
        if not self._conversation_history:
            return
        sanitized = [self._conversation_history[0].copy()]
        merged_count = 0
        for msg in self._conversation_history[1:]:
            if msg['role'] == sanitized[-1]['role']:
                sanitized[-1]['content'] += " " + msg['content']
                merged_count += 1
            else:
                sanitized.append(msg.copy())
        self._conversation_history = sanitized
        if merged_count > 0:
            logger.warning(f"Sanitized conversation history: merged {merged_count} consecutive messages")

    async def process_trigger(self, transcript: str, callback=None):
        """
        Process a triggered transcript with the LLM.
        """
        try:
            # Check for manual reset trigger
            if "reset context" in transcript.lower():
                self.reset_context()
                if callback:
                    reset_msg = "Context has been reset. How can I help you?"
                    logger.info(f"TTS sent: {reset_msg}")
                    await callback(reset_msg)
                return

            # Check for timeout and reset if needed
            if self._check_context_timeout():
                self.reset_context()
                print("\n[Context Timeout] Conversation history cleared due to inactivity.")

            # Update last message time
            self.last_message_time = time.time()

            # Get system prompt from loaded prompt data
            system_prompt = self.get_system_prompt()
            logger.info(f"Using {self.prompt_data.get('language_name', 'unknown')} prompt")

            # Sanitize conversation history to fix role alternation issues
            # self._sanitize_conversation_history()

            # Prepare messages with conversation history
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(self.conversation_history)
            messages.append({"role": "user", "content": transcript})

            # Log full context for debugging
            print("\n[Context Debug] Full conversation context being sent to server:")
            for idx, msg in enumerate(messages):
                print(f"\n{idx}. Role: {msg['role']}")
                print(f"   Content: {msg['content']}")
            print("\n[End Context Debug]")

            # Create chat completion request with streaming
            stream = await self.client.chat.completions.create(
                model=self.config["llm"]["model_name"],
                messages=messages,
                temperature=self.config["llm"]["conversation"]["temperature"],
                max_tokens=self.config["llm"]["conversation"]["response_max_tokens"],
                stream=True
            )
            
            # Stream the response while collecting it
            buffer = ""
            full_response = ""

            async for chunk in stream:
                delta = chunk.choices[0].delta
                content = None
                if delta:
                    try:
                        content = delta.get("content", None)
                    except AttributeError:
                        content = getattr(delta, "content", None)
                if content is not None:
                    print(content, end="", flush=True)
                    buffer += content
                    full_response += content
                    
                    # Look for tool calls in the buffer
                    tool_result = parse_tool_call(buffer)
                    if tool_result:
                        tool_name, args = tool_result
                        logger.debug(f"Found tool call: {tool_name} with args: {args}")
                        try:
                            tool = self.tool_registry.get_tool(tool_name)
                            if not tool:
                                raise ValueError(f"Tool '{tool_name}' not found")

                            result = await self.tool_registry.execute_tool(tool_name, args)
                            logger.debug(f"Tool execution result: {result}")
                            
                            # Store user message and assistant's tool call
                            self.conversation_history.append({"role": "user", "content": transcript})
                            self.conversation_history.append({"role": "assistant", "content": full_response})
                            
                            # Get current language from prompt data
                            current_language = self.prompt_data.get("language", "en")
                            
                            # Handle tool response based on llm_response flag
                            if tool.llm_response:
                                # Use LLM to format response
                                result_prompt = tool.format_result_prompt({"result": result})
                                response = await self.client.chat.completions.create(
                                    model=self.config["llm"]["model_name"],
                                    messages=[result_prompt],
                                    temperature=0.7,
                                    max_tokens=150
                                )
                                formatted_response = response.choices[0].message.content
                                # Store LLM-formatted response in history
                                self.conversation_history.append(
                                    {"role": "assistant", "content": formatted_response}
                                )
                                if callback:
                                    logger.info(f"TTS sent (LLM-formatted tool response): {formatted_response}")
                                    await callback(formatted_response)
                            else:
                                # Check if translation is needed
                                final_result = result
                                if tool.needs_translation and current_language != "en":
                                    final_result = await self.translate_tool_response(result, current_language)
                                
                                # Use direct response
                                self.conversation_history.append(
                                    {"role": "assistant", "content": final_result}
                                )
                                if callback:
                                    logger.info(f"TTS sent (tool response): {final_result}")
                                    await callback(final_result)
                            
                            # Clear buffer and return since we've handled the tool call
                            buffer = ""
                            return
                        except ValueError as e:
                            logger.error(f"Tool execution error: {str(e)}")
                            if callback:
                                error_msg = str(e)
                                logger.info(f"TTS sent (tool error): {error_msg}")
                                await callback(error_msg)
                            buffer = ""
                            return
                    
                    # Process complete sentences from the buffer
                    while True:
                        boundary = find_sentence_boundary(buffer)
                        if boundary == -1:
                            # No complete sentence in buffer, keep reading
                            break
                        sentence = buffer[:boundary+1].strip()
                        if sentence and callback:
                            logger.info(f"TTS sent (partial sentence): {sentence}")
                            await callback(sentence)
                        buffer = buffer[boundary+1:].strip()
                
                # If we detect the final chunk, break from the loop
                if chunk.choices[0].finish_reason is not None:
                    break

            # After streaming completes: flush any leftover text once
            if buffer.strip() and callback:
                sentence = buffer.strip()
                if not sentence.endswith(('.', '!', '?')):
                    sentence += "."
                logger.info(f"TTS sent (final leftover): {sentence}")
                await callback(sentence)
                
            # Store conversation history if no tool was called
            if not parse_tool_call(full_response):
                self.conversation_history.append({"role": "user", "content": transcript})
                self.conversation_history.append({"role": "assistant", "content": full_response})

            # Check token count and trim history if needed
            while True:
                all_messages = [{"role": "system", "content": system_prompt}]
                all_messages.extend(self.conversation_history)
                total_tokens = estimate_messages_tokens(all_messages)
                if total_tokens <= self.config["llm"]["conversation"]["max_tokens"]:
                    break
                if len(self.conversation_history) >= 2:
                    print("\n[Context Trim] Removing oldest messages to stay within token limit")
                    self.conversation_history = self.conversation_history[2:]
                else:
                    print("\n[Context Reset] Message too large, clearing history")
                    self.conversation_history = []
                    break

            print()

        except Exception as e:
            print(f"Error processing LLM request: {e}")
            return None


# Example usage:
if __name__ == "__main__":
    async def test():
        client = LLMClient()
        config = client.config
        await client.process_trigger(f"{config['assistant']['name']}, what's the weather like today?")
    
    asyncio.run(test())
