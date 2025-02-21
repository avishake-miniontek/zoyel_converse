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
from typing import List, Dict, Tuple, Optional
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
    ignoring punctuation that is part of a decimal number.
    """
    pattern = re.compile(r'(?<!\d)([.!?])(?=\s|$)')
    matches = list(pattern.finditer(text))
    if matches:
        return matches[-1].start()
    # Fallback: manually scan from the end
    for i in range(len(text) - 1, -1, -1):
        if text[i] in ".!?":
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

    async def process_trigger(self, transcript: str, callback=None):
        """
        Process a triggered transcript with the LLM.
        """
        try:
            # Check for manual reset trigger
            if "reset context" in transcript.lower():
                self.reset_context()
                if callback:
                    await callback("Context has been reset. How can I help you?")
                return

            # Check for timeout and reset if needed
            if self._check_context_timeout():
                self.reset_context()
                print("\n[Context Timeout] Conversation history cleared due to inactivity.")

            # Update last message time
            self.last_message_time = time.time()

            # Create system prompt with assistant name and tool instructions
            system_prompt = f"""You are {self.config['assistant']['name']}, a helpful AI assistant who communicates through voice. Important instructions for your responses:

1) Provide only plain text that will be converted to speech - never use markdown, asterisk *, code blocks, or special formatting.
2) Use natural, conversational language as if you're speaking to someone.
3) Never use bullet points, numbered lists, or special characters.
4) Keep responses concise and clear since they will be spoken aloud.
5) Express lists or multiple points in a natural spoken way using words like 'first', 'also', 'finally', etc.
6) Use punctuation only for natural speech pauses (periods, commas, question marks).

Available tools:

weather(city, state?, country?) - Get weather information for a location

To use a tool:
1. Write a brief intro like "Let me check that for you"
2. Call the tool with <tool></tool> tags
3. STOP - do not write anything after the tool call
4. The response will come in a new message

Example:
User: What's the weather in Chicago?
Assistant: Let me check that for you.
<tool>weather('Chicago', 'IL')</tool>"""

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
                                    await callback(formatted_response)
                            else:
                                # Use direct response
                                self.conversation_history.append(
                                    {"role": "assistant", "content": result}
                                )
                                if callback:
                                    await callback(result)
                            
                            # Clear buffer and return since we've handled the tool call
                            buffer = ""
                            return
                            
                        except ValueError as e:
                            logger.error(f"Tool execution error: {str(e)}")
                            if callback:
                                await callback(str(e))
                            buffer = ""
                            return
                    
                    # Process complete sentences from the buffer
                    while True:
                        boundary = find_sentence_boundary(buffer)
                        if boundary == -1:
                            # If no sentence boundary found and we've received the last chunk,
                            # force flush the remaining buffer with a period
                            if chunk.choices[0].finish_reason is not None and buffer.strip():
                                sentence = buffer.strip()
                                if not sentence.endswith(('.', '!', '?')):
                                    sentence += "."
                                if callback:
                                    asyncio.create_task(callback(sentence))
                                buffer = ""
                            break
                        sentence = buffer[:boundary+1].strip()
                        if sentence and callback:
                            asyncio.create_task(callback(sentence))
                        buffer = buffer[boundary+1:].strip()
            
            # Send any remaining text with proper punctuation
            if buffer.strip() and callback:
                sentence = buffer.strip()
                if not sentence.endswith(('.', '!', '?')):
                    sentence += "."
                asyncio.create_task(callback(sentence))
                
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
