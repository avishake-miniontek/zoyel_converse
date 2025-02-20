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
from typing import List, Dict
from src.token_counter import estimate_messages_tokens
import re
from gasp import WAILGenerator
from src.tools.tool_registry import ToolRegistry

def find_sentence_boundary(text: str) -> int:
    """
    Returns the index of a valid sentence boundary in the given text,
    ignoring punctuation that is part of a decimal number.

    The regex looks for a punctuation mark (., !, or ?)
    that is not immediately preceded by a digit and is followed by whitespace or the end of the string.
    If no match is found, a fallback manual scan is used.
    """
    pattern = re.compile(r'(?<!\d)([.!?])(?=\s|$)')
    matches = list(pattern.finditer(text))
    if matches:
        return matches[-1].start()
    # Fallback: manually scan from the end.
    for i in range(len(text) - 1, -1, -1):
        if text[i] in ".!?":
            if text[i] == '.' and i > 0 and i < len(text) - 1 and text[i - 1].isdigit() and text[i + 1].isdigit():
                continue
            return i
    return -1

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
        
        # Initialize tool registry and WAIL generator
        self.tool_registry = ToolRegistry()
        self.wail_generator = WAILGenerator()
        with open(os.path.join(os.path.dirname(__file__), 'tools/tool_schema.wail'), 'r') as f:
            self.wail_generator.load_wail(f.read())
        
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

    def _generate_example_args(self, schema: dict) -> dict:
        """Generate example arguments based on a JSON schema."""
        if not schema or "properties" not in schema:
            return {}
            
        example = {}
        for prop_name, prop_schema in schema["properties"].items():
            if prop_schema.get("type") == "object" and "properties" in prop_schema:
                example[prop_name] = self._generate_example_args(prop_schema)
            elif prop_schema.get("type") == "array":
                if "items" in prop_schema:
                    if prop_schema["items"].get("type") == "object":
                        example[prop_name] = [self._generate_example_args(prop_schema["items"])]
                    else:
                        example[prop_name] = ["example_value"]
                else:
                    example[prop_name] = []
            elif prop_schema.get("type") == "string":
                example[prop_name] = prop_schema.get("description", "example_value")
            elif prop_schema.get("type") == "number":
                example[prop_name] = 0
            elif prop_schema.get("type") == "boolean":
                example[prop_name] = True
            else:
                example[prop_name] = "example_value"
        return example

    async def process_trigger(self, transcript: str, callback=None):
        """
        Process a triggered transcript with the LLM.

        Args:
            transcript: The transcript text to process.
            callback: Optional callback function to handle streaming chunks.

        Returns:
            None, as responses are handled through the callback.
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

            # No need to check trigger word since it was already handled by find_trigger_word

            # Create system prompt with assistant name from config
            # Get available and enabled tools with their schemas
            tools = self.tool_registry.list_tools()
            if tools:
                tools_description = "\n\n".join(
                    f"Tool: {name}\n"
                    f"Description: {info['description']}\n"
                    f"Usage Instructions:\n{info['prompt_instructions']}\n"
                    f"Schema:\n{json.dumps(info['schema'], indent=2)}\n"
                    f"Example:\n"
                    f"<tool_call>\n"
                    f"{{\n"
                    f"    \"name\": \"{name}\",\n"
                    f"    \"arguments\": {json.dumps(self._generate_example_args(info['schema']), indent=4)}\n"
                    f"}}\n"
                    f"</tool_call>"
                    for name, info in tools.items()
                )
            else:
                tools_description = "No tools are currently enabled."

            system_prompt = f"""You are {self.config['assistant']['name']}, a helpful AI assistant who communicates through voice. Important instructions for your responses:

1) Provide only plain text that will be converted to speech - never use markdown, asterisk *, code blocks, or special formatting.
2) Use natural, conversational language as if you're speaking to someone.
3) Never use bullet points, numbered lists, or special characters.
4) Keep responses concise and clear since they will be spoken aloud.
5) Express lists or multiple points in a natural spoken way using words like 'first', 'also', 'finally', etc.
6) Use punctuation only for natural speech pauses (periods, commas, question marks).

Available Tools:

{tools_description}

When using a tool:
1) First acknowledge the user's request naturally
2) Then make the tool call with the exact format shown in the examples above
3) Wait for the tool's response before continuing"""

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
            
            # Stream the response while collecting it and checking for tool calls
            buffer = ""
            full_response = ""
            tool_call_buffer = ""
            in_tool_call = False
            
            async for chunk in stream:
                delta = chunk.choices[0].delta
                content = None
                if delta:
                    # Try dictionary-like access; if that fails, use attribute access.
                    try:
                        content = delta.get("content", None)
                    except AttributeError:
                        content = getattr(delta, "content", None)
                if content is not None:
                    print(content, end="", flush=True)
                    buffer += content
                    full_response += content
                    
                    # Check for tool calls
                    if "<tool_call>" in buffer:
                        in_tool_call = True
                        tool_call_start = buffer.index("<tool_call>") + len("<tool_call>")
                        buffer = buffer[tool_call_start:]
                        continue
                        
                    if in_tool_call:
                        tool_call_buffer += content
                        if "</tool_call>" in tool_call_buffer:
                            # Extract the tool call JSON
                            tool_call_end = tool_call_buffer.index("</tool_call>")
                            tool_call_json = tool_call_buffer[:tool_call_end].strip()
                            
                            # Reset tool call state
                            in_tool_call = False
                            tool_call_buffer = ""
                            buffer = buffer[tool_call_end + len("</tool_call>"):].strip()
                            
                            try:
                                # Process the tool call
                                tool_call = json.loads(tool_call_json)
                                try:
                                    # Get tool first to check response type
                                    tool = self.tool_registry.get_tool(tool_call["name"])
                                    
                                    # Execute the tool
                                    result = await self.tool_registry.execute_tool(
                                        tool_call["name"],
                                        tool_call["arguments"]
                                    )
                                    
                                    formatted_response = None
                                    if tool.llm_response:
                                        # Store user message in history
                                        self.conversation_history.append(
                                            {"role": "user", "content": transcript}
                                        )
                                        
                                        # Use LLM to format response
                                        result_prompt = tool.format_result_prompt(result)
                                        response = await self.client.chat.completions.create(
                                            model=self.config["llm"]["model_name"],
                                            messages=[result_prompt],
                                            temperature=0.7,
                                            max_tokens=150
                                        )
                                        formatted_response = response.choices[0].message.content
                                        
                                        # Store LLM response in history
                                        self.conversation_history.append(
                                            {"role": "assistant", "content": formatted_response}
                                        )
                                    else:
                                        # For non-LLM responses, just send the direct result
                                        formatted_response = result["result"]
                                        
                                        # Send response through callback
                                        if callback:
                                            await callback(formatted_response)
                                        
                                        # For non-LLM responses, store history and return to prevent further processing
                                        if not tool.llm_response:
                                            # Store conversation history
                                            self.conversation_history.append(
                                                {"role": "user", "content": transcript}
                                            )
                                            self.conversation_history.append(
                                                {"role": "assistant", "content": formatted_response}
                                            )
                                            return
                                        
                                except ValueError as e:
                                    if "disabled" in str(e):
                                        if callback:
                                            await callback(f"The tool '{tool_call['name']}' is currently disabled.")
                                    else:
                                        if callback:
                                            await callback(f"Error executing tool: {str(e)}")
                                except Exception as e:
                                    if callback:
                                        await callback(f"Error executing tool: {str(e)}")
                            except json.JSONDecodeError:
                                if callback:
                                    await callback("Invalid tool call format")
                            
                            
                    # Process complete sentences from the buffer.
                    while True and not in_tool_call:
                        boundary = find_sentence_boundary(buffer)
                        if boundary == -1:
                            break
                        # If punctuation is at the very end and is likely part of a decimal, wait for more text.
                        if boundary == len(buffer) - 1 and re.search(r'\d\.$', buffer):
                            break
                        sentence = buffer[:boundary+1].strip()
                        if sentence and callback:
                            asyncio.create_task(callback(sentence))
                        buffer = buffer[boundary+1:].strip()
            # Send any remaining text without waiting
            if buffer.strip() and callback:
                asyncio.create_task(callback(buffer.strip()))
                
            # Only store non-tool responses in history
            if not in_tool_call:
                # Store user message and assistant's response in history
                self.conversation_history.append({"role": "user", "content": transcript})
                self.conversation_history.append({"role": "assistant", "content": full_response})

            # Check token count and trim history if needed
            while True:
                all_messages = [{"role": "system", "content": system_prompt}]
                all_messages.extend(self.conversation_history)
                total_tokens = estimate_messages_tokens(all_messages)
                if total_tokens <= self.config["llm"]["conversation"]["max_tokens"]:
                    break
                # Remove the oldest exchange (user + assistant messages) if over the limit.
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
