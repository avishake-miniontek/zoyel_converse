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
from typing import Optional, List, Dict
from src.token_counter import estimate_messages_tokens
import re

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
        api_key = self.config["llm"].get("api_key", "not-needed")
        self.client = AsyncOpenAI(
            base_url=self.config["llm"]["api_base"],
            api_key=api_key
        )
        # Initialize conversation settings from config
        self.conversation_history: List[Dict[str, str]] = []
        self.last_message_time: float = 0
        self.context_timeout: float = self.config["llm"]["conversation"]["context_timeout"]
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from JSON file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _check_context_timeout(self) -> bool:
        """Check if the conversation context has timed out (more than 3 minutes since last message)."""
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
        
        Args:
            transcript: The transcript text to process
            callback: Optional callback function to handle streaming chunks
            
        Returns:
            None, as responses are handled through the callback
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

            # Create system prompt with assistant name from config
            system_prompt = f"""You are {self.config['assistant']['name']}, a helpful AI assistant who communicates through voice. Important instructions for your responses:

1) Provide only plain text that will be converted to speech - never use markdown, asterisk *, code blocks, or special formatting.
2) Use natural, conversational language as if you're speaking to someone.
3) Never use bullet points, numbered lists, or special characters.
4) Keep responses concise and clear since they will be spoken aloud.
5) Express lists or multiple points in a natural spoken way using words like 'first', 'also', 'finally', etc.
6) Use punctuation only for natural speech pauses (periods, commas, question marks)."""

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
            
            # Store the user's message in history before making the request
            self.conversation_history.append({"role": "user", "content": transcript})
            
            # Stream the response while collecting it
            buffer = ""
            full_response = ""
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and hasattr(delta, 'content') and delta.content is not None:
                    content = delta.content
                    print(content, end="", flush=True)
                    buffer += content
                    full_response += content
                    
                    # Use the helper to send complete sentences to TTS as they arrive.
                    while True:
                        boundary = find_sentence_boundary(buffer)
                        if boundary == -1:
                            break
                        # If the punctuation is at the very end and is likely part of a decimal (e.g. "0."), wait for more text.
                        if boundary == len(buffer) - 1 and re.search(r'\d\.$', buffer):
                            break
                        sentence = buffer[:boundary+1].strip()
                        if sentence and callback:
                            asyncio.create_task(callback(sentence))
                        buffer = buffer[boundary+1:].strip()
            # Send any remaining text without waiting
            if buffer.strip() and callback:
                asyncio.create_task(callback(buffer.strip()))
                
            # Store the assistant's response in history
            self.conversation_history.append({"role": "assistant", "content": full_response})

            # Check token count and trim history if needed
            while True:
                # Get all messages including system prompt
                all_messages = [{"role": "system", "content": system_prompt}]
                all_messages.extend(self.conversation_history)
                
                # Estimate total tokens
                total_tokens = estimate_messages_tokens(all_messages)
                
                if total_tokens <= self.config["llm"]["conversation"]["max_tokens"]:
                    break
                    
                # Remove oldest exchange (user + assistant messages) if we're over the limit
                if len(self.conversation_history) >= 2:
                    print("\n[Context Trim] Removing oldest messages to stay within token limit")
                    self.conversation_history = self.conversation_history[2:]
                else:
                    # If we somehow still exceed tokens with just the latest exchange,
                    # we have to clear everything
                    print("\n[Context Reset] Message too large, clearing history")
                    self.conversation_history = []
                    break

            # Print newline after response
            print()
            
        except Exception as e:
            print(f"Error processing LLM request: {e}")
            return None

# Example usage:
if __name__ == "__main__":
    import asyncio
    
    async def test():
        client = LLMClient()
        config = client.config
        response = await client.process_trigger(f"{config['assistant']['name']}, what's the weather like today?")
        print(f"Full response: {response}")
    
    asyncio.run(test())
