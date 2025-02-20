#!/usr/bin/env python3
"""
OpenAI model adapter for smolagents integration.

This adapter uses the AsyncOpenAI client to generate chat completions.
"""

import asyncio
from typing import List, Dict, Any
from openai import AsyncOpenAI
from smolagents.models import ChatMessage, MessageRole, Model

class OpenAIModelAdapter(Model):
    """Adapter to use AsyncOpenAI client with smolagents."""
    
    def __init__(self, client: AsyncOpenAI, model_name: str):
        self.client = client
        self.model_name = model_name
        
    async def __call__(self, messages: List[Dict[str, Any]], **kwargs) -> ChatMessage:
        """
        Generate a response using the OpenAI API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            **kwargs: Additional arguments to pass to the API.
            
        Returns:
            ChatMessage object containing the response.
        """
        try:
            # Ensure messages are in the correct format.
            formatted_messages = [
                {"role": msg.get("role", "user"), "content": msg.get("content", "")}
                for msg in messages
            ]
            
            # Create a task for the API call.
            api_task = asyncio.create_task(
                self.client.chat.completions.create(
                    model=self.model_name,
                    messages=formatted_messages,
                    stream=False,
                    **kwargs
                )
            )
            
            # Await the API call with a timeout.
            response = await asyncio.wait_for(api_task, timeout=20.0)
            
            if not response or not response.choices:
                raise ValueError("Invalid response from OpenAI API")
            
            message = response.choices[0].message
            if not message or not hasattr(message, 'content'):
                raise ValueError("Response message missing content")
            
            # Construct and return the ChatMessage.
            chat_message = ChatMessage(
                role=MessageRole.ASSISTANT,
                content=message.content
            )
            if not hasattr(chat_message, 'content'):
                raise ValueError("Failed to create ChatMessage with content")
                
            return chat_message
                
        except asyncio.TimeoutError:
            print("OpenAI API call timed out")
            raise ValueError("API request timed out")
        except Exception as e:
            print(f"Error in OpenAIModelAdapter: {str(e)}")
            raise ValueError(f"OpenAI API error: {str(e)}")
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert model settings to a dictionary."""
        return {"model_name": self.model_name}
