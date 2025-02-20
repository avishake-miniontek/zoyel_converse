#!/usr/bin/env python3
"""
voice_chat_agent.py

Integration module for the voice chat application using smolagents’ CodeAgent.
This module initializes a CodeAgent with a LiteLLMModel that loads model settings
from your existing config (and environment variables) and registers a custom tool:
add_two_numbers. When given a transcription, the agent will generate and execute Python code
(if a tool call is needed) and then return a final answer ready for text-to-speech.
"""

import json
import os
from smolagents import CodeAgent, LiteLLMModel
from src.tools.add_tool import add_two_numbers

class VoiceChatAgent:
    def __init__(self, config_path="config.json"):
        # Load configuration from the provided config file.
        self.config = self.load_config(config_path)
        # Override with environment variables if available.
        self.config["llm"]["model_name"] = os.getenv("MODEL_NAME", self.config["llm"]["model_name"])
        self.config["llm"]["api_key"] = os.getenv("API_SECRET_KEY", self.config["llm"]["api_key"])
        self.config["llm"]["api_base"] = os.getenv("API_BASE", self.config["llm"]["api_base"])
        
        # Adjust model name for OpenAI-compatible endpoints if needed.
        model_name = self.config["llm"]["model_name"]
        # If there is no slash, or it starts with a slash, assume a provider prefix is missing.
        if "/" not in model_name:
            model_name = "openai/" + model_name
        elif model_name.startswith("/"):
            model_name = "openai/" + model_name.lstrip("/")
        self.config["llm"]["model_name"] = model_name

        api_key = self.config["llm"]["api_key"]
        api_base = self.config["llm"]["api_base"]
        
        # Initialize the LLM model using LiteLLMModel (supports OpenAI-compatible endpoints).
        self.model = LiteLLMModel(model_id=model_name, api_key=api_key, api_base=api_base)
        
        # Create a CodeAgent with our custom add_two_numbers tool, using default system prompt
        self.agent = CodeAgent(tools=[add_two_numbers], model=self.model)
    
    def load_config(self, config_path: str) -> dict:
        """Load configuration from a JSON file."""
        with open(config_path, "r") as f:
            return json.load(f)
    
    def process_transcription(self, transcription: str) -> str:
        """
        Processes the user's transcription through the smolagents CodeAgent.
        
        The agent examines the transcription and, if a tool call is needed (for example, to add two numbers),
        the LLM generates a Python code snippet that calls the tool. The CodeAgent then executes that code and
        continues its chain-of-thought until it produces a final answer.
        
        Args:
            transcription: The user's input as plain text.
        
        Returns:
            A string representing the agent's final response.
        """
        try:
            # Let the agent process the transcription and return a response
            response = self.agent.run(transcription)
            return response
        except Exception as e:
            print(f"Error processing transcription: {str(e)}")
            return "I apologize, but I encountered an error. Could you please try again?"

# For demonstration and testing purposes.
if __name__ == "__main__":
    agent = VoiceChatAgent()
    user_input = "Mira, please add 10 and 5 for me."
    print("User Input:", user_input)
    final_response = agent.process_transcription(user_input)
    print("\nFinal Response from Agent:")
    print(final_response)
