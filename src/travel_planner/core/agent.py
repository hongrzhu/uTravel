"""
Core planner agent implementation for the travel planning system.
"""

import logging
from typing import Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain_core.messages import AIMessage, SystemMessage

from ..config.settings import SYSTEM_PROMPT
from ..utils.tools import tools

class PlannerAgent:
    """Main planner agent that handles conversation and planning logic."""
    
    def __init__(self, api_key: str):
        """Initialize the planner agent with API key."""
        self.llm = self._initialize_llm(api_key)
        self.llm_with_tools = self.llm.bind_tools(tools) if self.llm else None

    def _initialize_llm(self, api_key: str) -> Optional[ChatGoogleGenerativeAI]:
        """Initialize the LLM client."""
        if not api_key:
            logging.error("Gemini API Key is missing or invalid. Cannot initialize LLM.")
            return None
            
        try:
            return ChatGoogleGenerativeAI(
                model="gemini-1.5-pro-latest",
                temperature=0.7,
                google_api_key=api_key,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )
        except Exception as e:
            logging.error(f"Failed to initialize ChatGoogleGenerativeAI: {e}")
            return None

    def process_message(self, messages: list) -> Dict[str, Any]:
        """Process a message and return the agent's response."""
        if not self.llm_with_tools:
            return {
                "messages": [AIMessage(content="Sorry, the AI system is currently unavailable.")],
                "error_message": "LLM client failed to initialize."
            }

        try:
            # Add system prompt if not present
            if not messages or not isinstance(messages[0], SystemMessage):
                messages.insert(0, SystemMessage(content=SYSTEM_PROMPT))

            # Get AI response
            ai_response = self.llm_with_tools.invoke(messages)
            return {"messages": [ai_response], "error_message": None}

        except Exception as e:
            logging.error(f"LLM invocation failed: {e}")
            return {
                "messages": [AIMessage(content=f"Sorry, an error occurred: {str(e)}")],
                "error_message": f"LLM Error: {str(e)}"
            } 