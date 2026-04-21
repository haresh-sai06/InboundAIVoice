"""
Conversation Manager
Manages conversation state for standard Gemini API (text-only)
"""

import logging
from typing import List, Dict, Optional
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class ConversationManager:
    """Manages conversation history and state for text-based AI conversation."""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash", system_instruction: str = None):
        """
        Initialize conversation manager.
        
        Args:
            api_key: Gemini API key
            model: Gemini model name
            system_instruction: System prompt for AI behavior
        """
        self.api_key = api_key
        self.model = model
        self.system_instruction = system_instruction
        self.client = genai.Client(api_key=api_key)
        self.conversation_history: List[Dict] = []
        self.max_history = 10  # Keep last 10 turns
        
        logger.info(f"ConversationManager initialized with model: {model}")
    
    def add_user_message(self, text: str):
        """Add user message to conversation history."""
        self.conversation_history.append({
            "role": "user",
            "parts": [{"text": text}]
        })
        logger.debug(f"Added user message: {text}")
    
    def add_ai_message(self, text: str):
        """Add AI message to conversation history."""
        self.conversation_history.append({
            "role": "model",
            "parts": [{"text": text}]
        })
        logger.debug(f"Added AI message: {text}")
    
    def get_ai_response(self, user_text: str) -> str:
        """
        Get AI response from standard Gemini API.
        
        Args:
            user_text: User's text input
            
        Returns:
            AI response text
        """
        try:
            # Add user message to history
            self.add_user_message(user_text)
            
            # Prepare contents with history
            contents = []
            
            # Add system instruction as first user message (standard Gemini API doesn't support system role)
            if self.system_instruction:
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part(text=f"System instruction: {self.system_instruction}")]
                ))
            
            # Add conversation history
            for msg in self.conversation_history[-self.max_history:]:
                contents.append(types.Content(
                    role=msg["role"],
                    parts=[types.Part(text=msg["parts"][0]["text"])]
                ))
            
            # Generate response
            logger.info(f"Sending conversation history to Gemini API (history length: {len(contents)})")
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.8,
                    max_output_tokens=500
                )
            )
            
            ai_response = response.text.strip() if response.text else ""
            logger.info(f"AI response: {ai_response}")
            
            # Add AI response to history
            self.add_ai_message(ai_response)
            
            return ai_response
            
        except Exception as e:
            logger.error(f"Error getting AI response: {e}")
            return "I apologize, I'm having technical difficulties. Please try again."
    
    def reset_conversation(self):
        """Reset conversation history."""
        self.conversation_history = []
        logger.info("Conversation history reset")
    
    def get_history(self) -> List[Dict]:
        """Get current conversation history."""
        return self.conversation_history
    
    def set_system_instruction(self, instruction: str):
        """Update system instruction."""
        self.system_instruction = instruction
        logger.info("System instruction updated")
