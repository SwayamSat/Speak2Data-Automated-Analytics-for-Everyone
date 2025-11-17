"""
LLM client abstraction layer.
Supports multiple LLM providers: Gemini, OpenAI, and Anthropic.
"""
from typing import Optional, Dict, Any, List
import json
from abc import ABC, abstractmethod

from config import config


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> str:
        """
        Generate a completion for the given prompt.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional provider-specific parameters
            
        Returns:
            The generated text
        """
        pass
    
    def structured_complete(self, prompt: str, schema: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """
        Generate a structured (JSON) completion.
        
        Args:
            prompt: The input prompt
            schema: Optional JSON schema for the response
            **kwargs: Additional parameters
            
        Returns:
            Parsed JSON response as dictionary
        """
        # Add instruction for JSON output
        json_prompt = f"{prompt}\n\nRespond with valid JSON only."
        
        response = self.complete(json_prompt, **kwargs)
        
        # Parse JSON from response
        from utils import parse_json_from_llm_response
        return parse_json_from_llm_response(response)


class GeminiClient(LLMClient):
    """Google Gemini API client."""
    
    def __init__(self, api_key: str, model_name: str):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Google API key
            model_name: Model name (e.g., 'gemini-1.5-flash')
        """
        import google.generativeai as genai
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
    
    def complete(self, prompt: str, **kwargs) -> str:
        """Generate completion using Gemini."""
        temperature = kwargs.get('temperature', 0.7)
        max_tokens = kwargs.get('max_tokens', 2000)
        
        generation_config = {
            'temperature': temperature,
            'max_output_tokens': max_tokens,
        }
        
        response = self.model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        return response.text


class OpenAIClient(LLMClient):
    """OpenAI API client."""
    
    def __init__(self, api_key: str, model_name: str):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key
            model_name: Model name (e.g., 'gpt-4o-mini')
        """
        from openai import OpenAI
        
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
    
    def complete(self, prompt: str, **kwargs) -> str:
        """Generate completion using OpenAI."""
        temperature = kwargs.get('temperature', 0.7)
        max_tokens = kwargs.get('max_tokens', 2000)
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful data analysis assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content


class AnthropicClient(LLMClient):
    """Anthropic Claude API client."""
    
    def __init__(self, api_key: str, model_name: str):
        """
        Initialize Anthropic client.
        
        Args:
            api_key: Anthropic API key
            model_name: Model name (e.g., 'claude-3-5-sonnet-20241022')
        """
        from anthropic import Anthropic
        
        self.client = Anthropic(api_key=api_key)
        self.model_name = model_name
    
    def complete(self, prompt: str, **kwargs) -> str:
        """Generate completion using Anthropic."""
        temperature = kwargs.get('temperature', 0.7)
        max_tokens = kwargs.get('max_tokens', 2000)
        
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.content[0].text


def create_llm_client(provider: Optional[str] = None, api_key: Optional[str] = None, model_name: Optional[str] = None) -> LLMClient:
    """
    Factory function to create an LLM client.
    
    Args:
        provider: LLM provider ('gemini', 'openai', 'anthropic'). Defaults to config.
        api_key: API key. Defaults to config.
        model_name: Model name. Defaults to config.
        
    Returns:
        Initialized LLM client
    """
    provider = provider or config.llm_provider
    api_key = api_key or config.get_llm_api_key()
    model_name = model_name or config.get_llm_model_name()
    
    if not api_key:
        raise ValueError(f"No API key provided for {provider}")
    
    if provider == "gemini":
        return GeminiClient(api_key, model_name)
    elif provider == "openai":
        return OpenAIClient(api_key, model_name)
    elif provider == "anthropic":
        return AnthropicClient(api_key, model_name)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
